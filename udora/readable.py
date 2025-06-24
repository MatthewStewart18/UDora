"""
Simple Readable Adversarial String Optimization for UDora

This module implements a straightforward approach for generating readable adversarial strings
by optimizing token replacements based on perplexity scores from the target model.

The approach assumes users provide a readable initial string, then optimizes it by:
1. Computing perplexity scores for candidate token replacements
2. Preferring tokens that maintain lower perplexity (more natural language)
3. Balancing attack effectiveness with linguistic naturalness
"""

from typing import List, Dict, Optional
import torch
from torch import Tensor
import torch.nn.functional as F
import logging

logger = logging.getLogger("UDora")


class ReadableOptimizer:
    """
    Simple optimizer for readable adversarial strings using perplexity guidance.
    
    This class optimizes adversarial strings by considering the perplexity of
    candidate token replacements, favoring those that maintain natural language
    patterns while still being effective for the attack.
    """
    
    def __init__(self, tokenizer, model):
        """
        Initialize the readable optimizer.
        
        Args:
            tokenizer: Model tokenizer for text processing
            model: Language model for computing perplexity scores
        """
        self.tokenizer = tokenizer
        self.model = model
        
        # Cache for perplexity computations
        self._perplexity_cache = {}
        
    def compute_perplexity_bonus(self, 
                               current_ids: Tensor,
                               position_idx: int,
                               candidate_tokens: Tensor,
                               strength: float = 0.3) -> Tensor:
        """
        Compute perplexity-based bonus for candidate tokens at a specific position.
        
        Args:
            current_ids: Current token sequence, shape (seq_len,)
            position_idx: Position where we're considering replacements
            candidate_tokens: Candidate token IDs to evaluate, shape (num_candidates,)
            strength: Strength of perplexity guidance (0.0 to 1.0)
            
        Returns:
            Perplexity bonus for each candidate token, shape (num_candidates,)
        """
        if strength == 0.0:
            return torch.zeros_like(candidate_tokens, dtype=torch.float, device=current_ids.device)
        
        # Create context for perplexity evaluation
        context_window = 10  # Use surrounding tokens for context
        start_idx = max(0, position_idx - context_window)
        end_idx = min(len(current_ids), position_idx + context_window + 1)
        
        # Extract context before and after the position
        # Ensure position_idx is within bounds
        if position_idx >= len(current_ids):
            position_idx = len(current_ids) - 1
        if position_idx < 0:
            position_idx = 0
            
        context_before = current_ids[start_idx:position_idx]
        context_after = current_ids[position_idx + 1:end_idx]
        
        perplexity_scores = []
        
        with torch.no_grad():
            for candidate_token in candidate_tokens:
                # Create sequence with candidate token
                # Handle empty context gracefully
                sequence_parts = []
                if len(context_before) > 0:
                    sequence_parts.append(context_before)
                sequence_parts.append(candidate_token.unsqueeze(0))
                if len(context_after) > 0:
                    sequence_parts.append(context_after)
                
                test_sequence = torch.cat(sequence_parts)
                
                # Compute perplexity for this sequence
                perplexity = self._compute_sequence_perplexity(test_sequence)
                perplexity_scores.append(perplexity)
        
        perplexity_scores = torch.tensor(perplexity_scores, device=current_ids.device, dtype=torch.float)
        
        # Convert perplexity to bonus (lower perplexity = higher bonus)
        # Use negative log perplexity and normalize
        max_perplexity = perplexity_scores.max()
        min_perplexity = perplexity_scores.min()
        
        if max_perplexity > min_perplexity:
            # Normalize and invert (lower perplexity gets higher bonus)
            normalized_scores = (max_perplexity - perplexity_scores) / (max_perplexity - min_perplexity)
        else:
            # All perplexities are the same
            normalized_scores = torch.ones_like(perplexity_scores)
        
        # Scale by strength
        perplexity_bonus = normalized_scores * strength
        
        return perplexity_bonus
    
    def _compute_sequence_perplexity(self, token_ids: Tensor) -> float:
        """
        Compute perplexity for a token sequence.
        
        Args:
            token_ids: Token sequence, shape (seq_len,)
            
        Returns:
            Perplexity score (lower is better)
        """
        # Create cache key
        cache_key = tuple(token_ids.tolist())
        if cache_key in self._perplexity_cache:
            return self._perplexity_cache[cache_key]
        
        if len(token_ids) < 2:
            # Cannot compute perplexity for sequences shorter than 2 tokens
            self._perplexity_cache[cache_key] = 100.0  # High perplexity for invalid sequences
            return 100.0
        
        try:
            # Add batch dimension and ensure same device as model
            input_ids = token_ids.unsqueeze(0).to(next(self.model.parameters()).device)
            
            # Get model outputs
            with torch.no_grad():
                outputs = self.model(input_ids)
                logits = outputs.logits
            
            # Compute perplexity
            # Shift logits and labels for language modeling
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            
            # Compute cross-entropy loss
            loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # Convert loss to perplexity
            perplexity = torch.exp(loss).item()
            
            # Clamp perplexity to reasonable range
            perplexity = min(perplexity, 1000.0)  # Cap at 1000 to avoid overflow
            
            # Cache result
            self._perplexity_cache[cache_key] = perplexity
            
            return perplexity
            
        except Exception as e:
            logger.warning(f"Error computing perplexity: {e}")
            # Return high perplexity for failed computations
            perplexity = 100.0
            self._perplexity_cache[cache_key] = perplexity
            return perplexity
    
    def apply_readable_guidance(self,
                              grad: Tensor,
                              current_ids: Tensor,
                              position_idx: int,
                              strength: float = 0.3) -> Tensor:
        """
        Apply readable guidance to gradients by incorporating perplexity scores.
        
        Args:
            grad: Gradient tensor for vocabulary, shape (vocab_size,)
            current_ids: Current token sequence, shape (seq_len,)
            position_idx: Position being optimized
            strength: Strength of readable guidance (0.0 to 1.0)
            
        Returns:
            Modified gradient tensor, shape (vocab_size,)
        """
        if strength == 0.0:
            return grad
        
        vocab_size = grad.shape[0]
        device = grad.device
        
        # Get top-k candidates from gradient
        topk = min(100, vocab_size)  # Limit candidates for efficiency
        _, top_indices = torch.topk(-grad, topk, dim=0)  # Negative grad for topk candidates
        
        # Compute perplexity bonus for top candidates
        perplexity_bonus = self.compute_perplexity_bonus(
            current_ids, position_idx, top_indices, strength
        )
        
        # Create bonus tensor for full vocabulary
        vocab_bonus = torch.zeros_like(grad)
        vocab_bonus[top_indices] = perplexity_bonus
        
        # Apply bonus to gradients (subtract bonus from gradient to favor lower perplexity tokens)
        modified_grad = grad - vocab_bonus
        
        return modified_grad
    
    def evaluate_readability(self, text: str) -> Dict[str, float]:
        """
        Evaluate the readability of an adversarial string.
        
        Args:
            text: Text to evaluate
            
        Returns:
            Dictionary with readability metrics
        """
        # Tokenize the text
        if not text or not text.strip():
            # Handle empty text
            return {
                'perplexity': 1000.0,
                'readability_score': 0.0,
                'num_tokens': 0,
                'avg_token_length': 0.0,
                'has_spaces': False,
                'has_punctuation': False,
                'overall_score': 0.0
            }
        
        token_ids = self.tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")[0]
        
        # Compute overall perplexity
        overall_perplexity = self._compute_sequence_perplexity(token_ids)
        
        # Compute readability score (inverse of perplexity, normalized)
        # Lower perplexity = higher readability
        readability_score = 1.0 / (1.0 + overall_perplexity / 100.0)  # Normalize to [0, 1]
        
        # Compute additional metrics
        num_tokens = len(token_ids)
        avg_token_length = len(text) / max(num_tokens, 1)
        
        # Check for common readable patterns
        has_spaces = ' ' in text
        has_punctuation = any(c in text for c in '.,!?;:')
        
        # Combine metrics
        readability_metrics = {
            'perplexity': overall_perplexity,
            'readability_score': readability_score,
            'num_tokens': num_tokens,
            'avg_token_length': avg_token_length,
            'has_spaces': has_spaces,
            'has_punctuation': has_punctuation,
            'overall_score': readability_score * (1.0 + 0.1 * (has_spaces + has_punctuation))
        }
        
        return readability_metrics


def create_readable_optimizer(tokenizer, model) -> ReadableOptimizer:
    """
    Create a readable optimizer instance.
    
    Args:
        tokenizer: Model tokenizer
        model: Language model for perplexity computation
        
    Returns:
        ReadableOptimizer instance
    """
    return ReadableOptimizer(tokenizer, model)


# Backward compatibility alias
def create_injection_optimizer(tokenizer, model=None) -> ReadableOptimizer:
    """
    Create a readable optimizer instance (backward compatibility).
    
    Args:
        tokenizer: Model tokenizer
        model: Language model for perplexity computation
        
    Returns:
        ReadableOptimizer instance
    """
    if model is None:
        raise ValueError("Model is required for readable optimization")
    return ReadableOptimizer(tokenizer, model) 