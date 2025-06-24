"""
Loss Computation Functions for UDora

This module contains specialized loss functions used by UDora for optimizing
adversarial strings based on target action probabilities and rewards.
"""

from typing import List, Tuple
import torch
from torch import Tensor


def compute_udora_loss(logits: Tensor, 
                      target_ids: Tensor,
                      positions: List[int],
                      weight: float = 1.0,
                      position_index: int = 0) -> Tensor:
    """
    Compute UDora loss combining probability and reward components.
    
    This loss function encourages the model to generate target sequences
    by combining negative log-likelihood with consecutive matching rewards.
    
    Args:
        logits: Model logits, shape (batch_size, seq_len, vocab_size)
        target_ids: Target token IDs, shape (batch_size, target_len)
        positions: List of positions in the sequence for target tokens
        weight: Exponential weighting factor
        position_index: Index for position-based weighting
        
    Returns:
        Loss tensor, shape (batch_size,)
    """
    batch_size = logits.shape[0]
    device = logits.device
    
    # Extract logits at target positions
    position_tensor = torch.tensor(positions, dtype=torch.long, device=device)
    shift_logits = logits[..., position_tensor, :].contiguous()
    shift_labels = target_ids.repeat(batch_size, 1)
    
    # Compute probabilities and predictions
    probabilities = torch.softmax(shift_logits, dim=-1)
    correct_probs = probabilities.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
    negative_probs = -correct_probs
    
    predictions = torch.argmax(shift_logits, dim=-1)
    matches = predictions == shift_labels
    
    # Compute consecutive matching reward
    matched_reward = -1 * (torch.cumprod(matches.float(), dim=1).sum(dim=1))
    
    # Compute loss up to first mismatch
    seq_len = matches.size(1)
    indices = torch.arange(seq_len, device=device).unsqueeze(0).expand_as(matches)
    first_mismatch_indices = torch.where(
        ~matches, indices, torch.full_like(indices, seq_len)
    ).min(dim=1)[0]
    
    # Create mask up to and including first mismatch
    mask = indices <= first_mismatch_indices.unsqueeze(1)
    
    # Compute mean negative probability up to first mismatch
    neg_probs_masked = negative_probs * mask.float()
    sum_neg_probs = neg_probs_masked.sum(dim=1)
    mask_sum = mask.float().sum(dim=1)
    mean_neg_probs = sum_neg_probs / mask_sum
    
    # Final loss with normalization and weighting
    loss = (matched_reward + mean_neg_probs) / (seq_len + 1) * (weight ** position_index)
    
    return loss


def compute_cross_entropy_loss(logits: Tensor, 
                              target_ids: Tensor,
                              positions: List[int],
                              weight: float = 1.0,
                              position_index: int = 0) -> Tensor:
    """
    Compute standard cross-entropy loss for target sequences.
    
    Args:
        logits: Model logits, shape (batch_size, seq_len, vocab_size)
        target_ids: Target token IDs, shape (batch_size, target_len)
        positions: List of positions in the sequence for target tokens
        weight: Exponential weighting factor
        position_index: Index for position-based weighting
        
    Returns:
        Loss tensor, shape (batch_size,)
    """
    device = logits.device
    position_tensor = torch.tensor(positions, dtype=torch.long, device=device)
    shift_logits = logits[..., position_tensor, :].contiguous()
    shift_labels = target_ids.repeat(logits.shape[0], 1)
    
    # Compute cross-entropy loss
    loss = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)), 
        shift_labels.view(-1),
        reduction='none'
    )
    
    # Reshape and apply weighting
    loss = loss.view(logits.shape[0], -1).mean(dim=-1)
    loss = loss * (weight ** position_index)
    
    return loss


def apply_mellowmax_loss(logits: Tensor,
                        target_ids: Tensor, 
                        positions: List[int],
                        alpha: float = 1.0,
                        weight: float = 1.0,
                        position_index: int = 0) -> Tensor:
    """
    Apply mellowmax loss function for smoother optimization.
    
    Args:
        logits: Model logits, shape (batch_size, seq_len, vocab_size)
        target_ids: Target token IDs, shape (batch_size, target_len)
        positions: List of positions in the sequence for target tokens
        alpha: Mellowmax temperature parameter
        weight: Exponential weighting factor
        position_index: Index for position-based weighting
        
    Returns:
        Loss tensor, shape (batch_size,)
    """
    from .utils import mellowmax
    
    device = logits.device
    position_tensor = torch.tensor(positions, dtype=torch.long, device=device)
    shift_logits = logits[..., position_tensor, :].contiguous()
    shift_labels = target_ids.repeat(logits.shape[0], 1)
    
    # Extract target probabilities
    label_logits = torch.gather(shift_logits, -1, shift_labels.unsqueeze(-1)).squeeze(-1)
    
    # Apply mellowmax
    loss = mellowmax(-label_logits, alpha=alpha, dim=-1)
    loss = loss * (weight ** position_index)
    
    return loss 