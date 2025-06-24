"""
UDora: A Unified Red Teaming Framework against LLM Agents by Dynamically Hijacking Their Own Reasoning

This module implements the core UDora attack algorithm that dynamically optimizes adversarial strings
by leveraging LLM agents' own reasoning processes to trigger targeted malicious actions.

Key features:
- Dynamic position identification for noise insertion in reasoning traces
- Weighted interval scheduling for optimal target placement
- Support for both sequential and joint optimization modes
- Fallback mechanisms for robust attack execution
"""

import os
import copy
import gc
import logging
import re
from dataclasses import dataclass
from tqdm import tqdm
from typing import List, Optional, Union

import math
import torch
import transformers
from torch import Tensor
from transformers import set_seed

from .utils import INIT_CHARS, find_executable_batch_size, get_nonascii_toks, mellowmax, check_type, combine_with_overlap
from .scheduling import weighted_interval_scheduling, filter_intervals_by_sequential_mode, build_final_token_sequence
from .datasets import check_success_condition
from .text_processing import build_target_intervals, count_matched_locations, format_interval_debug_info
from .loss import compute_udora_loss, compute_cross_entropy_loss, apply_mellowmax_loss
from .readable import create_readable_optimizer

logger = logging.getLogger("UDora")
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


@dataclass
class UDoraConfig:
    """Configuration class for UDora attack parameters.
    
    Core Attack Parameters:
        num_steps: Maximum number of optimization iterations
        optim_str_init: Initial adversarial string(s) to optimize
        search_width: Number of candidate sequences to evaluate per iteration
        batch_size: Batch size for processing candidates (None for auto-sizing)
        topk: Top-k gradient directions to consider for token replacement
        n_replace: Number of token positions to update per sequence
        
    Advanced Optimization:
        buffer_size: Size of attack buffer for maintaining best candidates
        use_mellowmax: Whether to use mellowmax instead of cross-entropy loss
        mellowmax_alpha: Alpha parameter for mellowmax function
        early_stop: Stop optimization when target action is triggered
        
    Model Interface:
        use_prefix_cache: Cache prefix computations for efficiency
        allow_non_ascii: Allow non-ASCII tokens in optimization
        filter_ids: Filter token sequences that change after retokenization
        add_space_before_target: Add space before target text during matching
        max_new_tokens: Maximum tokens to generate during inference
        
    Experimental Features:
        minimize_reward: Minimize instead of maximize target probability (inverse attack)
        sequential: Use sequential vs joint optimization mode
        weight: Exponential weighting factor for multi-target optimization (weight^position)
        num_location: Number of target insertion locations
        prefix_update_frequency: How often to update reasoning context
        readable: Optimize for readable adversarial strings using perplexity guidance
        before_negative: Stop interval collection when encountering negative responses like "cannot"
        
    Utility:
        seed: Random seed for reproducibility
        verbosity: Logging level ("DEBUG", "INFO", "WARNING", "ERROR")
        dataset: Dataset name for success conditions. Predefined: "webshop", "injecagent", "agentharm". 
                Custom dataset names are also supported and use default text matching behavior.
    """
    # Core attack parameters
    num_steps: int = 250
    optim_str_init: Union[str, List[str]] = "x x x x x x x x x x x x x x x x x x x x x x x x x"
    search_width: int = 512
    batch_size: Optional[int] = None
    topk: int = 256
    n_replace: int = 1
    
    # Advanced optimization
    buffer_size: int = 0
    use_mellowmax: bool = False
    mellowmax_alpha: float = 1.0
    early_stop: bool = False
    
    # Model interface
    use_prefix_cache: bool = True
    allow_non_ascii: bool = False
    filter_ids: bool = True
    add_space_before_target: bool = False
    max_new_tokens: int = 300
    
    # Experimental features (beta)
    minimize_reward: bool = False  # beta, => minimizing the appearance of some specific noise
    sequential: bool = True
    weight: float = 1.0  # Exponential weighting factor for multi-target loss (weight^position)
    num_location: int = 2
    prefix_update_frequency: int = 1
    readable: bool = False  # beta, => optimize for readable adversarial strings using perplexity (requires readable optim_str_init)
    before_negative: bool = False  # beta, => stop interval collection when encountering negative responses
    
    # Utility
    seed: Optional[int] = None
    verbosity: str = "INFO"
    dataset: str = "webshop"  # Dataset name for different success conditions: "webshop", "injecagent", "agentharm"

@dataclass
class UDoraResult:
    """Results from a UDora attack execution.
    
    Attributes:
        best_loss: Lowest loss achieved during optimization
        best_string: Adversarial string that achieved the best loss
        best_generation: Model generation with the best adversarial string
        best_success: Whether the best attempt successfully triggered target action
        
        last_string: Final adversarial string from optimization
        last_generation: Model generation with the final adversarial string  
        last_success: Whether the final attempt successfully triggered target action
        
        vanilla_generation: Model generation without any adversarial string
        vanilla_success: Whether vanilla generation triggered target action
        
        all_generation: Complete history of generations during optimization
        losses: Loss values throughout optimization process
        strings: Adversarial strings throughout optimization process
    """
    # Best results
    best_loss: float
    best_string: str
    best_generation: Union[str, List[str]]
    best_success: Union[bool, List[bool]]
    
    # Final results
    last_string: str
    last_generation: Union[str, List[str]]
    last_success: Union[bool, List[bool]]
    
    # Baseline results
    vanilla_generation: Union[str, List[str]]
    vanilla_success: Union[bool, List[bool]]
    
    # Optimization history
    all_generation: List[list]
    losses: List[float]
    strings: List[str]
        
class AttackBuffer:
    """Buffer for maintaining the best adversarial candidates during optimization.
    
    Keeps track of the top candidates based on loss values, enabling efficient
    selection of promising adversarial strings for continued optimization.
    """
    
    def __init__(self, size: int):
        """Initialize attack buffer.
        
        Args:
            size: Maximum number of candidates to maintain (0 for single candidate)
        """
        self.buffer = []  # Elements are (loss: float, optim_ids: Tensor)
        self.size = size

    def add(self, loss: float, optim_ids: Tensor) -> None:
        """Add a new candidate to the buffer.
        
        Args:
            loss: Loss value for the candidate
            optim_ids: Token IDs of the adversarial string
        """
        if self.size == 0:
            self.buffer = [(loss, optim_ids)]
            return

        if len(self.buffer) < self.size:
            self.buffer.append((loss, optim_ids))
        else:
            # Only add if new loss is better than the worst in buffer
            worst_loss = max(item[0] for item in self.buffer)
            if loss < worst_loss:
                # Remove the worst item and add the new one
                worst_idx = max(range(len(self.buffer)), key=lambda i: self.buffer[i][0])
                self.buffer[worst_idx] = (loss, optim_ids)
            else:
                # New candidate is not better than any existing, don't add
                return

        # Keep buffer sorted by loss (best first)
        self.buffer.sort(key=lambda x: x[0])

    def get_best_ids(self) -> Tensor:
        """Get token IDs of the best candidate."""
        if not self.buffer:
            raise RuntimeError("Cannot get best IDs from empty attack buffer")
        return self.buffer[0][1]

    def get_lowest_loss(self) -> float:
        """Get the lowest (best) loss value."""
        if not self.buffer:
            raise RuntimeError("Cannot get lowest loss from empty attack buffer")
        return self.buffer[0][0]
    
    def get_highest_loss(self) -> float:
        """Get the highest (worst) loss value."""
        if not self.buffer:
            raise RuntimeError("Cannot get highest loss from empty attack buffer")
        return self.buffer[-1][0]
    
    def log_buffer(self, tokenizer):
        """Log current buffer contents for debugging."""
        message = "Attack buffer contents:"
        for loss, ids in self.buffer:
            optim_str = tokenizer.batch_decode(ids)[0]
            optim_str = optim_str.replace("\\", "\\\\")
            optim_str = optim_str.replace("\n", "\\n")
            message += f"\n  loss: {loss:.4f} | string: {optim_str}"
        logger.info(message)

def sample_ids_from_grad(
    ids: Tensor, 
    grad: Tensor, 
    search_width: int, 
    topk: int = 256,
    n_replace: int = 1,
    not_allowed_ids: Optional[Tensor] = None,
):
    """Returns `search_width` combinations of token ids based on the token gradient.

    Args:
        ids : Tensor, shape = (n_optim_ids)
            the sequence of token ids that are being optimized 
        grad : Tensor, shape = (n_optim_ids, vocab_size)
            the gradient of the GCG loss computed with respect to the one-hot token embeddings
        search_width : int
            the number of candidate sequences to return
        topk : int
            the topk to be used when sampling from the gradient
        n_replace : int
            the number of token positions to update per sequence
        not_allowed_ids : Tensor, shape = (n_ids)
            the token ids that should not be used in optimization
    
    Returns:
        sampled_ids : Tensor, shape = (search_width, n_optim_ids)
            sampled token ids
    """
    n_optim_tokens = len(ids)
    original_ids = ids.repeat(search_width, 1)

    if not_allowed_ids is not None:
        grad[:, not_allowed_ids.to(grad.device)] = float("inf")

    topk_ids = (-grad).topk(topk, dim=1).indices

    sampled_ids_pos = torch.argsort(torch.rand((search_width, n_optim_tokens), device=grad.device))[..., :n_replace]
    sampled_ids_val = torch.gather(
        topk_ids[sampled_ids_pos],
        2,
        torch.randint(0, topk, (search_width, n_replace, 1), device=grad.device)
    ).squeeze(2)

    new_ids = original_ids.scatter_(1, sampled_ids_pos, sampled_ids_val)

    return new_ids

def sample_ids_from_grad_readable(
    ids: Tensor, 
    grad: Tensor, 
    search_width: int, 
    readable_optimizer=None,
    readable_strength: float = 0.3,
    topk: int = 256,
    n_replace: int = 1,
    not_allowed_ids: Tensor = None,
):
    """Sample candidate token sequences with readable guidance.
    
    This function extends the standard gradient sampling with readable guidance
    to encourage natural language patterns in adversarial strings using perplexity.

    Args:
        ids: Current token sequence being optimized, shape (n_optim_ids,)
        grad: Gradient w.r.t. one-hot token embeddings, shape (n_optim_ids, vocab_size)
        search_width: Number of candidate sequences to generate
        readable_optimizer: ReadableOptimizer instance for guidance
        readable_strength: Strength of readable guidance (0.0 to 1.0)
        topk: Number of top gradient directions to consider per position
        n_replace: Number of token positions to modify per candidate
        not_allowed_ids: Token IDs to exclude from optimization (e.g., non-ASCII)
    
    Returns:
        Tensor of sampled token sequences, shape (search_width, n_optim_ids)
    """
    n_optim_tokens = len(ids)
    original_ids = ids.repeat(search_width, 1)

    if not_allowed_ids is not None:
        grad[:, not_allowed_ids.to(grad.device)] = float("inf")

    # Apply readable guidance to gradients if enabled
    if readable_optimizer is not None and readable_strength > 0.0:
        for pos in range(n_optim_tokens):
            grad[pos] = readable_optimizer.apply_readable_guidance(
                grad[pos], ids, pos, readable_strength
            )

    topk_ids = (-grad).topk(topk, dim=1).indices

    sampled_ids_pos = torch.argsort(torch.rand((search_width, n_optim_tokens), device=grad.device))[..., :n_replace]
    sampled_ids_val = torch.gather(
        topk_ids[sampled_ids_pos],
        2,
        torch.randint(0, topk, (search_width, n_replace, 1), device=grad.device)
    ).squeeze(2)

    new_ids = original_ids.scatter_(1, sampled_ids_pos, sampled_ids_val)

    return new_ids


def filter_ids(ids: Tensor, tokenizer: transformers.PreTrainedTokenizer):
    """Filter out token sequences that change after decode-encode round trip.
    
    This ensures that adversarial strings remain consistent when processed
    through the tokenizer's decode/encode cycle, which is important for
    maintaining attack effectiveness.

    Args:
        ids: Token ID sequences to filter, shape (search_width, n_optim_ids)
        tokenizer: Model's tokenizer for decode/encode operations
    
    Returns:
        Filtered token sequences that survive round-trip, shape (new_search_width, n_optim_ids)
        
    Raises:
        RuntimeError: If no sequences survive filtering (suggests bad initialization)
    """
    ids_decoded = tokenizer.batch_decode(ids)
    filtered_ids = []

    for i in range(len(ids_decoded)):
        # Retokenize the decoded token ids
        ids_encoded = tokenizer(ids_decoded[i], return_tensors="pt", add_special_tokens=False).to(ids.device)["input_ids"][0]
        if torch.equal(ids[i], ids_encoded):
           filtered_ids.append(ids[i]) 
    
    if not filtered_ids:
        # This occurs in some cases, e.g. using the Llama-3 tokenizer with a bad initialization
        raise RuntimeError(
            "No token sequences are the same after decoding and re-encoding. "
            "Consider setting `filter_ids=False` or trying a different `optim_str_init`"
        )
    
    return torch.stack(filtered_ids)

class UDora:
    """UDora: Unified Red Teaming Framework for LLM Agents.
    
    This class implements the core UDora attack algorithm that dynamically hijacks
    LLM agents' reasoning processes to trigger targeted malicious actions. The attack
    works by:
    
    1. Gathering the agent's initial reasoning response
    2. Identifying optimal positions for inserting target "noise" 
    3. Optimizing adversarial strings to maximize noise likelihood
    4. Iteratively refining the attack based on agent's evolving reasoning
    
    The approach adapts to different agent reasoning styles and can handle both
    malicious environment and malicious instruction scenarios.
    """
    
    def __init__(
        self, 
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizer,
        config: UDoraConfig,
    ):
        """Initialize UDora attack framework.
        
        Args:
            model: Target LLM to attack
            tokenizer: Tokenizer for the target model
            config: Attack configuration parameters
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

        # Core components
        self.embedding_layer = model.get_input_embeddings()
        self.not_allowed_ids = None if config.allow_non_ascii else get_nonascii_toks(tokenizer, device=model.device)
        self.batch_prefix_cache = None

        # Initialize readable optimizer if enabled
        self.readable_optimizer = None
        if config.readable:
            self.readable_optimizer = create_readable_optimizer(tokenizer, model)
            logger.info("Readable optimization enabled - adversarial strings will be optimized for natural language patterns")

        # Attack state
        self.stop_flag = False

        # Performance warnings
        if model.dtype in (torch.float32, torch.float64):
            logger.warning(f"Model is in {model.dtype}. Use a lower precision data type, if possible, for much faster optimization.")

        if model.device == torch.device("cpu"):
            logger.warning("Model is on the CPU. Use a hardware accelerator for faster optimization.")

        # Handle missing chat template
        if not tokenizer.chat_template:
            logger.warning("Tokenizer does not have a chat template. Assuming base model and setting chat template to empty.")
            tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
    
    def run(
        self,
        messages: Union[List[Union[str, List[dict]]], Union[str, List[dict]]],
        targets: Union[List[Union[str, List[str]]], str],
    ) -> UDoraResult:
        """Execute UDora attack to generate adversarial strings.
        
        This is the main entry point for running UDora attacks. It handles both
        single and batch inputs, supporting various message formats and target
        specifications.
        
        Args:
            messages: Input messages/conversations to attack. Can be:
                - Single string
                - List of strings  
                - Single conversation (list of dicts with 'role'/'content')
                - List of conversations
            targets: Target actions/tools to trigger. Can be:
                - Single target string
                - List of target strings
                - List of lists for multiple targets per message
                
        Returns:
            UDoraResult containing attack results, including best/final adversarial
            strings, generated responses, success indicators, and optimization history.
        """
        model = self.model
        tokenizer = self.tokenizer
        config = self.config

        if config.seed is not None:
            set_seed(config.seed)
            torch.use_deterministic_algorithms(True, warn_only=True)
        
        # Process input messages
        messages = self._process_input_messages(messages)
        self.targets = self._process_targets(targets, len(messages))
        
        # Initialize batch state
        self._initialize_batch_state(len(messages))
        
        # Prepare templates
        self.messages = messages
        templates = self._prepare_templates(messages)
        self.templates = templates
        
        # Use provided readable initialization
        self._update_target_and_context(config.optim_str_init)

        # Initialize the attack buffer
        buffer = self.init_buffer()
        optim_ids = buffer.get_best_ids()

        losses = []
        optim_strings = []
        
        # Main optimization loop
        for step in tqdm(range(config.num_steps)):
            if self.stop_flag:
                losses.append(buffer.get_lowest_loss())
                optim_ids = buffer.get_best_ids()
                optim_str = tokenizer.batch_decode(optim_ids)[0]
                optim_strings.append(optim_str)
                break
                
            # Compute gradients and sample candidates
            optim_ids_onehot_grad = self.compute_token_gradient(optim_ids)
            
            with torch.no_grad():
                # Sample candidates with readable guidance
                if config.readable and self.readable_optimizer:
                    sampled_ids = sample_ids_from_grad_readable(
                        optim_ids.squeeze(0),
                        optim_ids_onehot_grad.squeeze(0),
                        config.search_width,
                        self.readable_optimizer,
                        readable_strength=0.3,  # Moderate strength for readability
                        topk=config.topk,
                        n_replace=config.n_replace,
                        not_allowed_ids=self.not_allowed_ids,
                    )
                else:
                    sampled_ids = sample_ids_from_grad(
                        optim_ids.squeeze(0),
                        optim_ids_onehot_grad.squeeze(0),
                        config.search_width,
                        topk=config.topk,
                        n_replace=config.n_replace,
                        not_allowed_ids=self.not_allowed_ids,
                    )

                if config.filter_ids:
                    sampled_ids = filter_ids(sampled_ids, tokenizer)

                # Compute loss on candidates
                loss = self._compute_batch_loss(sampled_ids)
                
                # Ensure loss is a tensor for min/argmin operations
                if not isinstance(loss, torch.Tensor):
                    loss = torch.tensor(loss, device=self.model.device)
                
                current_loss = loss.min().item()
                optim_ids = sampled_ids[loss.argmin()].unsqueeze(0)

                # Update buffer
                losses.append(current_loss)
                if buffer.size == 0 or len(buffer.buffer) < buffer.size or current_loss < buffer.get_highest_loss():
                    buffer.add(current_loss, optim_ids)
                    
            optim_ids = buffer.get_best_ids()
            optim_str = tokenizer.batch_decode(optim_ids)[0]
            optim_strings.append(optim_str)
            
            # Log readability metrics if optimization is enabled
            if self.config.readable and self.readable_optimizer is not None:
                readability_metrics = self.readable_optimizer.evaluate_readability(optim_str)
                logger.info(f"Readability Score: {readability_metrics['readability_score']:.3f} "
                          f"(perplexity: {readability_metrics['perplexity']:.2f}, "
                          f"overall: {readability_metrics['overall_score']:.3f})")
            
            buffer.log_buffer(tokenizer)                
            
            # Update context periodically
            if (step + 1) % self.config.prefix_update_frequency == 0:
                self._update_target_and_context(optim_str)
            else:
                self._generate_and_store_responses(optim_str)
                    
            if self.stop_flag:
                logger.info("Early stopping due to finding a perfect match.") 
                break
              
        # Generate final results
        return self._compile_results(losses, optim_strings)

    def _process_input_messages(self, messages):
        """Process and normalize input messages."""
        if isinstance(messages, str):
            return [[{"role": "user", "content": messages}]]
        elif check_type(messages, str):
            return [[{"role": "user", "content": message}] for message in messages]
        elif check_type(messages, dict):
            return [messages]
        return copy.deepcopy(messages)
    
    def _process_targets(self, targets, num_messages):
        """Process and normalize target specifications."""
        if isinstance(targets, str):
            return [[targets]] * num_messages
        elif check_type(targets, str):
            return [[target] for target in targets]
        else:
            return [[target for target in target_list] for target_list in targets]
    
    def _initialize_batch_state(self, batch_size):
        """Initialize batch processing state variables."""
        self.batch_before_ids = [[] for _ in range(batch_size)]
        self.batch_after_ids = [[] for _ in range(batch_size)]
        self.batch_target_ids = [[] for _ in range(batch_size)]
        self.batch_before_embeds = [[] for _ in range(batch_size)]
        self.batch_after_embeds = [[] for _ in range(batch_size)]
        self.batch_target_embeds = [[] for _ in range(batch_size)]
        self.batch_prefix_cache = [[] for _ in range(batch_size)]
        self.batch_all_generation = [[] for _ in range(batch_size)]
    
    def _prepare_templates(self, messages):
        """Prepare conversation templates from messages."""
        templates = [
            self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True) 
            if isinstance(message, list) else message
            for message in messages
        ]
        
        def bos_filter(template):
            if self.tokenizer.bos_token and template.startswith(self.tokenizer.bos_token):
                return template.replace(self.tokenizer.bos_token, "")
            return template
        
        templates = list(map(bos_filter, templates))
        
        # Add optim_str placeholder if not present
        for i, conversation in enumerate(messages):
            if isinstance(conversation, list):
                if not any("{optim_str}" in message.get("content", "") for message in conversation):
                    conversation[-1]["content"] += "{optim_str}"
                    # Regenerate template
                    templates[i] = self.tokenizer.apply_chat_template(
                        conversation, tokenize=False, add_generation_prompt=True
                    )
                    templates[i] = bos_filter(templates[i])
        
        return templates
    
    def _compute_batch_loss(self, sampled_ids):
        """Compute loss for a batch of candidate sequences."""
        new_search_width = sampled_ids.shape[0]
        batch_size = new_search_width if self.config.batch_size is None else self.config.batch_size
        
        total_loss = 0
        for idx in range(len(self.batch_before_embeds)):
            if self.batch_prefix_cache and self.batch_prefix_cache[idx]:
                input_embeds = torch.cat([
                    self.embedding_layer(sampled_ids),
                    self.batch_after_embeds[idx].repeat(new_search_width, 1, 1),
                    *[target_embedding.repeat(new_search_width, 1, 1) 
                      for target_embedding in self.batch_target_embeds[idx]]
                ], dim=1)
            else:
                input_embeds = torch.cat([
                    self.batch_before_embeds[idx].repeat(new_search_width, 1, 1),
                    self.embedding_layer(sampled_ids),
                    self.batch_after_embeds[idx].repeat(new_search_width, 1, 1),
                    *[target_embedding.repeat(new_search_width, 1, 1) 
                      for target_embedding in self.batch_target_embeds[idx]]
                ], dim=1)
            
            loss_result = find_executable_batch_size(self.compute_candidates_loss, batch_size)(input_embeds, idx)
            total_loss += loss_result

        return total_loss / len(self.batch_before_embeds)
    
    def _compute_generation_success(self, generation_str: str, batch_idx: int) -> bool:
        """Compute success status for a single generation."""
        if self.config.minimize_reward:
            return all(target not in generation_str for target in self.targets[batch_idx])
        else:
            return check_success_condition(generation_str, self.targets[batch_idx], self.config.dataset)

    def _generate_and_store_responses(self, optim_str):
        """Generate responses for current adversarial string and store them."""
        for k, template in enumerate(self.templates):
            outputs, decoding_string = self.greedy_generation(template, optim_str, k)
            print(decoding_string)
            # Log the decoded string when verbosity is INFO or DEBUG
            logger.info(f"Model's response: {decoding_string}")
            
            # Compute success status for this generation
            success = self._compute_generation_success(decoding_string, k)
            
            self.batch_all_generation[k].append((optim_str, decoding_string, success))
    
    def _compile_results(self, losses, optim_strings):
        """Compile final attack results."""
        min_loss_index = losses.index(min(losses))
        
        # Generate responses for evaluation
        vanilla_generation = [self.greedy_generation(template, '')[1] for template in self.templates]
        last_generation = [self.greedy_generation(template, optim_strings[-1])[1] 
                          for template in self.templates]
        
        # Add last_generation to all_generation for each batch
        last_success = []
        for k, gen in enumerate(last_generation):
            success = self._compute_generation_success(gen, k)
            last_success.append(success)
            # Add the last generation to batch_all_generation
            self.batch_all_generation[k].append((optim_strings[-1], gen, success))
        
        # Compute success metrics
        def compute_success(generation_list):
            if self.config.minimize_reward:
                return [all(target not in gen for target in self.targets[i]) 
                       for i, gen in enumerate(generation_list)]
            else:
                return [check_success_condition(gen, self.targets[i], self.config.dataset) 
                       for i, gen in enumerate(generation_list)]
        
        # Find best result among successful generations, or fallback to min loss
        best_index = self._find_best_successful_index(losses, optim_strings)
        best_generation = [self.greedy_generation(template, optim_strings[best_index])[1] 
                          for template in self.templates]
        
        return UDoraResult(
            best_loss=losses[best_index],
            best_string=optim_strings[best_index],
            best_generation=best_generation,
            best_success=compute_success(best_generation),
            
            last_string=optim_strings[-1],
            last_generation=last_generation,
            last_success=compute_success(last_generation),
            
            vanilla_generation=vanilla_generation,
            vanilla_success=compute_success(vanilla_generation),
            
            all_generation=self.batch_all_generation,
            losses=losses,
            strings=optim_strings,
        )
    
    def _find_best_successful_index(self, losses, optim_strings):
        """Find the index of the best successful attack, or fallback to minimum loss."""
        successful_indices = []
        
        # Determine the maximum number of generations across all batches
        max_generations = max(len(self.batch_all_generation[k]) for k in range(len(self.batch_all_generation))) if self.batch_all_generation else 0
        
        # Check each generation step for success
        for step_idx in range(max_generations):
            step_successful = True
            
            for batch_idx in range(len(self.batch_all_generation)):
                # Find the generation corresponding to this step
                if step_idx < len(self.batch_all_generation[batch_idx]):
                    generation_tuple = self.batch_all_generation[batch_idx][step_idx]
                    
                    # Extract success status from the tuple
                    # For tuples with success: (optim_str, decoding_string, success) or (optim_str, decoding_string, later_response, success)
                    if len(generation_tuple) >= 3:
                        # Check if the last element is a boolean (success status)
                        if isinstance(generation_tuple[-1], bool):
                            batch_success = generation_tuple[-1]
                        else:
                            # Fallback: compute success from generation string
                            generation_str = generation_tuple[1] if len(generation_tuple) > 1 else ""
                            batch_success = self._compute_generation_success(generation_str, batch_idx)
                    else:
                        # Fallback for old tuple format without success status
                        generation_str = generation_tuple[1] if len(generation_tuple) > 1 else ""
                        batch_success = self._compute_generation_success(generation_str, batch_idx)
                    
                    if not batch_success:
                        step_successful = False
                        break
                else:
                    # If we don't have generation for this step, consider it unsuccessful
                    step_successful = False
                    break
            
            if step_successful:
                successful_indices.append(step_idx)
        
        # Among successful attacks, find the one with minimum loss
        if successful_indices:
            # For successful indices, use the corresponding loss if available, otherwise use the minimum loss
            if successful_indices[0] < len(losses):
                best_successful_idx = min(successful_indices, key=lambda idx: losses[idx] if idx < len(losses) else float('inf'))
                logger.info(f"Found {len(successful_indices)} successful attacks. Best successful at step {best_successful_idx} with loss {losses[best_successful_idx] if best_successful_idx < len(losses) else 'N/A'}.")
                return best_successful_idx
            else:
                # Early stopping case: successful attack found before any optimization steps
                logger.info(f"Found {len(successful_indices)} successful attacks during initialization (before optimization loop).")
                return 0  # Return index 0 to use the first available loss/string
        else:
            # No successful attacks found, fallback to minimum loss
            if losses:
                min_loss_idx = losses.index(min(losses))
                logger.info(f"No successful attacks found. Using minimum loss at step {min_loss_idx} with loss {losses[min_loss_idx]:.4f}")
                return min_loss_idx
            else:
                # No losses available (early stopping before any optimization)
                logger.info("No successful attacks found and no optimization losses available. Using index 0.")
                return 0

    def init_buffer(self) -> AttackBuffer:
        """Initialize the attack buffer with initial adversarial candidates."""
        model = self.model
        tokenizer = self.tokenizer
        config = self.config

        logger.info(f"Initializing attack buffer of size {config.buffer_size}...")

        buffer = AttackBuffer(config.buffer_size)

        if isinstance(config.optim_str_init, str):
            init_optim_ids = tokenizer(config.optim_str_init, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)
        else:
            if len(config.optim_str_init) != config.buffer_size:
                logger.warning(f"Using {len(config.optim_str_init)} initializations but buffer size is set to {config.buffer_size}")
            try:
                init_optim_ids = tokenizer(config.optim_str_init, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)
            except ValueError:
                logger.error("Unable to create buffer. Ensure that all initializations tokenize to the same length.")

        true_buffer_size = 1

        # Compute initial losses
        init_buffer_losses = self._compute_batch_loss(init_optim_ids)
        
        # Ensure init_buffer_losses is a tensor
        if isinstance(init_buffer_losses, torch.Tensor):
            if init_buffer_losses.dim() == 0:
                # Scalar tensor, expand to match buffer size
                init_buffer_losses = init_buffer_losses.unsqueeze(0).expand(true_buffer_size)
        else:
            # Convert to tensor if it's a scalar
            init_buffer_losses = torch.tensor([init_buffer_losses], device=model.device)

        # Populate buffer
        for i in range(true_buffer_size):
            if isinstance(init_buffer_losses, torch.Tensor):
                loss_val = init_buffer_losses[i].item() if init_buffer_losses.numel() > i else init_buffer_losses.item()
            else:
                loss_val = float(init_buffer_losses)
            buffer.add(loss_val, init_optim_ids[[i]])
        
        buffer.log_buffer(tokenizer)
        logger.info("Initialized attack buffer.")
        
        return buffer
    
    def compute_token_gradient(self, optim_ids: Tensor) -> Tensor:
        """Compute gradients of UDora loss w.r.t. one-hot token embeddings."""
        model = self.model
        embedding_layer = self.embedding_layer

        # Create one-hot encoding
        optim_ids_onehot = torch.nn.functional.one_hot(optim_ids, num_classes=embedding_layer.num_embeddings)
        optim_ids_onehot = optim_ids_onehot.to(dtype=model.dtype, device=model.device)
        optim_ids_onehot.requires_grad_()

        optim_embeds = optim_ids_onehot @ embedding_layer.weight

        # Compute loss across all batches
        total_loss = 0
        for idx in range(len(self.batch_before_embeds)):
            if self.batch_prefix_cache and self.batch_prefix_cache[idx]:
                input_embeds = torch.cat([optim_embeds, self.batch_after_embeds[idx], *self.batch_target_embeds[idx]], dim=1)
                output = model(inputs_embeds=input_embeds, past_key_values=self.batch_prefix_cache[idx])
            else:
                input_embeds = torch.cat([self.batch_before_embeds[idx], optim_embeds, self.batch_after_embeds[idx], *self.batch_target_embeds[idx]], dim=1)
                output = model(inputs_embeds=input_embeds)
            
            logits = output.logits
            batch_loss = self._compute_target_loss(logits, idx)
            total_loss += batch_loss
                
        mean_loss = total_loss / len(self.batch_before_embeds)
        optim_ids_onehot_grad = torch.autograd.grad(outputs=[mean_loss], inputs=[optim_ids_onehot])[0]
        
        if self.config.minimize_reward:
            optim_ids_onehot_grad = -optim_ids_onehot_grad

        return optim_ids_onehot_grad
    
    def _compute_target_loss(self, logits, idx):
        """Compute loss for target sequences in a batch."""
        shift = logits.shape[1] - sum([target_embedding.shape[1] for target_embedding in self.batch_target_embeds[idx]])
        current_shift = shift - 1
        total_loss = 0
        
        for p in range(len(self.batch_target_embeds[idx])):
            length = self.batch_target_embeds[idx][p].shape[1]
            
            # Only compute loss on even indices (target sequences)
            if p % 2 == 0:
                positions = list(range(current_shift, current_shift + length))
                
                if self.config.use_mellowmax:
                    loss = apply_mellowmax_loss(
                        logits, self.batch_target_ids[idx][p], positions,
                        self.config.mellowmax_alpha, self.config.weight, p//2
                    )
                else:
                    loss = compute_cross_entropy_loss(
                        logits, self.batch_target_ids[idx][p], positions,
                        self.config.weight, p//2
                    )
                
                total_loss += loss
                        
            current_shift += length
            
        return total_loss

    def compute_candidates_loss(self, search_batch_size: int, input_embeds: Tensor, idx: int) -> Tensor:
        """Compute UDora loss for candidate adversarial sequences."""
        all_loss = []
        prefix_cache_batch = []
        
        prefix_cache = self.batch_prefix_cache[idx] if self.batch_prefix_cache else None
            
        for i in range(0, input_embeds.shape[0], search_batch_size):
            with torch.no_grad():
                input_embeds_batch = input_embeds[i:i+search_batch_size]
                current_batch_size = input_embeds_batch.shape[0]
                
                if prefix_cache:
                    if not prefix_cache_batch or current_batch_size != search_batch_size:
                        prefix_cache_batch = [[x.expand(current_batch_size, -1, -1, -1) for x in prefix_cache[i]] 
                                            for i in range(len(prefix_cache))]
                    outputs = self.model(inputs_embeds=input_embeds_batch, past_key_values=prefix_cache_batch)
                else:
                    outputs = self.model(inputs_embeds=input_embeds_batch)

                logits = outputs.logits
                
                # Compute UDora loss
                loss = self._compute_udora_batch_loss(logits, idx, current_batch_size)
                
                del outputs
                gc.collect()
                torch.cuda.empty_cache()
                all_loss.append(loss)
                
        result = torch.cat(all_loss, dim=0)
        return -result if self.config.minimize_reward else result
    
    def _compute_udora_batch_loss(self, logits, idx, batch_size):
        """Compute UDora loss for a batch of candidates."""
        tmp = logits.shape[1] - sum([target_embedding.shape[1] for target_embedding in self.batch_target_embeds[idx]])
        
        positions = []
        all_shift_labels = []
        current_shift = tmp - 1
        
        for p in range(len(self.batch_target_embeds[idx])):
            length = self.batch_target_embeds[idx][p].shape[1]
            if p % 2 == 0:
                positions.append(list(range(current_shift, current_shift + length)))
                all_shift_labels.append(self.batch_target_ids[idx][p])
            current_shift += length
        
        current_loss = 0
        active_mask = torch.ones(batch_size, dtype=torch.bool, device=self.model.device)
        
        for p in range(len(positions)):
            loss = compute_udora_loss(
                logits, all_shift_labels[p], positions[p], 
                self.config.weight, p
            )
            
            loss = loss * active_mask.float()
            current_loss += loss

            if not self.config.sequential:
                seq_len = len(positions[p])
                active_mask = active_mask & (torch.ones(batch_size, dtype=torch.bool, device=self.model.device))
                if not active_mask.any():
                    break
                    
        return current_loss

    @torch.no_grad()
    def _update_target_and_context(self, optim_str):
        """Update target positions and reasoning context using interval scheduling."""
        model = self.model
        tokenizer = self.tokenizer
        embedding_layer = self.embedding_layer
        
        for k, template in enumerate(self.templates):
            # Generate response and get intervals
            outputs, decoding_string = self.greedy_generation(template, optim_str, k)
            # Log the decoded string when verbosity is INFO or DEBUG
            generated_ids = outputs.generated_ids
            logits = outputs.scores
            probs_list = [logits_step.softmax(dim=-1)[0] for logits_step in logits]
            
            # Build target intervals
            intervals = build_target_intervals(
                generated_ids, self.targets[k], tokenizer, probs_list,
                self.config.add_space_before_target, self.config.before_negative
            )
            
            # Apply weighted interval scheduling
            selected_indices = weighted_interval_scheduling(intervals, self.config.num_location)
            
            # Filter based on sequential mode
            selected_indices = filter_intervals_by_sequential_mode(
                intervals, selected_indices, self.config.sequential
            )
            
            # Build final token sequence
            final_generated_ids = build_final_token_sequence(
                intervals, selected_indices, generated_ids, model.device
            )
            
            # Log results
            matched_count = count_matched_locations(intervals)
            logger.info(f"Batch {k}: Found {len(intervals)} intervals, {matched_count} matched, selected {len(selected_indices)}")
                
            all_response_ids = torch.cat(final_generated_ids, dim=-1) if final_generated_ids else torch.tensor([], device=model.device)
            surrogate_response = tokenizer.decode(all_response_ids[0] if len(all_response_ids.shape) > 1 else all_response_ids, skip_special_tokens=False)
#             print(later_response)
            logger.info(f"Target context update:\n**Generated response**: {decoding_string}\n**Surrogate response**: {surrogate_response}")
                
            # Update embeddings and context
            self._update_batch_embeddings(k, template, final_generated_ids)
            
            # Compute success status for this generation
            success = self._compute_generation_success(decoding_string, k)
            
            self.batch_all_generation[k].append((optim_str, decoding_string, surrogate_response, success))
            
            del outputs
            gc.collect()
            torch.cuda.empty_cache()
    
    def _update_batch_embeddings(self, k, template, final_generated_ids):
        """Update batch embeddings and cache for a specific batch index."""
        before_str, after_str = template.split("{optim_str}")
        
        before_ids = self.tokenizer([before_str.rstrip()], add_special_tokens=False, padding=False, return_tensors="pt")["input_ids"].to(self.model.device)
        after_ids = self.tokenizer([after_str.lstrip()], add_special_tokens=False, padding=False, return_tensors="pt")["input_ids"].to(self.model.device)
        
        if final_generated_ids:
            assistant_response_ids = final_generated_ids.pop(0)
            after_ids = torch.cat((after_ids, assistant_response_ids), dim=-1)
        
        # Update embeddings
        before_embeds, after_embeds = [self.embedding_layer(ids) for ids in (before_ids, after_ids)]
        target_embeds = [self.embedding_layer(ids) for ids in final_generated_ids]
        
        # Update prefix cache if needed
        if self.config.use_prefix_cache and isinstance(self.batch_before_embeds[k], list):
            output = self.model(inputs_embeds=before_embeds, use_cache=True)
            self.batch_prefix_cache[k] = output.past_key_values
            self.batch_before_ids[k] = before_ids
            self.batch_before_embeds[k] = before_embeds
            
        self.batch_after_ids[k] = after_ids
        self.batch_target_ids[k] = final_generated_ids
        self.batch_after_embeds[k] = after_embeds
        self.batch_target_embeds[k] = target_embeds

    @torch.no_grad()
    def greedy_generation(self, template, optim_str, idx=None):
        """Generate agent response using greedy decoding."""
        # Prepare input
        if optim_str == '':
            input_ids = self.tokenizer([template.replace(' {optim_str}', '')], 
                                     add_special_tokens=False, padding=False, return_tensors="pt")["input_ids"].to(self.model.device)
        else:
            input_ids = self.tokenizer([template.replace('{optim_str}', optim_str)], 
                                     add_special_tokens=False, padding=False, return_tensors="pt")["input_ids"].to(self.model.device)
        
        attn_masks = torch.ones_like(input_ids).to(self.model.device)
        
        # Generate response
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attn_masks, 
            top_k=0, 
            top_p=1.0,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
            output_hidden_states=False,
            output_attentions=False,
            pad_token_id=self.tokenizer.eos_token_id,
            past_key_values=self.batch_prefix_cache[idx] if idx is not None and self.batch_prefix_cache and self.batch_prefix_cache[idx] else None,
            max_new_tokens=self.config.max_new_tokens,
        )
        
        outputs.generated_ids = outputs.sequences[0, input_ids.shape[1]:].tolist()
        if self.tokenizer.eos_token_id and outputs.generated_ids and outputs.generated_ids[-1] == self.tokenizer.eos_token_id:
            outputs.generated_ids = outputs.generated_ids[:-1]
            
        decoded_string = self.tokenizer.decode(outputs.generated_ids, skip_special_tokens=False)
        
        # Check for early stopping
        if idx is not None and self.config.early_stop:
            if self.config.minimize_reward:
                if all(tgt_text not in decoded_string for tgt_text in self.targets[idx]):
                    self.stop_flag = True
            elif check_success_condition(decoded_string, self.targets[idx], self.config.dataset):
                self.stop_flag = True
                
        return outputs, decoded_string

def run(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    messages: Union[str, List[dict]],
    target: str,
    config: Optional[UDoraConfig] = None,
) -> UDoraResult:
    """Execute UDora attack with simplified API."""
    if config is None:
        config = UDoraConfig()
    
    logger.setLevel(getattr(logging, config.verbosity))
    
    udora = UDora(model, tokenizer, config)
    result = udora.run(messages, target)
    return result
    