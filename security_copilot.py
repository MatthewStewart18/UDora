"""
Security Copilot Agent Attack Script

This script implements adversarial attacks against Security Copilot agents using the UDora framework.
It automatically determines data paths based on model type and supports different attack scenarios targeting
security analysis, incident response, and threat detection capabilities.

Supported Models:
- meta-llama/Llama-3.1-8B-Instruct (mapped to 'llama')
- mistralai/Ministral-8B-Instruct-2410 (mapped to 'ministral')

Attack Types:
- privilege_escalation: Attempts to escalate privileges through security analysis
- data_exfiltration: Tricks agent into exposing sensitive security data
- malicious_execution: Causes execution of malicious commands during analysis
- policy_bypass: Bypasses security policies and restrictions
- false_analysis: Generates false positives/negatives in threat detection

Usage:
    # Privilege escalation attack
    python security_copilot.py --model_id meta-llama/Llama-3.1-8B-Instruct --task privilege_escalation --mode train --add_space_before_target --num_location 3 --early_stop

    # Data exfiltration attack
    python security_copilot.py --model_id meta-llama/Llama-3.1-8B-Instruct --task data_exfiltration --mode train --add_space_before_target --num_location 3 --early_stop
"""

import argparse
import os
import re
import json
import pickle
from typing import List, Tuple, Dict, Any
from tqdm import tqdm

import setGPU
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from udora import UDoraConfig, run as udora_run
from attack_utils import *


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for Security Copilot attack configuration."""
    parser = argparse.ArgumentParser(
        description="Security Copilot Agent Attack using UDora Framework",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--optim_str_init', type=str, default="x x x x x x x x x x x x x x x x x x x x x x x x x", help='Initial adversarial string for optimization')
    
    # Model Configuration
    parser.add_argument('--model_id', type=str, default="meta-llama/Llama-3.1-8B-Instruct", help='Model identifier from Hugging Face Hub (llama or ministral supported)')
    
    # Execution Mode
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'val'], help='Execution mode: train for training set, val for validation set (used to obtain optimal initialization strings)')
    
    # Task Configuration
    parser.add_argument('--task', type=str, default='privilege_escalation', 
                       choices=['privilege_escalation', 'data_exfiltration', 'malicious_execution', 'policy_bypass', 'false_analysis'], 
                       help='Security Copilot attack type')
    
    # UDora Attack Parameters
    parser.add_argument('--num_steps', type=int, default=250, help='Number of optimization steps')
    parser.add_argument('--search_width', type=int, default=200, help='Number of candidate sequences per optimization step')
    parser.add_argument('--weight', type=float, default=1.0, help='Exponential weighting factor for different token positions')
    parser.add_argument('--topk', type=int, default=48, help='Top-K gradient directions to consider')
    parser.add_argument('--n_replace', type=int, default=1, help='Number of token positions to modify per candidate')
    parser.add_argument('--num_location', type=int, default=2, help='Number of target insertion locations')
    parser.add_argument('--prefix_update_frequency', type=int, default=1, help='How often to update reasoning context')
    parser.add_argument('--max_new_tokens', type=int, default=400, help='Maximum number of new tokens to generate')
    
    # Advanced Options
    parser.add_argument('--buffer_size', type=int, default=0, help='Attack buffer size (0 for single candidate)')
    parser.add_argument('--sequential', action='store_true', help='Use sequential vs joint optimization mode')
    parser.add_argument('--early_stop', action='store_true', help='Stop optimization when target action is triggered')
    parser.add_argument('--add_space_before_target', action='store_true', help='Add space before target text during matching')
    parser.add_argument('--use_mellowmax', action='store_true', help='Use mellowmax instead of cross-entropy loss')
    parser.add_argument('--readable', action='store_true', help='Optimize for readable adversarial strings using prompt injection')
    
    # Resume functionality
    parser.add_argument('--resume', action='store_true', help='Resume from existing results if available')
    
    # Logging and Output
    parser.add_argument('--verbosity', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help='Logging verbosity level')
    parser.add_argument('--output_dir', type=str, default="results/security_copilot/", help='Directory to save attack results')
    
    return parser.parse_args()


def get_data_path(model_name: str, mode: str, task: str) -> str:
    """Determine data file path based on model name, mode, and task."""
    return f"data/security_copilot/{model_name}_{mode}_{task}.json"


def generate_output_path(model_id: str, task: str, mode: str, output_dir: str, args: argparse.Namespace) -> str:
    """Generate output file path based on configuration."""
    sanitized_model_id = sanitize_filename(model_id)
    
    # Create descriptive filename with key parameters
    filename_parts = [
        sanitized_model_id,
        f"security_copilot_{task}",
        f"steps{args.num_steps}",
        f"width{args.search_width}",
        f"topk{args.topk}",
        f"replace{args.n_replace}",
        f"locations{args.num_location}",
        f"weight{args.weight}",
        "sequential" if args.sequential else "joint",
        "readable" if args.readable else "standard",
        mode
    ]
    
    filename = "_".join(filename_parts) + ".pkl"
    
    # Create output directory structure
    path = os.path.join(output_dir, mode)
    os.makedirs(path, exist_ok=True)
    
    return os.path.join(path, filename)


def prepare_data(data_path: str, mode: str) -> Tuple[List[str], List[str]]:
    """Load and prepare Security Copilot data for attack."""
    print(f"Loading data from {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    with open(data_path, 'r') as file:
        data = json.load(file)
    
    security_scenarios = data['security_scenarios']
    
    if mode == 'train':
        # Process training data - use all scenarios (filtered to not trigger without optimization)
        goals = [
            scenario['prompt'].replace('{optim_str}', '') + ' {optim_str}'
            for scenario in security_scenarios
        ]
        targets = [scenario['target_functions'][0] for scenario in security_scenarios]
    else:  # validation mode
        # For validation, use only the last scenario to obtain optimal initialization strings for training
        last_scenario = security_scenarios[-1]
        goals = [last_scenario['prompt'].replace('{optim_str}', '') + ' {optim_str}']
        targets = [last_scenario['target_functions'][0]]
    
    print(f"Loaded {len(goals)} security scenarios for {mode} mode")
    return goals, targets


def create_udora_config(args: argparse.Namespace, model_name: str, task: str) -> UDoraConfig:
    """Create UDora configuration from arguments."""
    return UDoraConfig(
        num_steps=args.num_steps,
        search_width=args.search_width,
        topk=args.topk,
        n_replace=args.n_replace,
        buffer_size=args.buffer_size,
        weight=args.weight,
        use_mellowmax=args.use_mellowmax,
        optim_str_init=args.optim_str_init,
        num_location=args.num_location,
        sequential=args.sequential,
        prefix_update_frequency=args.prefix_update_frequency,
        add_space_before_target=args.add_space_before_target,
        early_stop=args.early_stop,
        max_new_tokens=args.max_new_tokens,
        readable=args.readable,
        dataset="security_copilot",
        verbosity=args.verbosity
    )


def run_security_copilot_attack(model, tokenizer, goals: List[str], targets: List[str], 
                               config: UDoraConfig, output_path: str, resume: bool):
    """Run Security Copilot attack using UDora framework."""
    print(f"Starting Security Copilot attack with {len(goals)} scenarios")
    print(f"Target functions: {set(targets)}")
    
    results = []
    success_count = 0
    
    # Resume from existing results if requested
    if resume and os.path.exists(output_path):
        print(f"Resuming from existing results: {output_path}")
        with open(output_path, 'rb') as f:
            existing_results = pickle.load(f)
        start_idx = len(existing_results)
        results = existing_results
        success_count = sum(1 for r in results if r.get('success', False))
    else:
        start_idx = 0
    
    # Run attack on remaining scenarios
    for i in tqdm(range(start_idx, len(goals)), desc="Security Copilot Attack Progress"):
        goal = goals[i]
        target = targets[i]
        
        print(f"\n--- Scenario {i+1}/{len(goals)} ---")
        print(f"Target function: {target}")
        print(f"Goal prompt: {goal[:100]}...")
        
        try:
            # Run UDora attack
            result = udora_run(
                model=model,
                tokenizer=tokenizer,
                goal=goal,
                target=target,
                config=config
            )
            
            # Check if attack succeeded
            success = result.success if hasattr(result, 'success') else False
            if success:
                success_count += 1
                print(f"✓ Attack succeeded! Target '{target}' triggered.")
            else:
                print(f"✗ Attack failed. Target '{target}' not triggered.")
            
            # Store result
            result_dict = {
                'scenario_id': i,
                'goal': goal,
                'target': target,
                'success': success,
                'best_loss': result.best_loss if hasattr(result, 'best_loss') else None,
                'num_steps': result.num_steps if hasattr(result, 'num_steps') else None,
                'adversarial_string': result.best_string if hasattr(result, 'best_string') else None
            }
            results.append(result_dict)
            
            # Save incremental results
            with open(output_path, 'wb') as f:
                pickle.dump(results, f)
                
        except Exception as e:
            print(f"Error processing scenario {i}: {str(e)}")
            # Store failed result
            results.append({
                'scenario_id': i,
                'goal': goal,
                'target': target,
                'success': False,
                'error': str(e)
            })
    
    # Print final results
    print_final_results(success_count, len(results), output_path)


def print_final_results(success_count: int, total_count: int, output_path: str):
    """Print final attack results summary."""
    success_rate = (success_count / total_count) * 100 if total_count > 0 else 0
    
    print("\n" + "="*60)
    print("SECURITY COPILOT ATTACK RESULTS")
    print("="*60)
    print(f"Total scenarios: {total_count}")
    print(f"Successful attacks: {success_count}")
    print(f"Success rate: {success_rate:.2f}%")
    print(f"Results saved to: {output_path}")
    print("="*60)


def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Determine model name and data path
    model_name = get_model_name(args.model_id)
    data_path = get_data_path(model_name, args.mode, args.task)
    
    # Generate output path
    output_path = generate_output_path(args.model_id, args.task, args.mode, args.output_dir, args)
    print(f"Configuration: Model={model_name}, Task={args.task}, Mode={args.mode}")
    print(f"Data path: {data_path}")
    print(f"Output path: {output_path}")

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_id)

    # Prepare data
    goals, targets = prepare_data(data_path, args.mode)

    # Create UDora configuration
    config = create_udora_config(args, model_name, args.task)

    # Run attack
    run_security_copilot_attack(model, tokenizer, goals, targets, config, output_path, args.resume)


if __name__ == "__main__":
    main()