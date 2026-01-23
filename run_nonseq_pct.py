"""
Non-Sequential MCTS-PCT Inference Script

Run bin packing with non-sequential package selection:
1. Select optimal package from K candidates using Set Transformer
2. Place selected package using MCTS-PCT

Usage:
    python run_nonseq_pct.py --model-path ./logs/experiment/xxx/PCT-xxx.pt \
        --selector-path ./logs/selector/best.pt --setting 1
    
    # Without trained selector (uses PCT-guided selection)
    python run_nonseq_pct.py --model-path ./logs/experiment/xxx/PCT-xxx.pt \
        --setting 1 --use-pct-guidance
"""

import sys
import os
import argparse
import time
import json
import torch
import numpy as np
import random
from datetime import datetime
from collections import deque
from typing import List, Tuple, Dict, Any, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import DRL_GAT
from tools import load_policy, registration_envs
from mcts_pct import MCTSPlannerPCT, MCTSConfig
from set_transformer.package_selector import (
    PackageSelector, 
    BinStateEncoder, 
    PackageFeatureEncoder,
    PCTGuidedLabelGenerator
)
import givenData


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Non-Sequential MCTS-PCT Inference')
    
    # Model paths
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to pretrained PCT model (.pt file)')
    parser.add_argument('--selector-path', type=str, default=None,
                        help='Path to trained PackageSelector model')
    parser.add_argument('--setting', type=int, default=1,
                        help='Experiment setting (1/2/3)')
    
    # Non-sequential parameters
    parser.add_argument('--queue-size', type=int, default=5,
                        help='Candidate queue size K')
    parser.add_argument('--use-pct-guidance', action='store_true',
                        help='Use PCT-guided selection instead of trained selector')
    
    # MCTS parameters
    parser.add_argument('--n-simulations', type=int, default=50,
                        help='Number of MCTS simulations per decision')
    parser.add_argument('--lookahead', type=int, default=4,
                        help='Lookahead horizon N (tree depth)')
    parser.add_argument('--c-puct', type=float, default=1.0,
                        help='PUCT exploration constant')
    parser.add_argument('--wsp-weight', type=float, default=0.5,
                        help='Wasted Space Penalty weight λ')
    
    # Environment settings
    parser.add_argument('--num-items', type=int, default=100,
                        help='Number of items per episode')
    parser.add_argument('--num-episodes', type=int, default=1,
                        help='Number of episodes to run')
    parser.add_argument('--lnes', type=str, default='EMS',
                        help='Leaf Node Expansion Scheme')
    
    # Network architecture (must match pretrained model)
    parser.add_argument('--embedding-size', type=int, default=64)
    parser.add_argument('--hidden-size', type=int, default=128)
    parser.add_argument('--gat-layer-num', type=int, default=1)
    parser.add_argument('--internal-node-holder', type=int, default=80)
    parser.add_argument('--leaf-node-holder', type=int, default=50)
    
    # Selector model architecture
    parser.add_argument('--selector-hidden-dim', type=int, default=128)
    parser.add_argument('--selector-num-heads', type=int, default=4)
    parser.add_argument('--bin-state-dim', type=int, default=32)
    parser.add_argument('--package-dim', type=int, default=8)
    
    # Device
    parser.add_argument('--no-cuda', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('--device', type=int, default=0,
                        help='GPU device ID')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Dataset (optional)
    parser.add_argument('--load-dataset', action='store_true',
                        help='Load items from dataset file')
    parser.add_argument('--dataset-path', type=str, default=None,
                        help='Path to dataset file (.pt)')
    
    # Logging
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed progress')
    
    # Save results
    parser.add_argument('--save-results', action='store_true',
                        help='Save episode results to JSON')
    parser.add_argument('--result-dir', type=str, default='./nonseq_results',
                        help='Directory to save results')
    
    args = parser.parse_args()
    
    # Derive additional parameters
    args.container_size = givenData.container_size
    args.item_size_set = givenData.item_size_set
    
    if args.setting == 1:
        args.internal_node_length = 6
    elif args.setting == 2:
        args.internal_node_length = 6
    elif args.setting == 3:
        args.internal_node_length = 7
    
    args.normFactor = 1.0 / np.max(args.container_size)
    
    return args


def create_environment(args):
    """Create packing environment."""
    from pct_envs.PctContinuous0.bin3D import PackingContinuous
    
    env = PackingContinuous(
        setting=args.setting,
        container_size=args.container_size,
        item_set=args.item_size_set,
        internal_node_holder=args.internal_node_holder,
        leaf_node_holder=args.leaf_node_holder,
        next_holder=1,
        LNES=args.lnes,
    )
    env.normFactor = args.normFactor
    
    return env


def load_pct_model(args, device):
    """Load pretrained PCT model."""
    pct_policy = DRL_GAT(args)
    pct_policy = load_policy(args.model_path, pct_policy)
    pct_policy = pct_policy.to(device)
    pct_policy.eval()
    
    print(f"Loaded PCT model from: {args.model_path}")
    return pct_policy


def load_selector_model(args, device):
    """Load trained PackageSelector model."""
    if args.selector_path is None or not os.path.exists(args.selector_path):
        return None
    
    selector = PackageSelector(
        bin_state_dim=args.bin_state_dim,
        package_dim=args.package_dim,
        hidden_dim=args.selector_hidden_dim,
        num_heads=args.selector_num_heads,
    )
    
    checkpoint = torch.load(args.selector_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        selector.load_state_dict(checkpoint['model_state_dict'])
    else:
        selector.load_state_dict(checkpoint)
    
    selector = selector.to(device)
    selector.eval()
    
    print(f"Loaded PackageSelector from: {args.selector_path}")
    return selector


def generate_items(args, env):
    """Generate item sequence for episode."""
    items = []
    for _ in range(args.num_items):
        item_sizes = random.choice(args.item_size_set)
        if args.setting == 3:
            density = random.random()
            while density == 0:
                density = random.random()
            items.append((*item_sizes, density))
        else:
            items.append(item_sizes)
    return items


def load_items_from_dataset(args, trajectory_idx=0):
    """Load items from dataset file."""
    if not args.dataset_path or not os.path.exists(args.dataset_path):
        raise ValueError(f"Dataset not found: {args.dataset_path}")
    
    data = torch.load(args.dataset_path)
    
    if isinstance(data, list):
        if trajectory_idx >= len(data):
            trajectory_idx = trajectory_idx % len(data)
        
        traj = data[trajectory_idx]
        
        if isinstance(traj, dict) and 'items' in traj:
            items = traj['items']
        elif isinstance(traj, (list, tuple)):
            items = list(traj)
        else:
            items = [traj]
        
        processed_items = []
        for item in items[:args.num_items]:
            if hasattr(item, 'tolist'):
                processed_items.append(item.tolist())
            elif isinstance(item, (list, tuple)):
                processed_items.append(list(item))
            else:
                processed_items.append(item)
        
        return processed_items
    
    elif isinstance(data, dict):
        if 'items' in data:
            return data['items'][:args.num_items]
    
    return data[:args.num_items]


def save_episode_results(boxes, container_size, episode_idx, results, output_dir='./nonseq_results'):
    """Save episode results to JSON."""
    date_str = datetime.now().strftime('%Y-%m-%d')
    folder_path = os.path.join(output_dir, date_str)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    cube_definition = []
    for i, box in enumerate(boxes):
        cube_definition.append([
            float(box.x), float(box.y), float(box.z),
            float(box.lx), float(box.ly), float(box.lz),
            i + 1
        ])
    
    data = {
        'episode': episode_idx,
        'mode': 'non-sequential',
        'container_size': list(container_size),
        'cube_definition': cube_definition,
        'statistics': results
    }
    
    time_str = datetime.now().strftime('%H_%M_%S')
    file_name = f'nonseq_episode_{episode_idx}_{time_str}.json'
    file_path = os.path.join(folder_path, file_name)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to: {file_path}")
    return file_path


def run_episode_nonseq(args, env, selector, planner, pct_policy, 
                        bin_encoder, pkg_encoder, all_items, 
                        episode_idx=0, device='cuda'):
    """
    Run single episode with non-sequential package selection.
    
    Two-stage decision process:
    1. Select optimal package from K candidates (Set Transformer or PCT-guided)
    2. Find optimal placement for selected package (MCTS-PCT)
    """
    env.reset()
    
    total_items = len(all_items)
    packed_count = 0
    total_reward = 0.0
    step_times = []
    selection_stats = []
    
    # Initialize candidate queue
    candidate_queue = deque(maxlen=args.queue_size)
    item_source = iter(all_items)
    
    # Fill initial queue
    for _ in range(args.queue_size):
        try:
            candidate_queue.append(next(item_source))
        except StopIteration:
            break
    
    print(f"\n{'='*60}")
    print(f"Episode {episode_idx + 1}: {total_items} items (Non-Sequential)")
    print(f"Container size: {args.container_size}")
    print(f"Queue size K={args.queue_size}")
    print(f"MCTS config: n={args.n_simulations}, N={args.lookahead}")
    print(f"{'='*60}")
    
    step_idx = 0
    
    while len(candidate_queue) > 0:
        step_start = time.time()
        step_idx += 1
        
        K = len(candidate_queue)
        candidates = list(candidate_queue)
        
        # ========== Stage 1: Package Selection ==========
        current_state = env.space.get_state()
        
        if args.use_pct_guidance or selector is None:
            # Use PCT-guided selection
            label_gen = PCTGuidedLabelGenerator(pct_policy, env, device=str(device))
            selected_idx, probs = label_gen.generate_labels(current_state, candidates)
            env.space.set_state(current_state)  # Restore state
        else:
            # Use trained selector
            bin_state = bin_encoder.extract(env)
            bin_state_tensor = torch.FloatTensor(bin_state).unsqueeze(0).to(device)
            
            pkg_features = pkg_encoder.encode(candidates)
            pkg_tensor = torch.FloatTensor(pkg_features).unsqueeze(0).to(device)
            
            with torch.no_grad():
                selected_idx, probs = selector.select(
                    bin_state_tensor, 
                    pkg_tensor,
                    deterministic=True
                )
            selected_idx = selected_idx.item()
            probs = probs.cpu().numpy()[0]
        
        selected_package = candidates[selected_idx]
        
        # ========== Stage 2: Placement Selection ==========
        # Set selected package in environment
        env.next_box = list(selected_package[:3])
        env.next_den = selected_package[3] if len(selected_package) > 3 else 1.0
        
        # Restore state before MCTS
        env.space.set_state(current_state)
        
        # Run MCTS search for placement
        best_action, mcts_stats = planner.search(current_state, [selected_package])
        
        # Restore state after MCTS
        env.space.set_state(current_state)
        
        # ========== Execute Placement ==========
        placement_success = False
        
        if best_action is not None and 'error' not in mcts_stats:
            leaf_node_vec = env.get_possible_position()
            
            if best_action < len(leaf_node_vec) and np.sum(leaf_node_vec[best_action]) > 0:
                leaf_node = leaf_node_vec[best_action]
                action, next_box = env.LeafNode2Action(leaf_node)
                
                idx = (round(action[1], 6), round(action[2], 6))
                rotation_flag = action[0]
                
                succeeded = env.space.drop_box(
                    next_box, idx, rotation_flag, 
                    env.next_den, args.setting
                )
                
                if succeeded:
                    # Update EMS
                    packed_box = env.space.boxes[-1]
                    if hasattr(env, 'LNES') and env.LNES == 'EMS':
                        env.space.GENEMS([
                            packed_box.lx, packed_box.ly, packed_box.lz,
                            round(packed_box.lx + packed_box.x, 6),
                            round(packed_box.ly + packed_box.y, 6),
                            round(packed_box.lz + packed_box.z, 6)
                        ])
                    
                    packed_count += 1
                    placement_success = True
                    
                    box_volume = next_box[0] * next_box[1] * next_box[2]
                    bin_volume = args.container_size[0] * args.container_size[1] * args.container_size[2]
                    reward = (box_volume / bin_volume) * 10
                    total_reward += reward
        
        step_time = time.time() - step_start
        step_times.append(step_time)
        
        # Record selection statistics
        selection_stats.append({
            'step': step_idx,
            'selected_idx': selected_idx,
            'selected_package': selected_package[:3],
            'queue_size': K,
            'success': placement_success
        })
        
        if placement_success:
            # Remove placed package from queue
            del candidate_queue[selected_idx]
            
            # Add new package to queue
            try:
                candidate_queue.append(next(item_source))
            except StopIteration:
                pass
            
            if args.verbose or step_idx % 10 == 0:
                ratio = env.space.get_ratio()
                print(f"  Step {step_idx:3d}: selected P{selected_idx+1} {selected_package[:3]}, "
                      f"ratio={ratio:.4f}, time={step_time:.2f}s")
        else:
            # Placement failed - try to remove this package and continue
            if args.verbose:
                print(f"  Step {step_idx:3d}: placement failed for P{selected_idx+1}")
            
            # Remove failed package
            del candidate_queue[selected_idx]
            
            # Try to add new package
            try:
                candidate_queue.append(next(item_source))
            except StopIteration:
                pass
    
    # Final statistics
    final_ratio = env.space.get_ratio()
    avg_step_time = np.mean(step_times) if step_times else 0.0
    
    results = {
        'episode': episode_idx,
        'mode': 'non-sequential',
        'packed_count': packed_count,
        'total_items': total_items,
        'space_ratio': final_ratio,
        'total_reward': total_reward,
        'avg_step_time': avg_step_time,
        'total_time': sum(step_times),
        'queue_size': args.queue_size,
        'timestamp': datetime.now().isoformat()
    }
    
    print(f"\n{'='*60}")
    print(f"Episode {episode_idx + 1} Results (Non-Sequential):")
    print(f"  Packed: {packed_count}/{total_items} items")
    print(f"  Space Utilization: {final_ratio:.4f} ({final_ratio*100:.2f}%)")
    print(f"  Total Reward: {total_reward:.4f}")
    print(f"  Avg Step Time: {avg_step_time:.3f}s")
    print(f"  Total Time: {sum(step_times):.2f}s")
    print(f"{'='*60}")
    
    return results


def main():
    """Main entry point."""
    args = get_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Setup device
    if args.no_cuda:
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'#'*60}")
    print("Non-Sequential MCTS-PCT Inference")
    print(f"{'#'*60}")
    print(f"Device: {device}")
    print(f"PCT Model: {args.model_path}")
    print(f"Selector Model: {args.selector_path or 'PCT-Guided'}")
    print(f"Setting: {args.setting}")
    print(f"Queue Size: K={args.queue_size}")
    
    # Register environments
    registration_envs()
    
    # Create environment
    env = create_environment(args)
    print(f"Environment created: {args.container_size}")
    
    # Load PCT model
    pct_policy = load_pct_model(args, device)
    
    # Load selector model (or use PCT-guided)
    selector = None
    if not args.use_pct_guidance and args.selector_path:
        selector = load_selector_model(args, device)
    
    if selector is None:
        print("Using PCT-guided package selection")
        args.use_pct_guidance = True
    
    # Create feature encoders
    bin_encoder = BinStateEncoder(
        state_dim=args.bin_state_dim,
        container_size=tuple(args.container_size)
    )
    pkg_encoder = PackageFeatureEncoder(
        package_dim=args.package_dim,
        container_size=tuple(args.container_size)
    )
    
    # Create MCTS config
    config = MCTSConfig(
        lookahead_horizon=args.lookahead,
        n_simulations=args.n_simulations,
        discount_factor=1.0,
        wsp_weight=args.wsp_weight,
        c_puct=args.c_puct,
    )
    
    # Create MCTS planner
    planner = MCTSPlannerPCT(
        pct_policy=pct_policy,
        env=env,
        config=config,
        device=str(device),
    )
    print("MCTS planner initialized")
    
    # Run episodes
    all_results = []
    
    for ep in range(args.num_episodes):
        # Generate or load items
        if args.load_dataset:
            items = load_items_from_dataset(args, trajectory_idx=ep)
            print(f"\nLoaded trajectory {ep + 1} with {len(items)} items")
        else:
            items = generate_items(args, env)
        
        # Run non-sequential episode
        results = run_episode_nonseq(
            args, env, selector, planner, pct_policy,
            bin_encoder, pkg_encoder, items, ep, device
        )
        all_results.append(results)
        
        # Save results
        if args.save_results:
            save_episode_results(
                boxes=env.space.boxes,
                container_size=args.container_size,
                episode_idx=ep,
                results=results,
                output_dir=args.result_dir
            )
    
    # Summary statistics
    if len(all_results) > 1:
        avg_ratio = np.mean([r['space_ratio'] for r in all_results])
        std_ratio = np.std([r['space_ratio'] for r in all_results])
        avg_packed = np.mean([r['packed_count'] for r in all_results])
        avg_step_time = np.mean([r['avg_step_time'] for r in all_results])
        total_time = sum([r.get('total_time', 0) for r in all_results])
        
        print(f"\n{'#'*60}")
        print("Summary over all episodes (Non-Sequential):")
        print(f"  Episodes: {len(all_results)}")
        print(f"  Average Space Ratio: {avg_ratio:.4f} ({avg_ratio*100:.2f}%) ± {std_ratio:.4f}")
        print(f"  Average Packed: {avg_packed:.1f} / {args.num_items} items")
        print(f"  Average Step Time: {avg_step_time:.2f}s")
        print(f"  Total Time: {total_time:.1f}s")
        print(f"{'#'*60}")
    
    return all_results


if __name__ == '__main__':
    main()
