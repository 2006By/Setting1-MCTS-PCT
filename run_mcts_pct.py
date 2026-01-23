"""
MCTS-PCT Inference Script

Run MCTS-enhanced bin packing using a frozen pretrained PCT model.
This implements the online MPC loop with lookahead planning.

Usage:
    python run_mcts_pct.py --model-path ./logs/experiment/PCT-xxx.pt --setting 2
    
    # With custom parameters
    python run_mcts_pct.py --model-path ./logs/experiment/PCT-xxx.pt \
        --setting 2 --n-simulations 100 --lookahead 4 --wsp-weight 0.5
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

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import DRL_GAT
from tools import load_policy, registration_envs
from mcts_pct import MCTSPlannerPCT, MCTSConfig
import givenData


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='MCTS-PCT Inference')
    
    # Model settings
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to pretrained PCT model (.pt file)')
    parser.add_argument('--setting', type=int, default=2,
                        help='Experiment setting (1/2/3)')
    
    # MCTS parameters
    parser.add_argument('--n-simulations', type=int, default=100,
                        help='Number of MCTS simulations per decision')
    parser.add_argument('--lookahead', type=int, default=4,
                        help='Lookahead horizon N (tree depth)')
    parser.add_argument('--c-puct', type=float, default=1.0,
                        help='PUCT exploration constant')
    parser.add_argument('--wsp-weight', type=float, default=0.5,
                        help='Wasted Space Penalty weight λ')
    
    # Environment settings
    parser.add_argument('--num-items', type=int, default=150,
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
    
    # Save results for visualization
    parser.add_argument('--save-results', action='store_true',
                        help='Save episode results to JSON for visualization in notebook')
    parser.add_argument('--result-dir', type=str, default='./mcts_results',
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


def save_episode_boxes(boxes, container_size, episode_idx, results, output_dir='./mcts_results'):
    """
    Save episode packing result to JSON file for visualization.
    
    The saved format is compatible with the notebook's plot_cube_plotly function:
    cube_definition = [[l, w, h, x, y, z], ...] where:
        l, w, h = box dimensions (size)
        x, y, z = box position (lower corner)
    
    Args:
        boxes: List of Box objects from env.space.boxes
        container_size: [length, width, height] of container
        episode_idx: Episode index
        results: Episode statistics dict
        output_dir: Output directory
        
    Returns:
        str: Path to saved JSON file
    """
    date_str = datetime.now().strftime('%Y-%m-%d')
    folder_path = os.path.join(output_dir, date_str)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # Convert Box objects to the format expected by notebook's plot_cube_plotly:
    # [l, w, h, x, y, z] = [box.x, box.y, box.z, box.lx, box.ly, box.lz]
    cube_definition = []
    for i, box in enumerate(boxes):
        cube_definition.append([
            float(box.x),   # length (l)
            float(box.y),   # width (w) 
            float(box.z),   # height (h)
            float(box.lx),  # x position
            float(box.ly),  # y position
            float(box.lz),  # z position
            i + 1           # label (optional 7th element)
        ])
    
    data = {
        'episode': episode_idx,
        'container_size': list(container_size),
        'cube_definition': cube_definition,  # Ready for plot_cube_plotly()
        'statistics': {
            'packed_count': results.get('packed_count', len(boxes)),
            'total_items': results.get('total_items', 0),
            'space_ratio': results.get('space_ratio', 0),
            'total_reward': results.get('total_reward', 0),
            'timestamp': datetime.now().isoformat()
        }
    }
    
    time_str = datetime.now().strftime('%H_%M_%S')
    file_name = f'episode_{episode_idx}_{time_str}.json'
    file_path = os.path.join(folder_path, file_name)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Episode result saved to: {file_path}")
    return file_path


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
    # Create model architecture
    pct_policy = DRL_GAT(args)
    
    # Load weights
    pct_policy = load_policy(args.model_path, pct_policy)
    pct_policy = pct_policy.to(device)
    pct_policy.eval()
    
    print(f"Loaded PCT model from: {args.model_path}")
    return pct_policy


def generate_items(args, env):
    """Generate item sequence for episode."""
    items = []
    for _ in range(args.num_items):
        # Random selection from item set
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
    """
    Load items from dataset file.
    
    Args:
        args: Arguments with dataset_path and num_items
        trajectory_idx: Which trajectory to load (0-indexed)
        
    Returns:
        List of items from the specified trajectory
    """
    if not args.dataset_path or not os.path.exists(args.dataset_path):
        raise ValueError(f"Dataset not found: {args.dataset_path}")
    
    data = torch.load(args.dataset_path)
    
    # Handle different dataset formats
    if isinstance(data, list):
        # List of trajectories - select specific trajectory
        if trajectory_idx >= len(data):
            trajectory_idx = trajectory_idx % len(data)  # Cycle if needed
        
        traj = data[trajectory_idx]
        
        if isinstance(traj, dict) and 'items' in traj:
            items = traj['items']
        elif isinstance(traj, (list, tuple)):
            items = list(traj)
        else:
            items = [traj]
        
        # Convert to proper format: [[l, w, h], ...] or [(l, w, h), ...]
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


def run_episode(args, env, planner, items, episode_idx=0):
    """
    Run single episode with MCTS-PCT.
    
    Args:
        args: Arguments
        env: PackingDiscrete environment
        planner: MCTSPlannerPCT
        items: List of items [(l,w,h), ...] or [(l,w,h,density), ...]
        episode_idx: Episode number for logging
        
    Returns:
        dict: Episode statistics
    """
    env.reset()
    
    total_items = len(items)
    packed_count = 0
    total_reward = 0.0
    item_idx = 0
    step_times = []
    
    print(f"\n{'='*60}")
    print(f"Episode {episode_idx + 1}: {total_items} items")
    print(f"Container size: {args.container_size}")
    print(f"MCTS config: n={args.n_simulations}, N={args.lookahead}, λ={args.wsp_weight}")
    print(f"{'='*60}")
    
    while item_idx < total_items:
        step_start = time.time()
        
        # Get lookahead window
        lookahead_end = min(item_idx + args.lookahead, total_items)
        lookahead_items = items[item_idx:lookahead_end]
        
        if len(lookahead_items) == 0:
            break
        
        # Set current item in environment
        current_item = lookahead_items[0]
        env.next_box = list(current_item[:3])
        if len(current_item) > 3:
            env.next_den = current_item[3]
        else:
            env.next_den = 1.0
        
        # Get current state
        current_state = env.space.get_state()
        
        # Run MCTS search
        best_action, stats = planner.search(current_state, lookahead_items)
        
        # IMPORTANT: Restore environment state after MCTS search
        # (MCTS modifies the environment during simulations)
        env.space.set_state(current_state)
        
        # Execute best action
        if best_action is not None and 'error' not in stats:
            # Get leaf node for action
            leaf_node_vec = env.get_possible_position()
            
            if best_action < len(leaf_node_vec) and np.sum(leaf_node_vec[best_action]) > 0:
                leaf_node = leaf_node_vec[best_action]
                action, next_box = env.LeafNode2Action(leaf_node)
                
                # Execute placement directly on space (avoid env.step side effects)
                idx = (round(action[1], 6), round(action[2], 6))
                rotation_flag = action[0]
                succeeded = env.space.drop_box(next_box, idx, rotation_flag, env.next_den, args.setting)
                
                if succeeded:
                    # Update EMS after placement
                    packed_box = env.space.boxes[-1]
                    if hasattr(env, 'LNES') and env.LNES == 'EMS':
                        env.space.GENEMS([packed_box.lx, packed_box.ly, packed_box.lz,
                                         round(packed_box.lx + packed_box.x, 6),
                                         round(packed_box.ly + packed_box.y, 6),
                                         round(packed_box.lz + packed_box.z, 6)])
                    
                    packed_count += 1
                    box_volume = next_box[0] * next_box[1] * next_box[2]
                    bin_volume = args.container_size[0] * args.container_size[1] * args.container_size[2]
                    reward = (box_volume / bin_volume) * 10
                    total_reward += reward
                    
                    step_time = time.time() - step_start
                    step_times.append(step_time)
                    
                    if args.verbose or (item_idx + 1) % 10 == 0:
                        ratio = env.space.get_ratio()
                        print(f"  Step {item_idx + 1:3d}: packed item {current_item[:3]}, "
                              f"ratio={ratio:.4f}, visits={stats.get('root_visits', 0)}, "
                              f"time={step_time:.2f}s")
                else:
                    print(f"  Step {item_idx + 1}: Placement failed, bin full")
                    break
            else:
                print(f"  Step {item_idx + 1}: Invalid action {best_action}, skipping")
        else:
            print(f"  Step {item_idx + 1}: No valid action found, skipping")
        
        item_idx += 1
    
    # Final statistics
    final_ratio = env.space.get_ratio()
    avg_step_time = np.mean(step_times) if step_times else 0.0
    
    results = {
        'episode': episode_idx,
        'packed_count': packed_count,
        'total_items': total_items,
        'space_ratio': final_ratio,
        'total_reward': total_reward,
        'avg_step_time': avg_step_time,
        'total_time': sum(step_times),
    }
    
    print(f"\n{'='*60}")
    print(f"Episode {episode_idx + 1} Results:")
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
    print("MCTS-PCT Inference")
    print(f"{'#'*60}")
    print(f"Device: {device}")
    print(f"Model: {args.model_path}")
    print(f"Setting: {args.setting}")
    
    # Register environments
    registration_envs()
    
    # Create environment
    env = create_environment(args)
    print(f"Environment created: {args.container_size}")
    
    # Load PCT model
    pct_policy = load_pct_model(args, device)
    
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
            items = load_items_from_dataset(args, trajectory_idx=ep)  # Each episode uses different trajectory
            print(f"\nLoaded trajectory {ep + 1} with {len(items)} items")
        else:
            items = generate_items(args, env)
        
        # Run episode
        results = run_episode(args, env, planner, items, ep)
        all_results.append(results)
        
        # Save results for visualization in notebook
        if args.save_results:
            save_episode_boxes(
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
        total_time = sum([r.get('total_time', r['avg_step_time'] * r['packed_count']) for r in all_results])
        
        print(f"\n{'#'*60}")
        print("Summary over all episodes:")
        print(f"  Episodes: {len(all_results)}")
        print(f"  Average Space Ratio: {avg_ratio:.4f} ({avg_ratio*100:.2f}%) ± {std_ratio:.4f}")
        print(f"  Average Packed: {avg_packed:.1f} / {args.num_items} items")
        print(f"  Average Step Time: {avg_step_time:.2f}s")
        print(f"  Total Time: {total_time:.1f}s ({total_time/3600:.2f} hours)")
        print(f"{'#'*60}")
    
    return all_results


if __name__ == '__main__':
    main()
