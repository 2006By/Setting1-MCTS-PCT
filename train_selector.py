"""
Train Package Selector Model

Trains the PackageSelector using hybrid strategy:
1. PCT-Guided Imitation Learning: Use PCT Critic values as soft labels
2. Optional RL Fine-tuning: PPO-style policy gradient

Usage:
    python train_selector.py --model-path ./logs/experiment/xxx/PCT-xxx.pt --setting 1
    
    # With custom parameters
    python train_selector.py --model-path ./logs/experiment/xxx/PCT-xxx.pt \
        --setting 1 --epochs 100 --batch-size 32 --lr 1e-4
"""

import sys
import os
import argparse
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from datetime import datetime
from collections import deque
from typing import List, Tuple, Dict, Any, Optional
from torch.utils.tensorboard import SummaryWriter

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import DRL_GAT
from tools import load_policy, registration_envs
from set_transformer.package_selector import (
    PackageSelector,
    BinStateEncoder,
    PackageFeatureEncoder,
    PCTGuidedLabelGenerator
)
import givenData


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Package Selector')
    
    # Model paths
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to pretrained PCT model (.pt file)')
    parser.add_argument('--setting', type=int, default=1,
                        help='Experiment setting (1/2/3)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--episodes-per-epoch', type=int, default=10,
                        help='Episodes to collect per epoch')
    parser.add_argument('--steps-per-episode', type=int, default=50,
                        help='Steps per episode for data collection')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='Weight decay')
    
    # Selector model architecture
    parser.add_argument('--hidden-dim', type=int, default=128,
                        help='Hidden dimension')
    parser.add_argument('--num-heads', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--num-layers', type=int, default=2,
                        help='Number of Set Attention layers')
    parser.add_argument('--bin-state-dim', type=int, default=32,
                        help='Bin state feature dimension')
    parser.add_argument('--package-dim', type=int, default=8,
                        help='Package feature dimension')
    
    # Non-sequential parameters
    parser.add_argument('--queue-size', type=int, default=5,
                        help='Candidate queue size K')
    parser.add_argument('--num-items', type=int, default=100,
                        help='Number of items per episode')
    
    # Training strategy
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Temperature for soft labels')
    parser.add_argument('--label-smoothing', type=float, default=0.1,
                        help='Label smoothing factor')
    parser.add_argument('--use-kl-loss', action='store_true',
                        help='Use KL divergence loss (default: cross-entropy)')
    
    # RL fine-tuning (optional)
    parser.add_argument('--rl-finetune', action='store_true',
                        help='Enable RL fine-tuning after imitation')
    parser.add_argument('--rl-epochs', type=int, default=20,
                        help='RL fine-tuning epochs')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor for RL')
    parser.add_argument('--entropy-coef', type=float, default=0.01,
                        help='Entropy coefficient for exploration')
    
    # Environment settings
    parser.add_argument('--lnes', type=str, default='EMS',
                        help='Leaf Node Expansion Scheme')
    
    # Network architecture (must match pretrained model)
    parser.add_argument('--embedding-size', type=int, default=64)
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
    
    # Logging and checkpoints
    parser.add_argument('--log-dir', type=str, default='./logs/selector',
                        help='TensorBoard log directory')
    parser.add_argument('--save-dir', type=str, default='./logs/selector',
                        help='Model save directory')
    parser.add_argument('--save-interval', type=int, default=10,
                        help='Save model every N epochs')
    parser.add_argument('--log-interval', type=int, default=1,
                        help='Log every N epochs')
    
    args = parser.parse_args()
    
    # Derive additional parameters
    args.container_size = givenData.container_size
    args.item_size_set = givenData.item_size_set
    args.hidden_size = 128
    
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
    
    return pct_policy


def generate_items(args):
    """Generate item sequence."""
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


class ReplayBuffer:
    """Experience replay buffer for training data."""
    
    def __init__(self, max_size: int = 10000):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, bin_state: np.ndarray, pkg_features: np.ndarray, 
            soft_labels: np.ndarray, hard_label: int):
        """Add experience to buffer."""
        self.buffer.append((bin_state, pkg_features, soft_labels, hard_label))
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample a batch of experiences."""
        indices = np.random.choice(len(self.buffer), min(batch_size, len(self.buffer)), replace=False)
        
        batch = [self.buffer[i] for i in indices]
        
        bin_states = torch.FloatTensor(np.stack([b[0] for b in batch]))
        pkg_features = torch.FloatTensor(np.stack([b[1] for b in batch]))
        soft_labels = torch.FloatTensor(np.stack([b[2] for b in batch]))
        hard_labels = torch.LongTensor([b[3] for b in batch])
        
        return bin_states, pkg_features, soft_labels, hard_labels
    
    def __len__(self):
        return len(self.buffer)


def collect_data(args, env, pct_policy, bin_encoder, pkg_encoder, 
                 label_generator, buffer, device):
    """
    Collect training data by running episodes.
    
    For each step, generate PCT-guided labels for the candidate packages.
    """
    env.reset()
    
    # Initialize candidate queue
    items = generate_items(args)
    candidate_queue = deque(maxlen=args.queue_size)
    item_source = iter(items)
    
    for _ in range(args.queue_size):
        try:
            candidate_queue.append(next(item_source))
        except StopIteration:
            break
    
    steps_collected = 0
    
    while len(candidate_queue) > 0 and steps_collected < args.steps_per_episode:
        candidates = list(candidate_queue)
        current_state = env.space.get_state()
        
        # Extract features
        bin_state = bin_encoder.extract(env)
        pkg_features = pkg_encoder.encode(candidates)
        
        # Generate PCT-guided labels
        hard_label, soft_labels = label_generator.generate_labels(
            current_state, candidates
        )
        env.space.set_state(current_state)
        
        # Add to buffer
        buffer.add(bin_state, pkg_features, soft_labels, hard_label)
        steps_collected += 1
        
        # Simulate placing the selected package
        selected_pkg = candidates[hard_label]
        env.next_box = list(selected_pkg[:3])
        env.next_den = selected_pkg[3] if len(selected_pkg) > 3 else 1.0
        
        leaf_nodes = env.get_possible_position()
        valid_mask = leaf_nodes[:, 8] > 0
        
        if valid_mask.sum() > 0:
            # Place at first valid position
            first_valid = np.where(valid_mask)[0][0]
            leaf = leaf_nodes[first_valid]
            action, next_box = env.LeafNode2Action(leaf)
            
            success = env.space.drop_box(
                next_box, (action[1], action[2]), action[0],
                env.next_den, args.setting
            )
            
            if success:
                # Update EMS
                packed_box = env.space.boxes[-1]
                if hasattr(env, 'LNES') and env.LNES == 'EMS':
                    env.space.GENEMS([
                        packed_box.lx, packed_box.ly, packed_box.lz,
                        round(packed_box.lx + packed_box.x, 6),
                        round(packed_box.ly + packed_box.y, 6),
                        round(packed_box.lz + packed_box.z, 6)
                    ])
                
                # Remove from queue and add new
                del candidate_queue[hard_label]
                try:
                    candidate_queue.append(next(item_source))
                except StopIteration:
                    pass
            else:
                # Remove failed package
                del candidate_queue[hard_label]
                try:
                    candidate_queue.append(next(item_source))
                except StopIteration:
                    pass
        else:
            # No valid placement, skip
            del candidate_queue[hard_label]
            try:
                candidate_queue.append(next(item_source))
            except StopIteration:
                pass
    
    return steps_collected


def train_epoch(args, selector, optimizer, buffer, device, use_kl=False):
    """Train for one epoch."""
    selector.train()
    
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0
    
    # Number of batches per epoch
    n_batches = max(1, len(buffer) // args.batch_size)
    
    for _ in range(n_batches):
        bin_states, pkg_features, soft_labels, hard_labels = buffer.sample(args.batch_size)
        
        bin_states = bin_states.to(device)
        pkg_features = pkg_features.to(device)
        soft_labels = soft_labels.to(device)
        hard_labels = hard_labels.to(device)
        
        optimizer.zero_grad()
        
        scores, probs = selector(bin_states, pkg_features)
        
        if use_kl:
            # KL divergence loss
            log_probs = F.log_softmax(scores, dim=-1)
            loss = F.kl_div(log_probs, soft_labels, reduction='batchmean')
        else:
            # Cross-entropy with label smoothing
            if args.label_smoothing > 0:
                # Soft cross-entropy
                log_probs = F.log_softmax(scores, dim=-1)
                # Interpolate between hard and soft labels
                targets = (1 - args.label_smoothing) * F.one_hot(hard_labels, scores.size(-1)).float() \
                          + args.label_smoothing * soft_labels
                loss = -(targets * log_probs).sum(dim=-1).mean()
            else:
                loss = F.cross_entropy(scores, hard_labels)
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(selector.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Calculate accuracy
        preds = scores.argmax(dim=-1)
        acc = (preds == hard_labels).float().mean()
        
        total_loss += loss.item()
        total_acc += acc.item()
        num_batches += 1
    
    return total_loss / num_batches, total_acc / num_batches


def evaluate(args, selector, env, pct_policy, bin_encoder, pkg_encoder, device):
    """Evaluate selector on full episodes."""
    selector.eval()
    env.reset()
    
    items = generate_items(args)
    candidate_queue = deque(maxlen=args.queue_size)
    item_source = iter(items)
    
    for _ in range(args.queue_size):
        try:
            candidate_queue.append(next(item_source))
        except StopIteration:
            break
    
    packed_count = 0
    
    while len(candidate_queue) > 0:
        candidates = list(candidate_queue)
        
        # Get selector prediction
        bin_state = bin_encoder.extract(env)
        pkg_features = pkg_encoder.encode(candidates)
        
        bin_state_tensor = torch.FloatTensor(bin_state).unsqueeze(0).to(device)
        pkg_tensor = torch.FloatTensor(pkg_features).unsqueeze(0).to(device)
        
        with torch.no_grad():
            selected_idx, _ = selector.select(bin_state_tensor, pkg_tensor, deterministic=True)
        selected_idx = selected_idx.item()
        
        # Try to place
        selected_pkg = candidates[selected_idx]
        env.next_box = list(selected_pkg[:3])
        env.next_den = selected_pkg[3] if len(selected_pkg) > 3 else 1.0
        
        leaf_nodes = env.get_possible_position()
        valid_mask = leaf_nodes[:, 8] > 0
        
        if valid_mask.sum() > 0:
            first_valid = np.where(valid_mask)[0][0]
            leaf = leaf_nodes[first_valid]
            action, next_box = env.LeafNode2Action(leaf)
            
            success = env.space.drop_box(
                next_box, (action[1], action[2]), action[0],
                env.next_den, args.setting
            )
            
            if success:
                packed_box = env.space.boxes[-1]
                if hasattr(env, 'LNES') and env.LNES == 'EMS':
                    env.space.GENEMS([
                        packed_box.lx, packed_box.ly, packed_box.lz,
                        round(packed_box.lx + packed_box.x, 6),
                        round(packed_box.ly + packed_box.y, 6),
                        round(packed_box.lz + packed_box.z, 6)
                    ])
                packed_count += 1
        
        del candidate_queue[selected_idx]
        try:
            candidate_queue.append(next(item_source))
        except StopIteration:
            pass
    
    return env.space.get_ratio(), packed_count


def main():
    """Main training loop."""
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
    
    # Create directories
    timestamp = datetime.now().strftime('%Y.%m.%d-%H-%M-%S')
    save_dir = os.path.join(args.save_dir, f'selector-{timestamp}')
    log_dir = os.path.join(args.log_dir, f'selector-{timestamp}')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"\n{'#'*60}")
    print("Training Package Selector")
    print(f"{'#'*60}")
    print(f"Device: {device}")
    print(f"PCT Model: {args.model_path}")
    print(f"Save Dir: {save_dir}")
    print(f"Queue Size: K={args.queue_size}")
    
    # Setup TensorBoard
    writer = SummaryWriter(log_dir)
    
    # Register environments
    registration_envs()
    
    # Create environment
    env = create_environment(args)
    print(f"Environment created: {args.container_size}")
    
    # Load PCT model
    pct_policy = load_pct_model(args, device)
    print("PCT model loaded")
    
    # Create selector model
    selector = PackageSelector(
        bin_state_dim=args.bin_state_dim,
        package_dim=args.package_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
    ).to(device)
    print(f"Selector model created: {sum(p.numel() for p in selector.parameters())} parameters")
    
    # Create feature encoders
    bin_encoder = BinStateEncoder(
        state_dim=args.bin_state_dim,
        container_size=tuple(args.container_size)
    )
    pkg_encoder = PackageFeatureEncoder(
        package_dim=args.package_dim,
        container_size=tuple(args.container_size)
    )
    
    # Create label generator
    label_generator = PCTGuidedLabelGenerator(
        pct_policy, env, device=str(device),
        temperature=args.temperature
    )
    
    # Create optimizer
    optimizer = optim.Adam(
        selector.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Create replay buffer
    buffer = ReplayBuffer(max_size=50000)
    
    # Training loop
    best_ratio = 0.0
    
    print(f"\nStarting training for {args.epochs} epochs...")
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        # Collect data
        total_steps = 0
        for _ in range(args.episodes_per_epoch):
            steps = collect_data(
                args, env, pct_policy, bin_encoder, pkg_encoder,
                label_generator, buffer, device
            )
            total_steps += steps
            env.reset()
        
        # Train
        loss, acc = train_epoch(args, selector, optimizer, buffer, device, 
                               use_kl=args.use_kl_loss)
        
        # Evaluate
        eval_ratio, eval_packed = evaluate(
            args, selector, env, pct_policy, bin_encoder, pkg_encoder, device
        )
        
        epoch_time = time.time() - epoch_start
        
        # Logging
        writer.add_scalar('train/loss', loss, epoch)
        writer.add_scalar('train/accuracy', acc, epoch)
        writer.add_scalar('eval/space_ratio', eval_ratio, epoch)
        writer.add_scalar('eval/packed_count', eval_packed, epoch)
        
        if epoch % args.log_interval == 0:
            print(f"Epoch {epoch:4d} | Loss: {loss:.4f} | Acc: {acc:.3f} | "
                  f"Eval Ratio: {eval_ratio:.4f} | Packed: {eval_packed} | "
                  f"Buffer: {len(buffer)} | Time: {epoch_time:.1f}s")
        
        # Save best model
        if eval_ratio > best_ratio:
            best_ratio = eval_ratio
            torch.save({
                'epoch': epoch,
                'model_state_dict': selector.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_ratio': best_ratio,
                'args': vars(args),
            }, os.path.join(save_dir, 'best.pt'))
            print(f"  -> New best model saved! Ratio: {best_ratio:.4f}")
        
        # Save checkpoint
        if epoch % args.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': selector.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'eval_ratio': eval_ratio,
                'args': vars(args),
            }, os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pt'))
    
    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': selector.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'eval_ratio': eval_ratio,
        'args': vars(args),
    }, os.path.join(save_dir, 'final.pt'))
    
    print(f"\n{'#'*60}")
    print("Training Complete!")
    print(f"Best Space Ratio: {best_ratio:.4f}")
    print(f"Models saved to: {save_dir}")
    print(f"{'#'*60}")
    
    writer.close()


if __name__ == '__main__':
    main()
