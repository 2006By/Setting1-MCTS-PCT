# -*- coding: utf-8 -*-
"""
缁熶竴璇勪及鑴氭湰
鏀寔璇勪及:
  1. Joint Transformer-PCT 妯″瀷 (--model-path joint_model_xxx.pt)
  2. 绾?PCT 妯″瀷浣滀负 baseline (--pure-pct --model-path PCT-xxx.pt)

涓ょ妯″紡浣跨敤瀹屽叏鐩稿悓鐨勭幆澧冿紙SlidingWindowEnvWrapper锛夊拰鏁版嵁鍔犺浇鏂瑰紡锛?
鍞竴鍖哄埆鏄?selector 鍐崇瓥锛歫oint 妯″瀷鐢?Set Transformer 閫夋嫨锛宲ure PCT 鍥哄畾閫夌涓€涓€?
"""
import sys
import os
import torch
import numpy as np
import argparse
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools import registration_envs, get_leaf_nodes
from joint_model import JointSelectorPCT
from model import DRL_GAT
from sliding_window_env import SlidingWindowEnvWrapper, normalize_to_trajectories
import gym
import givenData


def get_eval_args():
    """鑾峰彇璇勪及鍙傛暟"""
    parser = argparse.ArgumentParser(description='Unified Model Evaluation')
    
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--test-dataset', type=str, default='datasets/test.pt', help='Test dataset path')
    parser.add_argument('--num-episodes', type=int, default=-1, help='Number of episodes to evaluate (-1 for all)')
    parser.add_argument('--setting', type=int, default=1, help='Experiment setting')
    parser.add_argument('--continuous', action='store_true', help='Use continuous environment')
    parser.add_argument('--no-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--device', type=int, default=0, help='GPU device ID')
    parser.add_argument('--window-size', type=int, default=5, help='Window size')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    # 绾?PCT baseline 妯″紡
    parser.add_argument('--pure-pct', action='store_true',
                        help='Evaluate pure PCT model (always select first candidate, no selector)')
    
    # PCT 鍙傛暟
    parser.add_argument('--embedding-size', type=int, default=64)
    parser.add_argument('--hidden-size', type=int, default=128)
    parser.add_argument('--gat-layer-num', type=int, default=1)
    parser.add_argument('--internal-node-holder', type=int, default=100)
    parser.add_argument('--leaf-node-holder', type=int, default=50)
    parser.add_argument('--lnes', type=str, default='EMS')
    # Keep evaluation default aligned with joint training (shuffle=True),
    # and avoid argparse bool parsing pitfalls.
    parser.add_argument('--shuffle', dest='shuffle', action='store_true',
                        help='Enable leaf-node shuffling (default)')
    parser.add_argument('--no-shuffle', dest='shuffle', action='store_false',
                        help='Disable leaf-node shuffling')
    parser.set_defaults(shuffle=True)
    
    args = parser.parse_args()
    
    if args.no_cuda:
        args.device = 'cpu'
    
    args.container_size = givenData.container_size
    args.item_size_set = givenData.item_size_set
    
    if args.continuous:
        args.id = 'PctContinuous-v0'
    else:
        args.id = 'PctDiscrete-v0'
    
    if args.setting == 1:
        args.internal_node_length = 6
    elif args.setting == 2:
        args.internal_node_length = 6
    elif args.setting == 3:
        args.internal_node_length = 7
    
    args.normFactor = 1.0 / np.max(args.container_size)
    
    return args


def create_env(args, seed=0):
    """鍒涘缓璇勪及鐜"""
    env = gym.make(
        args.id,
        setting=args.setting,
        container_size=args.container_size,
        item_set=args.item_size_set,
        data_name=None,
        load_test_data=False,
        internal_node_holder=args.internal_node_holder,
        leaf_node_holder=args.leaf_node_holder,
        LNES=args.lnes,
        shuffle=args.shuffle,
        sample_from_distribution=False
    )
    env.seed(seed)
    return env


def load_pure_pct(args, device):
    """鍔犺浇绾?PCT 妯″瀷锛堝吋瀹?ACKTR 鍜?Adam 璁粌鐨勬潈閲嶏級"""
    model = DRL_GAT(args)
    checkpoint = torch.load(args.model_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # ACKTR 璁粌鐨勬ā鍨嬩細灏?nn.Linear 鎷嗗垎鎴?.module.weight 鍜?.add_bias._bias
    # 闇€瑕佹槧灏勫洖鏍囧噯鐨?.weight 鍜?.bias锛屽苟 squeeze bias 鐨勫浣欑淮搴?
    converted = {}
    for key, value in state_dict.items():
        new_key = key.replace('.module.weight', '.weight') \
                     .replace('.add_bias._bias', '.bias')
        # ACKTR AddBias 瀛樼殑 bias shape 鏄?[N, 1]锛屾爣鍑?nn.Linear 鏄?[N]
        if '.bias' in new_key and value.dim() == 2:
            value = value.squeeze(-1)
        converted[new_key] = value
    
    model.load_state_dict(converted)
    model = model.to(device)
    model.eval()
    return model


def load_joint_model(args, device):
    """Load joint model, compatible with KFAC/ACKTR SplitBias checkpoints."""
    model = JointSelectorPCT(args, window_size=args.window_size)
    checkpoint = torch.load(args.model_path, map_location=device)

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    converted = {}
    for key, value in state_dict.items():
        new_key = key.replace('.module.weight', '.weight') \
                     .replace('.add_bias._bias', '.bias')
        if '.bias' in new_key and isinstance(value, torch.Tensor) and value.dim() == 2:
            value = value.squeeze(-1)
        converted[new_key] = value

    model.load_state_dict(converted)
    model = model.to(device)
    model.eval()
    return model


def evaluate(args):
    """缁熶竴璇勪及鍑芥暟"""
    # 璁剧疆璁惧
    if isinstance(args.device, int):
        device = torch.device('cuda', args.device) if torch.cuda.is_available() else torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    mode_str = "Pure PCT (baseline)" if args.pure_pct else "Joint Transformer-PCT"
    print(f"Evaluation Mode: {mode_str}")
    print(f"Using device: {device}")
    
    # 鍔犺浇鏁版嵁闆嗗苟杞崲涓鸿建杩规牸寮?
    print(f"Loading dataset from {args.test_dataset}...")
    raw_data = torch.load(args.test_dataset)
    trajectories = normalize_to_trajectories(raw_data)
    total_items = sum(len(t) for t in trajectories)
    if args.num_episodes == -1:
        num_episodes = len(trajectories)
    else:
        num_episodes = min(args.num_episodes, len(trajectories))
    print(f"Loaded {len(trajectories)} trajectories, {total_items} total items")
    print(f"Evaluating {num_episodes} episodes (1 episode = 1 trajectory)")
    
    # 鍔犺浇妯″瀷
    print("Loading model...")
    if args.pure_pct:
        pct_model = load_pure_pct(args, device)
        joint_model = None
        print(f"Pure PCT model loaded from {args.model_path}")
    else:
        joint_model = load_joint_model(args, device)
        pct_model = None
        print(f"Joint model loaded from {args.model_path}")
    
    # 鍒涘缓鐜锛堜袱绉嶆ā寮忕敤瀹屽叏鐩稿悓鐨勭幆澧冿級
    env = create_env(args)
    wrapped_env = SlidingWindowEnvWrapper(
        env, trajectories, 
        window_size=args.window_size,
        normFactor=args.normFactor,
        traj_start_idx=0
    )
    
    # 璇勪及缁熻
    results = defaultdict(list)
    selector_infer_time = 0.0
    pct_infer_time = 0.0
    total_decisions = 0
    eval_wall_start = time.perf_counter()

    def _sync_cuda():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    def _get_episode_flatness(wrapper, ep_info):
        # Prefer explicit final score if provided by env info.
        if isinstance(ep_info, dict) and 'final_flatness_score' in ep_info:
            return float(ep_info.get('final_flatness_score', 0.0))
        # Fallback: compute from current packed scene at episode end.
        if hasattr(wrapper, '_compute_flatness_score'):
            try:
                return float(wrapper._compute_flatness_score())
            except Exception:
                pass
        # Last resort: use per-step flatness score in info (may be sparse).
        if isinstance(ep_info, dict) and 'flatness_score' in ep_info:
            return float(ep_info.get('flatness_score', 0.0))
        return 0.0
    
    for ep_idx in range(num_episodes):
        ep_wall_start = time.perf_counter()
        obs = wrapped_env.reset(episode_idx=ep_idx)
        done = False
        step_count = 0
        
        while not done:
            sel_elapsed = 0.0
            if args.pure_pct:
                # ===== 绾?PCT 妯″紡锛氬浐瀹氶€夌涓€涓€欓€夛紙绛変环浜庢寜椤哄簭澶勭悊鐗╁搧锛?=====
                selected_idx_val = 0
            else:
                # ===== Joint 妯″紡锛氱敤 Set Transformer 閫夋嫨 =====
                candidates = torch.FloatTensor(wrapped_env.get_candidates()).unsqueeze(0).to(device)
                container_state = torch.FloatTensor(wrapped_env.get_container_state()).unsqueeze(0).to(device)
                
                _sync_cuda()
                t_sel = time.perf_counter()
                with torch.no_grad():
                    selected_idx, _, _, _ = joint_model.selector(
                        candidates, container_state, deterministic=True
                    )
                _sync_cuda()
                sel_elapsed = (time.perf_counter() - t_sel)
                selector_infer_time += sel_elapsed
                selected_idx_val = selected_idx.item()
            
            # 璁剧疆閫変腑鐨勫€欓€夊苟閲嶆柊鐢熸垚 observation
            updated_obs = wrapped_env.set_selected_and_get_obs(selected_idx_val)
            updated_pct_obs = torch.FloatTensor(updated_obs).unsqueeze(0).to(device)
            
            # PCT 閫夋嫨鏀剧疆浣嶇疆
            _sync_cuda()
            t_pct = time.perf_counter()
            with torch.no_grad():
                all_nodes, leaf_nodes = get_leaf_nodes(
                    updated_pct_obs,
                    args.internal_node_holder,
                    args.leaf_node_holder
                )
                
                if args.pure_pct:
                    placement_log_prob, placement_idx, pct_entropy, pct_value = pct_model(
                        all_nodes, deterministic=True, normFactor=args.normFactor
                    )
                else:
                    placement_log_prob, placement_idx, pct_entropy, pct_value = joint_model.pct(
                        all_nodes, deterministic=True, normFactor=args.normFactor
                    )
            _sync_cuda()
            pct_elapsed = (time.perf_counter() - t_pct)
            pct_infer_time += pct_elapsed
            results['model_infer_ms_per_step'].append((sel_elapsed + pct_elapsed) * 1000.0)
            
            # 鑾峰彇鏀剧疆鍔ㄤ綔
            pct_action = leaf_nodes[0, placement_idx.item()].cpu().numpy()
            
            # 鎵ц鍔ㄤ綔
            obs, reward, done, info = wrapped_env.step(selected_idx_val, pct_action)
            step_count += 1
            total_decisions += 1
        
        # 璁板綍缁撴灉
        ratio = info.get('ratio', 0)
        total_placed = info.get('total_placed', 0)
        final_flatness = _get_episode_flatness(wrapped_env, info)
        
        results['ratios'].append(ratio)
        results['placed'].append(total_placed)
        results['flatness_scores'].append(final_flatness)
        ep_wall_elapsed = time.perf_counter() - ep_wall_start
        if step_count > 0:
            results['e2e_ms_per_box_episode'].append(ep_wall_elapsed * 1000.0 / step_count)
        else:
            results['e2e_ms_per_box_episode'].append(0.0)
        
        if args.verbose or (ep_idx + 1) % 10 == 0:
            print(
                f"Episode {ep_idx + 1}/{num_episodes}: "
                f"Ratio = {ratio:.4f}, Placed = {total_placed}, Flatness = {final_flatness:.4f}"
            )
    
    # 缁熻
    mean_ratio = np.mean(results['ratios'])
    std_ratio = np.std(results['ratios'])
    var_ratio = np.var(results['ratios'])
    max_ratio = np.max(results['ratios'])
    min_ratio = np.min(results['ratios'])
    mean_placed = np.mean(results['placed'])
    std_placed = np.std(results['placed'])
    var_placed = np.var(results['placed'])
    mean_flatness = np.mean(results['flatness_scores'])
    std_flatness = np.std(results['flatness_scores'])
    var_flatness = np.var(results['flatness_scores'])
    max_flatness = np.max(results['flatness_scores'])
    min_flatness = np.min(results['flatness_scores'])
    eval_wall_time = time.perf_counter() - eval_wall_start
    model_infer_time = selector_infer_time + pct_infer_time
    if total_decisions > 0:
        model_ms_per_box = model_infer_time * 1000.0 / total_decisions
        e2e_ms_per_box = eval_wall_time * 1000.0 / total_decisions
    else:
        model_ms_per_box = 0.0
        e2e_ms_per_box = 0.0
    model_ms_std = np.std(results['model_infer_ms_per_step']) if len(results['model_infer_ms_per_step']) > 0 else 0.0
    model_ms_var = np.var(results['model_infer_ms_per_step']) if len(results['model_infer_ms_per_step']) > 0 else 0.0
    e2e_ms_mean_episode = np.mean(results['e2e_ms_per_box_episode']) if len(results['e2e_ms_per_box_episode']) > 0 else 0.0
    e2e_ms_std_episode = np.std(results['e2e_ms_per_box_episode']) if len(results['e2e_ms_per_box_episode']) > 0 else 0.0
    e2e_ms_var_episode = np.var(results['e2e_ms_per_box_episode']) if len(results['e2e_ms_per_box_episode']) > 0 else 0.0
    
    print("\n" + "="*50)
    print(f"EVALUATION RESULTS ({mode_str})")
    print("="*50)
    print(f"Model: {args.model_path}")
    print(f"Test Data: {args.test_dataset}")
    print(f"Episodes: {num_episodes}")
    print(f"Window Size: {args.window_size}")
    print(f"Space Utilization:")
    print(f"  Mean: {mean_ratio:.4f} ({mean_ratio*100:.2f}%)")
    print(f"  Std:  {std_ratio:.4f}")
    print(f"  Var:  {var_ratio:.6f}")
    print(f"  Max:  {max_ratio:.4f} ({max_ratio*100:.2f}%)")
    print(f"  Min:  {min_ratio:.4f} ({min_ratio*100:.2f}%)")
    print(f"Boxes Placed:")
    print(f"  Mean: {mean_placed:.1f}")
    print(f"  Std:  {std_placed:.4f}")
    print(f"  Var:  {var_placed:.4f}")
    print("Packing Flatness:")
    print(f"  Mean: {mean_flatness:.4f}")
    print(f"  Std:  {std_flatness:.4f}")
    print(f"  Var:  {var_flatness:.6f}")
    print(f"  Max:  {max_flatness:.4f}")
    print(f"  Min:  {min_flatness:.4f}")
    print("Timing:")
    print(f"  Model inference total: {model_infer_time:.4f}s")
    print(f"  Model inference mean: {model_ms_per_box:.4f} ms/box")
    print(f"  Model inference std:  {model_ms_std:.4f} ms/box")
    print(f"  Model inference var:  {model_ms_var:.6f} (ms/box)^2")
    print(f"  End-to-end total: {eval_wall_time:.4f}s")
    print(f"  End-to-end mean(global): {e2e_ms_per_box:.4f} ms/box")
    print(f"  End-to-end mean(episode): {e2e_ms_mean_episode:.4f} ms/box")
    print(f"  End-to-end std(episode):  {e2e_ms_std_episode:.4f} ms/box")
    print(f"  End-to-end var(episode):  {e2e_ms_var_episode:.6f} (ms/box)^2")
    print("="*50)
    
    return results


if __name__ == '__main__':
    registration_envs()
    args = get_eval_args()
    evaluate(args)

