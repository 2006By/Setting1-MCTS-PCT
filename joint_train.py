# -*- coding: utf-8 -*-
"""
联合训练脚本
Set Transformer + PCT 端到端训练
"""
import sys
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import argparse

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from torch.utils.tensorboard import SummaryWriter
from tools import registration_envs, backup, get_leaf_nodes
from joint_model import JointSelectorPCT, JointRolloutStorage
from sliding_window_env import SlidingWindowEnvWrapper, normalize_to_trajectories
from vec_sliding_window_env import VecSlidingWindowEnv
from kfac import KFACOptimizer
import gym
import givenData


def get_joint_args():
    """获取联合训练参数"""
    parser = argparse.ArgumentParser(description='Joint Set Transformer + PCT Training')
    
    # 基础参数
    parser.add_argument('--setting', type=int, default=1, help='Experiment setting (1, 2, or 3)')
    parser.add_argument('--continuous', action='store_true', help='Use continuous environment')
    parser.add_argument('--no-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--device', type=int, default=0, help='GPU device ID')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # 数据集参数
    parser.add_argument('--train-dataset', type=str, default='datasets/train.pt', help='Training dataset path (flat item list)')
    
    # 滑动窗口参数
    parser.add_argument('--window-size', type=int, default=5, help='Candidate window size')
    parser.add_argument('--flatness-reward-coef', type=float, default=0.2,
                        help='Long-term flatness reward coefficient (set 0 to disable)')
    parser.add_argument('--flatness-interval', type=int, default=10,
                        help='Trigger flatness reward every N placed boxes')
    parser.add_argument('--flatness-min-placed', type=int, default=10,
                        help='Minimum placed boxes before flatness reward can be triggered')
    parser.add_argument('--flatness-grid-size', type=int, default=16,
                        help='Grid resolution for flatness surface estimation')
    
    # 训练参数
    parser.add_argument('--num-processes', type=int, default=16, help='Number of parallel environments')
    parser.add_argument('--num-steps', type=int, default=20, help='N-step training (increased for longer episodes)')
    parser.add_argument('--learning-rate', type=float, default=0.25, help='PCT KFAC base learning rate')
    parser.add_argument('--selector-learning-rate', type=float, default=0.25, help='Selector KFAC base learning rate')
    parser.add_argument('--gamma', type=float, default=1.0, help='Discount factor')
    parser.add_argument('--max-grad-norm', type=float, default=0.5, help='Max gradient norm')
    
    # 损失系数
    parser.add_argument('--actor-loss-coef', type=float, default=1.0, help='Actor loss coefficient')
    parser.add_argument('--critic-loss-coef', type=float, default=0.5, help='Critic loss coefficient')
    parser.add_argument('--entropy-coef', type=float, default=0.001, help='Entropy coefficient')
    
    # 模型参数
    parser.add_argument('--embedding-size', type=int, default=64, help='Embedding dimension')
    parser.add_argument('--hidden-size', type=int, default=128, help='Hidden layer dimension')
    parser.add_argument('--gat-layer-num', type=int, default=1, help='Number of GAT layers')
    parser.add_argument('--internal-node-holder', type=int, default=100, help='Max internal nodes')
    parser.add_argument('--leaf-node-holder', type=int, default=50, help='Max leaf nodes')
    
    # 日志参数
    parser.add_argument('--log-interval', type=int, default=10, help='Log interval')
    parser.add_argument('--save-interval', type=int, default=100, help='Model save interval')
    parser.add_argument('--latest-save-interval', type=int, default=10, help='Latest checkpoint save interval')
    parser.add_argument('--max-updates', type=int, default=10000, help='Maximum training updates')
    parser.add_argument('--run-name', type=str, default='', help='Run name for log/model directories')
    parser.add_argument('--resume-checkpoint', type=str, default='', help='Resume from a specific checkpoint path')
    parser.add_argument('--no-auto-resume', action='store_true', help='Disable auto resume from latest checkpoint')
    
    # 额外参数
    parser.add_argument('--lnes', type=str, default='EMS', help='Leaf node expansion scheme')
    parser.add_argument('--shuffle', type=bool, default=True, help='Shuffle leaf nodes')
    
    args = parser.parse_args()
    
    # 【强制设为要求的参数】
    args.setting = 1
    args.internal_node_holder = 100
    args.leaf_node_holder = 50
    args.continuous = True
    
    # 设置设备
    if args.no_cuda:
        args.device = 'cpu'
    
    # 容器和物品设置
    args.container_size = givenData.container_size
    args.item_size_set = givenData.item_size_set
    
    # 环境 ID
    if args.continuous:
        args.id = 'PctContinuous-v0'
    else:
        args.id = 'PctDiscrete-v0'
    
    # 节点长度
    if args.setting == 1:
        args.internal_node_length = 6
    elif args.setting == 2:
        args.internal_node_length = 6
    elif args.setting == 3:
        args.internal_node_length = 7
    
    # 归一化因子
    args.normFactor = 1.0 / np.max(args.container_size)
    
    return args


def create_env(args, seed):
    """创建单个 PCT 环境"""
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
        sample_from_distribution=False,
        sample_left_bound=None,
        sample_right_bound=None
    )
    env.seed(seed)
    return env


def _latest_pointer_file():
    return os.path.join('./logs/joint_models', 'latest_checkpoint.txt')


def _resolve_resume_path(args):
    if args.resume_checkpoint:
        return args.resume_checkpoint if os.path.exists(args.resume_checkpoint) else None
    if args.no_auto_resume:
        return None
    pointer = _latest_pointer_file()
    if not os.path.exists(pointer):
        return None
    try:
        with open(pointer, 'r', encoding='utf-8') as f:
            ckpt_path = f.read().strip()
        if ckpt_path and os.path.exists(ckpt_path):
            return ckpt_path
    except OSError:
        return None
    return None


def _write_latest_pointer(ckpt_path):
    os.makedirs('./logs/joint_models', exist_ok=True)
    with open(_latest_pointer_file(), 'w', encoding='utf-8') as f:
        f.write(ckpt_path)


def _save_training_checkpoint(path, model, selector_optimizer, pct_optimizer,
                              update, best_mean_ratio, args, log_dir, model_dir, time_str,
                              update_latest_pointer=True):
    ckpt = {
        'model_state_dict': model.state_dict(),
        'selector_optimizer_state_dict': selector_optimizer.state_dict(),
        'pct_optimizer_state_dict': pct_optimizer.state_dict(),
        'update': update,
        'best_mean_ratio': best_mean_ratio,
        'args': args,
        'log_dir': log_dir,
        'model_dir': model_dir,
        'time_str': time_str,
        'torch_rng_state': torch.get_rng_state(),
        'numpy_rng_state': np.random.get_state(),
        'python_rng_state': random.getstate(),
    }
    if torch.cuda.is_available():
        ckpt['cuda_rng_state_all'] = torch.cuda.get_rng_state_all()
    torch.save(ckpt, path)
    if update_latest_pointer:
        _write_latest_pointer(path)


def train(args):
    """主训练函数"""
    # 设置随机种子
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # 设置设备
    if isinstance(args.device, int):
        device = torch.device('cuda', args.device) if torch.cuda.is_available() else torch.device('cpu')
        if torch.cuda.is_available():
            torch.cuda.set_device(args.device)
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # 创建实验名称和日志
    resume_path = _resolve_resume_path(args)
    resume_state = None
    if resume_path is not None:
        print(f"Auto-resume checkpoint found: {resume_path}")
        resume_state = torch.load(resume_path, map_location='cpu')
        timeStr = resume_state.get('time_str', '')
        if not timeStr:
            timeStr = os.path.basename(os.path.dirname(resume_path))
        log_dir = resume_state.get('log_dir', f'./logs/joint_runs/{timeStr}')
        model_dir = resume_state.get('model_dir', f'./logs/joint_models/{timeStr}')
    else:
        run_name = args.run_name.strip()
        if run_name:
            timeStr = run_name
        else:
            timeStr = f"joint-{time.strftime('%Y.%m.%d-%H-%M-%S', time.localtime())}"
        log_dir = f'./logs/joint_runs/{timeStr}'
        model_dir = f'./logs/joint_models/{timeStr}'

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    print(f"Run name: {timeStr}")
    print(f"Log dir: {log_dir}")
    print(f"Model dir: {model_dir}")

    writer = SummaryWriter(log_dir=log_dir)
    
    # 加载数据集并转换为轨迹格式（与 PCT 的 LoadBoxCreator 一致）
    print(f"Loading dataset from {args.train_dataset}...")
    raw_data = torch.load(args.train_dataset)
    trajectories = normalize_to_trajectories(raw_data)
    total_items = sum(len(t) for t in trajectories)
    print(f"Loaded {len(trajectories)} trajectories, {total_items} total items")
    
    # 创建并行化多进程环境（每个环境从不同的轨迹开始）
    print("Creating explicit multiprocess environments...")
    def make_env_fn(i, start_idx):
        def _thunk():
            env = create_env(args, args.seed + i)
            return SlidingWindowEnvWrapper(
                env, 
                trajectories,
                window_size=args.window_size,
                normFactor=args.normFactor,
                traj_start_idx=start_idx,
                flatness_reward_coef=args.flatness_reward_coef,
                flatness_interval=args.flatness_interval,
                flatness_min_placed=args.flatness_min_placed,
                flatness_grid_size=args.flatness_grid_size
            )
        return _thunk

    env_fns = []
    for i in range(args.num_processes):
        traj_start = (i * len(trajectories)) // args.num_processes
        env_fns.append(make_env_fn(i, traj_start))
    
    vec_envs = VecSlidingWindowEnv(env_fns)
    print(f"Created {args.num_processes} parallel wrapped environments")
    
    # 创建联合模型
    print("Creating joint model...")
    model = JointSelectorPCT(args, window_size=args.window_size)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    selector_params = sum(p.numel() for p in model.selector.parameters())
    pct_params = sum(p.numel() for p in model.pct.parameters())
    
    print(f"Total parameters: {total_params:,}")
    print(f"  - Selector: {selector_params:,}")
    print(f"  - PCT: {pct_params:,}")
    
    # 创建分离的优化器（selector 和 PCT 各自独立）
    selector_optimizer = KFACOptimizer(model.selector, lr=args.selector_learning_rate)
    pct_optimizer = KFACOptimizer(model.pct, lr=args.learning_rate)

    start_update = 1
    best_mean_ratio = -float('inf')
    if resume_state is not None:
        model.load_state_dict(resume_state['model_state_dict'])
        if 'selector_optimizer_state_dict' in resume_state:
            selector_optimizer.load_state_dict(resume_state['selector_optimizer_state_dict'])
        if 'pct_optimizer_state_dict' in resume_state:
            pct_optimizer.load_state_dict(resume_state['pct_optimizer_state_dict'])
        start_update = int(resume_state.get('update', 0)) + 1
        best_mean_ratio = float(resume_state.get('best_mean_ratio', -float('inf')))

        if 'torch_rng_state' in resume_state:
            torch.set_rng_state(resume_state['torch_rng_state'])
        if 'numpy_rng_state' in resume_state:
            np.random.set_state(resume_state['numpy_rng_state'])
        if 'python_rng_state' in resume_state:
            random.setstate(resume_state['python_rng_state'])
        if torch.cuda.is_available() and 'cuda_rng_state_all' in resume_state:
            torch.cuda.set_rng_state_all(resume_state['cuda_rng_state_all'])

        print(f"Resumed from update {start_update - 1} | best mean ratio {best_mean_ratio:.4f}")
    
    # 创建 rollout storage
    obs_shape = (args.internal_node_holder + args.leaf_node_holder + 1, 9)
    storage = JointRolloutStorage(
        num_steps=args.num_steps,
        num_processes=args.num_processes,
        pct_obs_shape=obs_shape,
        window_size=args.window_size,
        container_feature_dim=4,
        gamma=args.gamma
    ).to(device)
    
    # 初始化环境状态
    print("Initializing environments...")
    pct_obs_np, candidates_np, container_states_np = vec_envs.reset()
    
    pct_obs = torch.FloatTensor(pct_obs_np).to(device)
    candidates = torch.FloatTensor(candidates_np).to(device)
    container_states = torch.FloatTensor(container_states_np).to(device)
    
    # 存储初始状态 (pct_obs 由 env step 时录入)
    storage.candidates[0].copy_(candidates)
    storage.container_states[0].copy_(container_states)
    
    # 训练统计
    episode_rewards = deque(maxlen=100)
    episode_ratios = deque(maxlen=100)
    episode_placed = deque(maxlen=100)
    flatness_reward_window = deque(maxlen=2000)
    flatness_score_window = deque(maxlen=2000)
    
    print("\nStarting training...")
    start_time = time.time()
    
    batchX = torch.arange(args.num_processes).to(device)

    if start_update > args.max_updates:
        print(f"Checkpoint update {start_update - 1} already >= max_updates {args.max_updates}, nothing to train.")
        writer.close()
        vec_envs.close()
        return

    for update in range(start_update, args.max_updates + 1):
        model.train()
        
        # 收集 n-step 经验
        for step in range(args.num_steps):
            # Step 1: Set Transformer 选择候选包裹
            with torch.no_grad():
                current_candidates = storage.candidates[step]
                current_container_states = storage.container_states[step]
                
                # 只运行 Set Transformer 选择候选
                selected_idx, selector_log_prob, selector_entropy, selector_value = model.selector(
                    current_candidates,
                    current_container_states,
                    deterministic=False
                )
            
            # Step 2: 设置选中的候选并重新生成 observation（多进程并行）
            selected_idx_np = selected_idx.cpu().numpy().flatten()
            updated_obs = vec_envs.set_selected_and_get_obs(selected_idx_np)
            
            # 转换为 tensor
            updated_pct_obs = torch.FloatTensor(updated_obs).to(device)
            
            # 【修复】记录 selector 选完后的 pct_obs，用于 PPO re-evaluation
            # 这样 storage 中的 pct_obs 与 selector 的选择一一对应
            selected_pct_obs = updated_pct_obs.clone()
            
            # Step 3: PCT 从新的 observation 中选择放置位置
            with torch.no_grad():
                all_nodes, leaf_nodes = get_leaf_nodes(
                    updated_pct_obs, 
                    args.internal_node_holder, 
                    args.leaf_node_holder
                )
                
                placement_log_prob, placement_idx, pct_entropy, pct_value = model.pct(
                    all_nodes,
                    deterministic=False,
                    normFactor=args.normFactor
                )
            
            # 获取选中的 leaf_node
            selected_leaf = leaf_nodes[batchX, placement_idx.squeeze()].cpu().numpy()
            
            # 执行环境步骤（多进程并行）
            obs_list, rewards, dones, infos, new_cands, new_cstates = vec_envs.step(
                selected_idx_np, selected_leaf
            )
            for info in infos:
                if isinstance(info, dict):
                    if 'flatness_reward' in info:
                        flatness_reward_window.append(float(info.get('flatness_reward', 0.0)))
                    if 'flatness_score' in info:
                        flatness_score_window.append(float(info.get('flatness_score', 0.0)))
            
            for i, (done, info) in enumerate(zip(dones, infos)):
                if done:
                    # 记录 episode 统计
                    ratio = info.get('ratio', 0)
                    placed = info.get('total_placed', 0)
                    episode_ratios.append(ratio)
                    episode_placed.append(placed)
                    episode_rewards.append(ratio * 10)
                    
                    # 输出每个 episode 的结果
                    print(f"  [Env {i}] Episode done: ratio={ratio:.4f}, placed={placed}")
            
            # 更新状态
            new_candidates = torch.FloatTensor(new_cands).to(device)
            new_container_states = torch.FloatTensor(new_cstates).to(device)
            
            # 存储经验
            # 【修复】使用 selected_pct_obs 而非 env.step 后的 obs
            # selected_pct_obs 包含了 selector 实际选择的 next_item 和对应的 leaf_nodes
            storage.insert(
                pct_obs=selected_pct_obs.view(args.num_processes, *obs_shape),
                candidates=new_candidates,
                container_state=new_container_states,
                selector_action=selected_idx.view(-1, 1),
                selector_log_prob=selector_log_prob.view(-1, 1),
                placement_action=placement_idx.view(-1, 1),
                placement_log_prob=placement_log_prob.view(-1, 1),
                reward=torch.FloatTensor(rewards).unsqueeze(-1).to(device),
                mask=torch.FloatTensor(1 - np.array(dones)).unsqueeze(-1).to(device)
            )
        
        # 计算 next value（由于 state 到了 T 时刻还没做 Selector 选择，使用 Selector 预测作为 V_T）
        with torch.no_grad():
            final_candidates = storage.candidates[-1]
            final_container_states = storage.container_states[-1]
            
            _, _, _, next_value = model.selector(
                final_candidates,
                final_container_states,
                deterministic=False
            )
        
        # 计算 returns
        storage.compute_returns(next_value)
        
        # 策略优化
        # 重新计算所有 log probs 和价值
        all_pct_obs = storage.pct_obs.view(-1, *obs_shape)
        all_candidates = storage.candidates[:-1].view(-1, args.window_size, 3)
        all_container_states = storage.container_states[:-1].view(-1, 4)
        all_selector_actions = storage.selector_actions.view(-1)
        all_placement_actions = storage.placement_actions.view(-1, 1)
        
        eval_outputs = model.evaluate_actions(
            all_candidates,
            all_container_states,
            all_pct_obs.view(-1, all_pct_obs.shape[-2] * all_pct_obs.shape[-1]),
            all_selector_actions,
            all_placement_actions
        )
        
        # ============ 分离 advantage ============
        selector_values = eval_outputs['selector_value'].view(args.num_steps, args.num_processes, 1)
        pct_values = eval_outputs['pct_value'].view(args.num_steps, args.num_processes, 1)
        
        selector_advantages = storage.returns[:-1] - selector_values
        pct_advantages = storage.returns[:-1] - pct_values
        
        # 【关键修复】Advantage 归一化
        # 原因：raw advantages 初始时很小（critic 快速学会预测 returns → advantages → 0），
        # 导致 actor 梯度 = advantage * d(log_prob)/d(params) ≈ 0，selector 完全学不动。
        # 归一化后，梯度 scale 保持一致，不再受 critic 收敛速度影响。
        selector_adv_normalized = selector_advantages.detach()
        if selector_adv_normalized.numel() > 1:
            adv_mean = selector_adv_normalized.mean()
            adv_std = selector_adv_normalized.std()
            selector_adv_normalized = (selector_adv_normalized - adv_mean) / (adv_std + 1e-8)
        
        pct_adv_normalized = pct_advantages.detach()
        if pct_adv_normalized.numel() > 1:
            adv_mean = pct_adv_normalized.mean()
            adv_std = pct_adv_normalized.std()
            pct_adv_normalized = (pct_adv_normalized - adv_mean) / (adv_std + 1e-8)
        
        # 分离 log probs
        selector_log_probs = eval_outputs['selector_log_prob'].view(args.num_steps, args.num_processes, 1)
        placement_log_probs = eval_outputs['placement_log_prob'].view(args.num_steps, args.num_processes, 1)
        
        # 分离 actor loss（使用归一化后的 advantage）
        selector_actor_loss = -(selector_adv_normalized * selector_log_probs).mean()
        pct_actor_loss = -(pct_adv_normalized * placement_log_probs).mean()
        actor_loss = selector_actor_loss + pct_actor_loss
        
        # 分离 critic loss
        selector_critic_loss = selector_advantages.pow(2).mean()
        pct_critic_loss = pct_advantages.pow(2).mean()
        critic_loss = (selector_critic_loss + pct_critic_loss) / 2
        
        # 分离 entropy
        selector_entropy = eval_outputs['selector_entropy'].mean()
        pct_entropy = eval_outputs['pct_entropy'].mean()
        total_entropy = selector_entropy + pct_entropy
        
        # 【关键修复】不对 selector 使用 entropy bonus
        # 原因：uniform 分布是 categorical entropy 的最大值点，
        # d(entropy)/d(scores) 在 uniform 处恒为 0，加了也没有梯度效果。
        # 而且 entropy bonus 的方向是鼓励 uniform（最大化熵），与我们想让 selector 学会区分的目标矛盾。
        total_loss = (
            args.actor_loss_coef * actor_loss +
            args.critic_loss_coef * critic_loss -
            args.entropy_coef * (pct_entropy + 0.5 * selector_entropy)  # selector 也需要熔奖励探索
        )
        
        # --- KFAC (ACKTR) Fisher Loss Backup for Selector ---
        if selector_optimizer.steps % selector_optimizer.Ts == 0:
            selector_optimizer.zero_grad()
            pg_fisher_loss_sel = - selector_log_probs.mean()
            value_noise_sel = torch.randn(selector_values.size(), device=device)
            sample_values_sel = selector_values + value_noise_sel
            vf_fisher_loss_sel = -(selector_values - sample_values_sel.detach()).pow(2).mean()
            fisher_loss_sel = pg_fisher_loss_sel + vf_fisher_loss_sel
            selector_optimizer.acc_stats = True
            fisher_loss_sel.backward(retain_graph=True)
            selector_optimizer.acc_stats = False

        # --- KFAC (ACKTR) Fisher Loss Backup for PCT ---
        if pct_optimizer.steps % pct_optimizer.Ts == 0:
            pct_optimizer.zero_grad()
            pg_fisher_loss_pct = - placement_log_probs.mean()
            value_noise_pct = torch.randn(pct_values.size(), device=device)
            sample_values_pct = pct_values + value_noise_pct
            vf_fisher_loss_pct = -(pct_values - sample_values_pct.detach()).pow(2).mean()
            fisher_loss_pct = pg_fisher_loss_pct + vf_fisher_loss_pct
            pct_optimizer.acc_stats = True
            fisher_loss_pct.backward(retain_graph=True)
            pct_optimizer.acc_stats = False

        # 反向传播 (Total Loss)
        selector_optimizer.zero_grad()
        pct_optimizer.zero_grad()
        total_loss.backward()
        
        # 分别裁剪梯度
        selector_grad_norm = nn.utils.clip_grad_norm_(model.selector.parameters(), args.max_grad_norm)
        pct_grad_norm = nn.utils.clip_grad_norm_(model.pct.parameters(), args.max_grad_norm)
        
        selector_optimizer.step()
        pct_optimizer.step()
        
        # 更新 storage
        storage.after_update()

        if update % args.latest_save_interval == 0:
            latest_ckpt_path = os.path.join(model_dir, 'checkpoint_latest.pt')
            _save_training_checkpoint(
                latest_ckpt_path, model, selector_optimizer, pct_optimizer,
                update, best_mean_ratio, args, log_dir, model_dir, timeStr,
                update_latest_pointer=True
            )
        
        # 日志记录
        if update % args.log_interval == 0 and len(episode_ratios) > 0:
            elapsed = time.time() - start_time
            fps = update * args.num_steps * args.num_processes / elapsed
            
            mean_ratio = np.mean(episode_ratios)
            max_ratio = np.max(episode_ratios)
            mean_placed = np.mean(episode_placed)

            if mean_ratio > best_mean_ratio:
                best_mean_ratio = mean_ratio
                best_model_path = os.path.join(model_dir, 'best_model.pt')
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'selector_optimizer_state_dict': selector_optimizer.state_dict(),
                    'pct_optimizer_state_dict': pct_optimizer.state_dict(),
                    'update': update,
                    'best_mean_ratio': best_mean_ratio,
                    'args': args
                }, best_model_path)
                best_ckpt_path = os.path.join(model_dir, 'best_checkpoint.pt')
                _save_training_checkpoint(
                    best_ckpt_path, model, selector_optimizer, pct_optimizer,
                    update, best_mean_ratio, args, log_dir, model_dir, timeStr,
                    update_latest_pointer=False
                )
                print(f"  New best mean ratio: {best_mean_ratio:.4f} | saved to {best_model_path}")
            
            print(f"Update {update} | FPS {fps:.0f}")
            print(f"  Mean ratio: {mean_ratio:.4f} | Max ratio: {max_ratio:.4f}")
            print(f"  Mean placed: {mean_placed:.1f}")
            print(f"  Actor loss: {actor_loss.item():.4f} | Critic loss: {critic_loss.item():.4f}")
            print(f"  Selector entropy: {selector_entropy.item():.4f} | PCT entropy: {pct_entropy.item():.4f} | Total entropy: {total_entropy.item():.4f}")
            print(f"  Selector grad norm: {selector_grad_norm:.4f} | PCT grad norm: {pct_grad_norm:.4f}")
            if len(flatness_reward_window) > 0:
                print(f"  Mean flatness reward: {np.mean(flatness_reward_window):.5f}")
            if len(flatness_score_window) > 0:
                print(f"  Mean flatness score: {np.mean(flatness_score_window):.5f}")
            print()
            
            # TensorBoard 日志
            writer.add_scalar('Ratio/Mean', mean_ratio, update)
            writer.add_scalar('Ratio/Max', max_ratio, update)
            writer.add_scalar('Placed/Mean', mean_placed, update)
            writer.add_scalar('Loss/Actor', actor_loss.item(), update)
            writer.add_scalar('Loss/Selector_Actor', selector_actor_loss.item(), update)
            writer.add_scalar('Loss/PCT_Actor', pct_actor_loss.item(), update)
            writer.add_scalar('Loss/Critic', critic_loss.item(), update)
            writer.add_scalar('Loss/Total', total_loss.item(), update)
            writer.add_scalar('Training/Selector_Entropy', selector_entropy.item(), update)
            writer.add_scalar('Training/PCT_Entropy', pct_entropy.item(), update)
            writer.add_scalar('Training/Total_Entropy', total_entropy.item(), update)
            writer.add_scalar('Training/Selector_Grad_Norm', selector_grad_norm, update)
            writer.add_scalar('Training/PCT_Grad_Norm', pct_grad_norm, update)
            writer.add_scalar('Training/FPS', fps, update)
            if len(flatness_reward_window) > 0:
                writer.add_scalar('Reward/Flatness_Mean', np.mean(flatness_reward_window), update)
            if len(flatness_score_window) > 0:
                writer.add_scalar('Flatness/Score_Mean', np.mean(flatness_score_window), update)
        
        # 保存模型
        if update % args.save_interval == 0:
            save_path = os.path.join(model_dir, f'joint_model_{update}.pt')
            torch.save({
                'model_state_dict': model.state_dict(),
                'selector_optimizer_state_dict': selector_optimizer.state_dict(),
                'pct_optimizer_state_dict': pct_optimizer.state_dict(),
                'update': update,
                'best_mean_ratio': best_mean_ratio,
                'args': args
            }, save_path)
            ckpt_path = os.path.join(model_dir, f'joint_checkpoint_{update}.pt')
            _save_training_checkpoint(
                ckpt_path, model, selector_optimizer, pct_optimizer,
                update, best_mean_ratio, args, log_dir, model_dir, timeStr,
                update_latest_pointer=True
            )
            print(f"Model saved to {save_path}")
    
    # 最终保存
    final_path = os.path.join(model_dir, 'joint_model_final.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'selector_optimizer_state_dict': selector_optimizer.state_dict(),
        'pct_optimizer_state_dict': pct_optimizer.state_dict(),
        'update': args.max_updates,
        'best_mean_ratio': best_mean_ratio,
        'args': args
    }, final_path)
    final_ckpt_path = os.path.join(model_dir, 'checkpoint_latest.pt')
    _save_training_checkpoint(
        final_ckpt_path, model, selector_optimizer, pct_optimizer,
        args.max_updates, best_mean_ratio, args, log_dir, model_dir, timeStr,
        update_latest_pointer=True
    )
    print(f"Final model saved to {final_path}")
    
    writer.close()
    vec_envs.close()
    print("Training completed!")


if __name__ == '__main__':
    # 注册环境
    registration_envs()
    
    # 获取参数
    args = get_joint_args()
    
    # 开始训练
    train(args)
