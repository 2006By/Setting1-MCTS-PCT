# -*- coding: utf-8 -*-
"""
联合模型
将 Set Transformer 和 PCT 封装在一起进行端到端训练
"""
import torch
import torch.nn as nn
from model import DRL_GAT
from set_transformer import SetTransformerSelector
from tools import get_leaf_nodes


class JointSelectorPCT(nn.Module):
    """
    联合 Set Transformer 和 PCT 的模型
    
    流程:
    1. Set Transformer 从5个候选包裹中选择1个
    2. PCT 选择放置位置
    
    两个模块一起训练，共享梯度
    """
    
    def __init__(self, args, window_size=5, container_feature_dim=4):
        """
        Args:
            args: PCT 参数
            window_size: 滑动窗口大小
            container_feature_dim: 容器状态特征维度
        """
        super(JointSelectorPCT, self).__init__()
        
        self.args = args
        self.window_size = window_size
        self.normFactor = args.normFactor
        
        # Set Transformer 选择器
        self.selector = SetTransformerSelector(
            box_dim=3,
            container_feature_dim=container_feature_dim,
            hidden_dim=64,
            num_heads=4,
            num_inds=4,
            num_candidates=window_size,
            normFactor=args.normFactor
        )
        
        # PCT 模型 (保持原有结构)
        self.pct = DRL_GAT(args)
        
        # 保存参数
        self.internal_node_holder = args.internal_node_holder
        self.leaf_node_holder = args.leaf_node_holder
        
    def forward(self, candidates, container_state, pct_obs, deterministic=False):
        """
        联合前向传播
        
        Args:
            candidates: 候选包裹 [batch_size, window_size, 3] (已归一化)
            container_state: 容器状态 [batch_size, container_feature_dim]
            pct_obs: PCT 观察状态 [batch_size, obs_dim]
            deterministic: 是否使用确定性策略
        
        Returns:
            dict: 包含所有输出的字典
        """
        batch_size = candidates.size(0)
        
        # 1. Set Transformer 选择包裹
        selected_idx, selector_log_prob, selector_entropy, selector_value = self.selector(
            candidates, container_state, deterministic=deterministic
        )
        
        # 2. 解析 PCT 观察
        # pct_obs 中的 next_item 已由环境的 set_selected_and_get_obs() 正确设置
        # 无需再从 candidates 重新计算覆盖
        all_nodes, leaf_nodes = get_leaf_nodes(
            pct_obs, 
            self.internal_node_holder, 
            self.leaf_node_holder
        )
        
        # 3. PCT 选择放置位置（直接使用环境设置好的 all_nodes）
        placement_log_prob, placement_idx, pct_entropy, pct_value = self.pct(
            all_nodes, 
            deterministic=deterministic, 
            normFactor=self.normFactor
        )
        
        return {
            'selected_idx': selected_idx,
            'selector_log_prob': selector_log_prob,
            'selector_entropy': selector_entropy,
            'selector_value': selector_value,
            'placement_idx': placement_idx,
            'placement_log_prob': placement_log_prob,
            'pct_entropy': pct_entropy,
            'pct_value': pct_value,
            'leaf_nodes': leaf_nodes,  # 返回原始 leaf_nodes
            'total_entropy': selector_entropy + pct_entropy
        }
    
    def evaluate_actions(self, candidates, container_state, pct_obs, 
                         selector_actions, placement_actions):
        """
        评估给定动作
        
        Args:
            candidates: 候选包裹 [batch_size, window_size, 3]
            container_state: 容器状态 [batch_size, container_feature_dim]
            pct_obs: PCT 观察状态 [batch_size, obs_dim]
            selector_actions: Set Transformer 的选择 [batch_size]
            placement_actions: PCT 的放置选择 [batch_size]
        
        Returns:
            dict: 包含 log_probs, entropy, values
        """
        batch_size = candidates.size(0)
        
        # 评估 Set Transformer 动作
        selector_log_prob, selector_entropy, selector_value = self.selector.evaluate_actions(
            candidates, container_state, selector_actions
        )
        
        # 评估 PCT 动作 - 需要根据 selector_actions 更新 next_item
        all_nodes, leaf_nodes = get_leaf_nodes(
            pct_obs, 
            self.internal_node_holder, 
            self.leaf_node_holder
        )
        
        # 获取选中的候选包裹（原始尺寸）
        selector_actions_expanded = selector_actions.view(batch_size, 1, 1).expand(-1, -1, 3)
        selected_box = candidates.gather(1, selector_actions_expanded).squeeze(1)
        selected_box_sorted, _ = torch.sort(selected_box, dim=-1)
        
        # 不要额外乘 normFactor！all_nodes 是原始尺寸，PCT 内部 forward 时统一乘 normFactor
        # 只更新 all_nodes 中的 next_item，不修改 leaf_nodes
        next_item_idx = self.internal_node_holder + self.leaf_node_holder
        all_nodes_updated = all_nodes.clone()
        all_nodes_updated[:, next_item_idx, 3:6] = selected_box_sorted
        
        pct_value, placement_log_prob, pct_entropy = self.pct.evaluate_actions(
            all_nodes_updated, 
            placement_actions, 
            normFactor=self.normFactor
        )
        
        return {
            'selector_log_prob': selector_log_prob,
            'selector_entropy': selector_entropy,
            'selector_value': selector_value,
            'placement_log_prob': placement_log_prob,
            'pct_entropy': pct_entropy,
            'pct_value': pct_value,
            'total_entropy': selector_entropy + pct_entropy
        }
    
    def get_pct_action(self, pct_obs, deterministic=False):
        """
        仅获取 PCT 的动作 (用于调试)
        """
        all_nodes, leaf_nodes = get_leaf_nodes(
            pct_obs, 
            self.internal_node_holder, 
            self.leaf_node_holder
        )
        
        placement_log_prob, placement_idx, pct_entropy, pct_value = self.pct(
            all_nodes, 
            deterministic=deterministic, 
            normFactor=self.normFactor
        )
        
        return placement_idx, leaf_nodes, pct_value


class JointRolloutStorage:
    """
    联合训练的经验存储
    
    存储 Set Transformer 和 PCT 的动作及其 log probabilities
    """
    
    def __init__(self, num_steps, num_processes, pct_obs_shape, 
                 window_size=5, container_feature_dim=4, gamma=1.0):
        """
        Args:
            num_steps: n-step 训练的步数
            num_processes: 并行进程数
            pct_obs_shape: PCT 观察维度
            window_size: 滑动窗口大小
            container_feature_dim: 容器状态特征维度
            gamma: 折扣因子
        """
        self.num_steps = num_steps
        self.num_processes = num_processes
        self.gamma = gamma
        
        # PCT 观察 (只存 selected_pct_obs，和 action 一一对应，无+1延时)
        self.pct_obs = torch.zeros(num_steps, num_processes, *pct_obs_shape)
        
        # 候选包裹
        self.candidates = torch.zeros(num_steps + 1, num_processes, window_size, 3)
        
        # 容器状态
        self.container_states = torch.zeros(num_steps + 1, num_processes, container_feature_dim)
        
        # Set Transformer 动作和 log probs
        self.selector_actions = torch.zeros(num_steps, num_processes, 1).long()
        self.selector_log_probs = torch.zeros(num_steps, num_processes, 1)
        
        # PCT 动作和 log probs
        self.placement_actions = torch.zeros(num_steps, num_processes, 1).long()
        self.placement_log_probs = torch.zeros(num_steps, num_processes, 1)
        
        # 奖励和 returns
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        
        # Masks (是否继续)
        self.masks = torch.ones(num_steps + 1, num_processes, 1)
        
        # 步数计数
        self.step = 0
        
    def to(self, device):
        """移动到指定设备"""
        self.pct_obs = self.pct_obs.to(device)
        self.candidates = self.candidates.to(device)
        self.container_states = self.container_states.to(device)
        self.selector_actions = self.selector_actions.to(device)
        self.selector_log_probs = self.selector_log_probs.to(device)
        self.placement_actions = self.placement_actions.to(device)
        self.placement_log_probs = self.placement_log_probs.to(device)
        self.rewards = self.rewards.to(device)
        self.returns = self.returns.to(device)
        self.masks = self.masks.to(device)
        return self
        
    def insert(self, pct_obs, candidates, container_state,
               selector_action, selector_log_prob,
               placement_action, placement_log_prob,
               reward, mask):
        """
        插入一步经验
        """
        self.pct_obs[self.step].copy_(pct_obs)
        self.candidates[self.step + 1].copy_(candidates)
        self.container_states[self.step + 1].copy_(container_state)
        
        self.selector_actions[self.step].copy_(selector_action)
        self.selector_log_probs[self.step].copy_(selector_log_prob)
        
        self.placement_actions[self.step].copy_(placement_action)
        self.placement_log_probs[self.step].copy_(placement_log_prob)
        
        self.rewards[self.step].copy_(reward)
        self.masks[self.step + 1].copy_(mask)
        
        self.step = (self.step + 1) % self.num_steps
        
    def after_update(self):
        """更新后复制最后一个状态到开头"""
        self.candidates[0].copy_(self.candidates[-1])
        self.container_states[0].copy_(self.container_states[-1])
        self.masks[0].copy_(self.masks[-1])
        
    def compute_returns(self, next_value):
        """计算 n-step returns"""
        self.returns[-1] = next_value
        for step in reversed(range(self.num_steps)):
            self.returns[step] = (
                self.returns[step + 1] * self.gamma * self.masks[step + 1] 
                + self.rewards[step]
            )


# 测试代码
if __name__ == '__main__':
    import argparse
    
    # 模拟 args
    class Args:
        embedding_size = 64
        hidden_size = 128
        gat_layer_num = 1
        internal_node_holder = 80
        internal_node_length = 6
        leaf_node_holder = 50
        normFactor = 0.0125
    
    args = Args()
    
    # 创建联合模型
    model = JointSelectorPCT(args, window_size=5)
    
    print("JointSelectorPCT created successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"  - Selector parameters: {sum(p.numel() for p in model.selector.parameters())}")
    print(f"  - PCT parameters: {sum(p.numel() for p in model.pct.parameters())}")
