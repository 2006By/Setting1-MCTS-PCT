# -*- coding: utf-8 -*-
"""
Set Transformer 模块
用于从候选包裹中选择最优包裹

参考论文: "Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self, dim_Q, dim_K, dim_V, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q_proj = self.fc_q(Q)
        K_proj = self.fc_k(K)
        V = self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q_proj.split(dim_split, dim=2), dim=0)
        K_ = torch.cat(K_proj.split(dim_split, dim=2), dim=0)
        V_ = torch.cat(V.split(dim_split, dim=2), dim=0)

        A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(dim_split), dim=2)
        O = torch.cat(A.bmm(V_).split(Q.size(0), dim=0), dim=2)
        O = Q_proj + F.relu(self.fc_o(O))  # 残差连接: Q + Attention(Q, K, V)
        return O


class SAB(nn.Module):
    """Self-Attention Block"""
    def __init__(self, dim_in, dim_out, num_heads):
        super(SAB, self).__init__()
        self.mab = MultiHeadAttention(dim_in, dim_in, dim_out, num_heads)

    def forward(self, X):
        return self.mab(X, X)


class PMA(nn.Module):
    """Pooling by Multihead Attention"""
    def __init__(self, dim, num_heads, num_seeds):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MultiHeadAttention(dim, dim, dim, num_heads)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)


class SetTransformerSelector(nn.Module):
    """
    Set Transformer 包裹选择器

    从5个候选包裹中选择1个最优包裹

    架构设计:
    - 包裹尺寸通过 Set Transformer 编码（只接收 box_dim=3，不混入 container_state）
    - 容器状态独立编码
    - 评分阶段融合三者: 候选编码 + 全局编码 + 容器编码
    - 不使用 LayerNorm（避免压缩候选之间的差异）
    """
    def __init__(self, box_dim=3, container_feature_dim=4, hidden_dim=64,
                 num_heads=4, num_inds=4, num_candidates=5, normFactor=0.0125):
        super(SetTransformerSelector, self).__init__()

        self.box_dim = box_dim
        self.container_feature_dim = container_feature_dim
        self.num_candidates = num_candidates
        self.normFactor = normFactor  # 用于将原始尺寸归一化到 [0,1]

        # 包裹尺寸编码
        self.input_embed = nn.Sequential(
            nn.Linear(box_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Set Transformer 编码器（不用 LayerNorm，直接 SAB）
        self.encoder = nn.Sequential(
            SAB(hidden_dim, hidden_dim, num_heads),
            SAB(hidden_dim, hidden_dim, num_heads),
        )

        # 全局池化
        self.global_pool = PMA(hidden_dim, num_heads, num_seeds=1)

        # 容器状态编码
        self.container_embed = nn.Sequential(
            nn.Linear(container_feature_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 评分头：候选编码 + 全局编码 + 容器编码
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Critic 头
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def _encode(self, candidates, container_state):
        """编码候选包裹和容器状态"""
        # 将原始尺寸归一化到 [0,1] 范围，与 container_state 的 scale 一致
        candidates_normalized = candidates * self.normFactor
        x = self.input_embed(candidates_normalized)
        encoded = self.encoder(x)
        global_feat = self.global_pool(encoded)
        container_feat = self.container_embed(container_state)
        return encoded, global_feat, container_feat

    def _compute_scores(self, encoded, global_feat, container_feat, num_candidates):
        """计算每个候选的得分"""
        global_expanded = global_feat.expand(-1, num_candidates, -1)
        container_expanded = container_feat.unsqueeze(1).expand(-1, num_candidates, -1)
        combined = torch.cat([encoded, global_expanded, container_expanded], dim=-1)
        scores = self.score_head(combined).squeeze(-1)
        return scores

    def forward(self, candidates, container_state, deterministic=False):
        batch_size = candidates.size(0)
        num_candidates = candidates.size(1)

        encoded, global_feat, container_feat = self._encode(candidates, container_state)
        scores = self._compute_scores(encoded, global_feat, container_feat, num_candidates)

        probs = F.softmax(scores, dim=-1)

        if deterministic:
            selected_idx = probs.argmax(dim=-1)
        else:
            dist = torch.distributions.Categorical(probs)
            selected_idx = dist.sample()

        log_probs = F.log_softmax(scores, dim=-1)
        selected_log_prob = log_probs.gather(1, selected_idx.view(-1, 1)).squeeze(-1)
        entropy = -(probs * log_probs).sum(dim=-1)

        critic_input = torch.cat([global_feat.squeeze(1), container_feat], dim=-1)
        value = self.critic_head(critic_input)

        return selected_idx, selected_log_prob, entropy, value

    def evaluate_actions(self, candidates, container_state, actions):
        batch_size = candidates.size(0)
        num_candidates = candidates.size(1)

        encoded, global_feat, container_feat = self._encode(candidates, container_state)
        scores = self._compute_scores(encoded, global_feat, container_feat, num_candidates)

        probs = F.softmax(scores, dim=-1)
        log_probs = F.log_softmax(scores, dim=-1)

        action_log_prob = log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        entropy = -(probs * log_probs).sum(dim=-1)

        critic_input = torch.cat([global_feat.squeeze(1), container_feat], dim=-1)
        value = self.critic_head(critic_input)

        return action_log_prob, entropy, value


# 测试代码
if __name__ == '__main__':
    batch_size = 4
    num_candidates = 5
    box_dim = 3
    container_feature_dim = 4

    model = SetTransformerSelector(
        box_dim=box_dim,
        container_feature_dim=container_feature_dim,
        hidden_dim=64,
        num_heads=4,
        num_inds=4
    )

    # 模拟真实输入：归一化后的包裹尺寸
    candidates = torch.tensor([
        [[0.21, 0.18, 0.06], [0.20, 0.15, 0.11], [0.35, 0.25, 0.08],
         [0.12, 0.10, 0.15], [0.28, 0.22, 0.09]]
    ] * batch_size)
    container_state = torch.tensor([[0.48, 0.60, 1.00, 0.72]] * batch_size)

    selected_idx, log_prob, entropy, value = model(candidates, container_state)

    print(f"Selected indices: {selected_idx}")
    print(f"Entropy: {entropy}")

    with torch.no_grad():
        encoded, global_feat, container_feat = model._encode(candidates, container_state)
        scores = model._compute_scores(encoded, global_feat, container_feat, num_candidates)
        print(f"\nScores per candidate: {scores[0]}")
        print(f"Score differences: {(scores[0] - scores[0].mean()).tolist()}")
        print(f"Max-min diff: {(scores[0].max() - scores[0].min()).item():.6f}")
    container_feature_dim = 4

    model = SetTransformerSelector(
        box_dim=box_dim,
        container_feature_dim=container_feature_dim,
        hidden_dim=64,
        num_heads=4,
        num_inds=4
    )

    # 模拟真实输入：归一化后的包裹尺寸
    candidates = torch.tensor([
        [[0.21, 0.18, 0.06], [0.20, 0.15, 0.11], [0.35, 0.25, 0.08],
         [0.12, 0.10, 0.15], [0.28, 0.22, 0.09]]
    ] * batch_size)
    container_state = torch.tensor([[0.48, 0.60, 1.00, 0.72]] * batch_size)

    selected_idx, log_prob, entropy, value = model(candidates, container_state)

    print(f"Selected indices: {selected_idx}")
    print(f"Entropy: {entropy}")

    with torch.no_grad():
        encoded, global_feat, container_feat = model._encode(candidates, container_state)
        scores = model._compute_scores(encoded, global_feat, container_feat, num_candidates)
        print(f"\nScores per candidate: {scores[0]}")
        print(f"Score differences: {(scores[0] - scores[0].mean()).tolist()}")
        print(f"Max-min diff: {(scores[0].max() - scores[0].min()).item():.6f}")
