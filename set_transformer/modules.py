"""
Set Transformer Modules

Based on: https://github.com/juho-lee/set_transformer
Paper: "Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks"

This module implements the core building blocks:
- MAB: Multi-head Attention Block
- SAB: Set Attention Block (self-attention)
- ISAB: Induced Set Attention Block (efficient version using inducing points)
- PMA: Pooling by Multi-head Attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MAB(nn.Module):
    """
    Multi-head Attention Block.
    
    Computes multi-head attention from query Q to key-value pair (K, V).
    
    Args:
        dim_Q: Dimension of query input
        dim_K: Dimension of key/value input  
        dim_V: Dimension of output
        num_heads: Number of attention heads
        ln: Whether to use layer normalization
    """
    
    def __init__(self, dim_Q: int, dim_K: int, dim_V: int, num_heads: int, ln: bool = False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        
        # Linear projections
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)
        
        # Layer normalization (optional)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        
        self.ln = ln
    
    def forward(self, Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        """
        Args:
            Q: Query tensor (batch, n_q, dim_Q)
            K: Key/Value tensor (batch, n_k, dim_K)
            
        Returns:
            Output tensor (batch, n_q, dim_V)
        """
        # Project to dim_V
        Q = self.fc_q(Q)  # (batch, n_q, dim_V)
        K = self.fc_k(K)  # (batch, n_k, dim_V)
        V = self.fc_v(K)  # (batch, n_k, dim_V)
        
        # Split for multi-head attention
        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, dim=2), dim=0)  # (batch*num_heads, n_q, dim_split)
        K_ = torch.cat(K.split(dim_split, dim=2), dim=0)  # (batch*num_heads, n_k, dim_split)
        V_ = torch.cat(V.split(dim_split, dim=2), dim=0)  # (batch*num_heads, n_k, dim_split)
        
        # Scaled dot-product attention
        A = torch.softmax(
            Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V),
            dim=2
        )  # (batch*num_heads, n_q, n_k)
        
        # Apply attention and concatenate heads
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), dim=0), dim=2)  # (batch, n_q, dim_V)
        
        # Layer norm + feedforward
        if self.ln:
            O = self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        if self.ln:
            O = self.ln1(O)
        
        return O


class SAB(nn.Module):
    """
    Set Attention Block.
    
    Self-attention over set elements. Permutation equivariant.
    
    Args:
        dim_in: Input dimension
        dim_out: Output dimension
        num_heads: Number of attention heads
        ln: Whether to use layer normalization
    """
    
    def __init__(self, dim_in: int, dim_out: int, num_heads: int, ln: bool = False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X: Input tensor (batch, n, dim_in)
            
        Returns:
            Output tensor (batch, n, dim_out)
        """
        return self.mab(X, X)


class ISAB(nn.Module):
    """
    Induced Set Attention Block.
    
    More efficient version of SAB using inducing points.
    Reduces O(n^2) complexity to O(n*m) where m is number of inducing points.
    
    Args:
        dim_in: Input dimension
        dim_out: Output dimension
        num_heads: Number of attention heads
        num_inds: Number of inducing points
        ln: Whether to use layer normalization
    """
    
    def __init__(self, dim_in: int, dim_out: int, num_heads: int, num_inds: int, ln: bool = False):
        super(ISAB, self).__init__()
        
        # Learnable inducing points
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        
        # Two MAB blocks
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X: Input tensor (batch, n, dim_in)
            
        Returns:
            Output tensor (batch, n, dim_out)
        """
        batch_size = X.size(0)
        
        # Inducing points attend to input
        H = self.mab0(self.I.repeat(batch_size, 1, 1), X)  # (batch, num_inds, dim_out)
        
        # Input attends to inducing point representation
        return self.mab1(X, H)  # (batch, n, dim_out)


class PMA(nn.Module):
    """
    Pooling by Multi-head Attention.
    
    Aggregates set to fixed number of outputs using learnable seed vectors.
    
    Args:
        dim: Dimension of input/output
        num_heads: Number of attention heads
        num_seeds: Number of output vectors (seeds)
        ln: Whether to use layer normalization
    """
    
    def __init__(self, dim: int, num_heads: int, num_seeds: int, ln: bool = False):
        super(PMA, self).__init__()
        
        # Learnable seed vectors
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X: Input tensor (batch, n, dim)
            
        Returns:
            Output tensor (batch, num_seeds, dim)
        """
        batch_size = X.size(0)
        return self.mab(self.S.repeat(batch_size, 1, 1), X)


class SetTransformerEncoder(nn.Module):
    """
    Set Transformer Encoder.
    
    Encodes a set of elements with permutation equivariance.
    
    Args:
        dim_input: Input feature dimension
        dim_hidden: Hidden dimension
        num_heads: Number of attention heads
        num_inds: Number of inducing points for ISAB
        num_layers: Number of ISAB layers
        ln: Whether to use layer normalization
    """
    
    def __init__(self,
                 dim_input: int,
                 dim_hidden: int,
                 num_heads: int = 4,
                 num_inds: int = 16,
                 num_layers: int = 2,
                 ln: bool = True):
        super(SetTransformerEncoder, self).__init__()
        
        layers = []
        # First layer: dim_input -> dim_hidden
        layers.append(ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln))
        # Subsequent layers: dim_hidden -> dim_hidden
        for _ in range(num_layers - 1):
            layers.append(ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln))
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X: Input tensor (batch, n, dim_input)
            
        Returns:
            Encoded tensor (batch, n, dim_hidden)
        """
        return self.layers(X)
