"""
Package Selector Model

This module implements the PackageSelector using Set Transformer architecture.
It selects the optimal package from a candidate queue based on current bin state.

The model uses:
- Set Transformer encoder for permutation-invariant package encoding
- Cross-attention to fuse package features with bin state
- Score head to output selection probabilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict, Any

from .modules import ISAB, MAB, SAB, PMA


class PackageSelector(nn.Module):
    """
    Package Selector based on Set Transformer.
    
    Selects the optimal package from K candidates based on current bin state.
    Uses Set Transformer for permutation-invariant processing of candidates.
    
    Args:
        bin_state_dim: Dimension of bin state features
        package_dim: Dimension of package features (l, w, h, density, etc.)
        hidden_dim: Hidden layer dimension
        num_heads: Number of attention heads
        num_inds: Number of inducing points for ISAB
        num_layers: Number of Set Attention layers
        ln: Whether to use layer normalization
    """
    
    def __init__(self,
                 bin_state_dim: int = 32,
                 package_dim: int = 8,
                 hidden_dim: int = 128,
                 num_heads: int = 4,
                 num_inds: int = 16,
                 num_layers: int = 2,
                 ln: bool = True):
        super(PackageSelector, self).__init__()
        
        self.bin_state_dim = bin_state_dim
        self.package_dim = package_dim
        self.hidden_dim = hidden_dim
        
        # 1. Package encoder (permutation equivariant)
        self.package_input = nn.Linear(package_dim, hidden_dim)
        self.package_encoder = nn.Sequential(
            ISAB(hidden_dim, hidden_dim, num_heads, num_inds, ln=ln),
            ISAB(hidden_dim, hidden_dim, num_heads, num_inds, ln=ln),
        )
        
        # 2. Bin state encoder
        self.bin_encoder = nn.Sequential(
            nn.Linear(bin_state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # 3. Cross-attention: packages attend to bin state
        self.cross_attention = MAB(hidden_dim, hidden_dim, hidden_dim, num_heads, ln=ln)
        
        # 4. Score head: output score for each package
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(self, 
                bin_state: torch.Tensor, 
                packages: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            bin_state: Bin state features (batch, bin_state_dim)
            packages: Package features (batch, K, package_dim)
            mask: Optional mask for invalid packages (batch, K), True = invalid
            
        Returns:
            scores: Raw scores for each package (batch, K)
            probs: Softmax probabilities (batch, K)
        """
        batch_size = packages.size(0)
        K = packages.size(1)
        
        # Encode packages
        pkg_emb = self.package_input(packages)  # (batch, K, hidden)
        pkg_emb = self.package_encoder(pkg_emb)  # (batch, K, hidden)
        
        # Encode bin state
        bin_emb = self.bin_encoder(bin_state)  # (batch, hidden)
        bin_emb = bin_emb.unsqueeze(1)  # (batch, 1, hidden)
        
        # Cross-attention: packages attend to bin state
        fused = self.cross_attention(pkg_emb, bin_emb)  # (batch, K, hidden)
        
        # Score each package
        scores = self.score_head(fused).squeeze(-1)  # (batch, K)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
        
        # Softmax probabilities
        probs = F.softmax(scores, dim=-1)
        
        return scores, probs
    
    def select(self, 
               bin_state: torch.Tensor, 
               packages: torch.Tensor,
               mask: Optional[torch.Tensor] = None,
               deterministic: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select the best package.
        
        Args:
            bin_state: Bin state features (batch, bin_state_dim)
            packages: Package features (batch, K, package_dim)
            mask: Optional mask for invalid packages
            deterministic: If True, select argmax; else sample
            
        Returns:
            selected_idx: Index of selected package (batch,)
            probs: Selection probabilities (batch, K)
        """
        scores, probs = self.forward(bin_state, packages, mask)
        
        if deterministic:
            selected_idx = scores.argmax(dim=-1)
        else:
            # Sample from distribution
            selected_idx = torch.multinomial(probs, 1).squeeze(-1)
        
        return selected_idx, probs
    
    def get_log_prob(self,
                     bin_state: torch.Tensor,
                     packages: torch.Tensor,
                     actions: torch.Tensor,
                     mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get log probability of actions (for RL training).
        
        Args:
            bin_state: Bin state features (batch, bin_state_dim)
            packages: Package features (batch, K, package_dim)
            actions: Selected package indices (batch,)
            mask: Optional mask for invalid packages
            
        Returns:
            log_probs: Log probabilities (batch,)
        """
        scores, probs = self.forward(bin_state, packages, mask)
        log_probs = F.log_softmax(scores, dim=-1)
        return log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)


class BinStateEncoder:
    """
    Extracts bin state features from environment.
    
    Features include:
    - Current space utilization ratio
    - Number of placed boxes (normalized)
    - EMS (Empty Maximal Space) count
    - Height map statistics
    - Box placement statistics
    """
    
    def __init__(self, state_dim: int = 32, container_size: Tuple[int, int, int] = (10, 10, 10)):
        self.state_dim = state_dim
        self.container_size = container_size
        self.container_volume = container_size[0] * container_size[1] * container_size[2]
    
    def extract(self, env) -> np.ndarray:
        """
        Extract bin state feature vector from environment.
        
        Args:
            env: Packing environment with space attribute
            
        Returns:
            state: Feature vector (state_dim,)
        """
        space = env.space
        features = []
        
        # 1. Space utilization ratio
        features.append(space.get_ratio())
        
        # 2. Number of placed boxes (normalized)
        num_boxes = len(space.boxes) if hasattr(space, 'boxes') else 0
        features.append(num_boxes / 100.0)
        
        # 3. EMS count (normalized)
        ems_count = len(space.EMS) if hasattr(space, 'EMS') else 0
        features.append(ems_count / 50.0)
        
        # 4. Available height ratio (average remaining height)
        if hasattr(space, 'plain') and space.plain is not None:
            height_map = space.plain
            avg_height = np.mean(height_map) if height_map.size > 0 else 0
            max_height = np.max(height_map) if height_map.size > 0 else 0
            features.append(avg_height / self.container_size[2])
            features.append(max_height / self.container_size[2])
        else:
            features.extend([0.0, 0.0])
        
        # 5. Statistics from box_vec
        if hasattr(space, 'box_vec'):
            box_vec = space.box_vec  # (internal_node_holder, 9)
            valid_mask = box_vec[:, -1] > 0
            if valid_mask.sum() > 0:
                valid_boxes = box_vec[valid_mask]
                # Average normalized dimensions
                for i in range(3, 6):  # dimensions
                    features.append(float(np.mean(valid_boxes[:, i])))
                # Average normalized positions
                for i in range(6, 9):  # positions
                    features.append(float(np.mean(valid_boxes[:, i])))
            else:
                features.extend([0.0] * 6)
        else:
            features.extend([0.0] * 6)
        
        # 6. Density statistics (if available)
        if hasattr(space, 'boxes') and len(space.boxes) > 0:
            densities = [b.den if hasattr(b, 'den') else 1.0 for b in space.boxes]
            features.append(np.mean(densities))
            features.append(np.std(densities) if len(densities) > 1 else 0.0)
        else:
            features.extend([0.0, 0.0])
        
        # Pad to fixed dimension
        state = np.zeros(self.state_dim, dtype=np.float32)
        state[:min(len(features), self.state_dim)] = features[:self.state_dim]
        
        return state
    
    def extract_from_state_dict(self, state_dict: Dict[str, Any]) -> np.ndarray:
        """
        Extract features from a state dictionary (for MCTS nodes).
        
        Args:
            state_dict: State dictionary from space.get_state()
            
        Returns:
            state: Feature vector (state_dim,)
        """
        features = []
        
        # 1. Utilization from boxes_data
        boxes_data = state_dict.get('boxes_data', [])
        total_volume = sum(b['x'] * b['y'] * b['z'] for b in boxes_data)
        ratio = total_volume / self.container_volume
        features.append(ratio)
        
        # 2. Number of boxes
        features.append(len(boxes_data) / 100.0)
        
        # 3. EMS count
        ems = state_dict.get('EMS', [])
        features.append(len(ems) / 50.0)
        
        # 4-5. Height statistics
        if boxes_data:
            heights = [b['lz'] + b['z'] for b in boxes_data]
            features.append(np.mean(heights) / self.container_size[2])
            features.append(np.max(heights) / self.container_size[2])
        else:
            features.extend([0.0, 0.0])
        
        # 6-11. Box statistics
        if boxes_data:
            dims = np.array([[b['x'], b['y'], b['z']] for b in boxes_data])
            pos = np.array([[b['lx'], b['ly'], b['lz']] for b in boxes_data])
            for i in range(3):
                features.append(np.mean(dims[:, i]) / self.container_size[i])
            for i in range(3):
                features.append(np.mean(pos[:, i]) / self.container_size[i])
        else:
            features.extend([0.0] * 6)
        
        # Pad to fixed dimension
        state = np.zeros(self.state_dim, dtype=np.float32)
        state[:min(len(features), self.state_dim)] = features[:self.state_dim]
        
        return state


class PackageFeatureEncoder:
    """
    Encodes package features for the selector model.
    
    Features include:
    - Normalized dimensions (l, w, h)
    - Density
    - Volume ratio
    - Aspect ratio
    - Surface area ratio
    """
    
    def __init__(self, 
                 package_dim: int = 8, 
                 container_size: Tuple[int, int, int] = (10, 10, 10)):
        self.package_dim = package_dim
        self.container_size = container_size
        self.container_volume = container_size[0] * container_size[1] * container_size[2]
    
    def encode(self, packages: List[Tuple]) -> np.ndarray:
        """
        Encode package features.
        
        Args:
            packages: List of packages [(l, w, h, [density]), ...]
            
        Returns:
            features: Feature matrix (K, package_dim)
        """
        K = len(packages)
        features = np.zeros((K, self.package_dim), dtype=np.float32)
        
        for i, pkg in enumerate(packages):
            l, w, h = pkg[:3]
            density = pkg[3] if len(pkg) > 3 else 1.0
            
            # Normalized dimensions
            features[i, 0] = l / self.container_size[0]
            features[i, 1] = w / self.container_size[1]
            features[i, 2] = h / self.container_size[2]
            
            # Density
            features[i, 3] = density
            
            # Volume ratio
            volume = l * w * h
            features[i, 4] = volume / self.container_volume
            
            # Aspect ratio (flatness)
            sorted_dims = sorted([l, w, h])
            features[i, 5] = sorted_dims[0] / (sorted_dims[2] + 1e-6)
            
            # Surface area ratio
            surface = 2 * (l * w + w * h + l * h)
            max_surface = 2 * sum(self.container_size[i] * self.container_size[j] 
                                   for i in range(3) for j in range(i+1, 3))
            features[i, 6] = surface / max_surface
            
            # Valid flag
            features[i, 7] = 1.0
        
        return features
    
    def encode_batch(self, batch_packages: List[List[Tuple]]) -> np.ndarray:
        """
        Encode a batch of package lists.
        
        Args:
            batch_packages: List of package lists
            
        Returns:
            features: Feature tensor (batch, K, package_dim)
        """
        batch_size = len(batch_packages)
        max_K = max(len(pkgs) for pkgs in batch_packages)
        
        features = np.zeros((batch_size, max_K, self.package_dim), dtype=np.float32)
        
        for b, packages in enumerate(batch_packages):
            features[b, :len(packages), :] = self.encode(packages)
        
        return features


class PCTGuidedLabelGenerator:
    """
    Generates training labels using PCT Actor/Critic.
    
    For each candidate package, evaluates its value using PCT Critic
    and generates soft labels based on these values.
    """
    
    def __init__(self, pct_policy, env, device: str = 'cuda', temperature: float = 1.0):
        """
        Args:
            pct_policy: Pretrained PCT model (DRL_GAT)
            env: Packing environment
            device: Computation device
            temperature: Softmax temperature for soft labels
        """
        self.pct_policy = pct_policy
        self.env = env
        self.device = torch.device(device)
        self.temperature = temperature
        
        # Ensure model is in eval mode
        self.pct_policy.eval()
    
    @torch.no_grad()
    def generate_labels(self, 
                        current_state: Dict[str, Any],
                        candidate_packages: List[Tuple]) -> Tuple[int, np.ndarray]:
        """
        Generate labels for candidate packages.
        
        Uses PCT Critic to evaluate the value of placing each package.
        
        Args:
            current_state: Current bin state (from space.get_state())
            candidate_packages: List of candidate packages
            
        Returns:
            best_idx: Index of best package (hard label)
            soft_labels: Soft probability labels (K,)
        """
        K = len(candidate_packages)
        values = []
        
        for pkg in candidate_packages:
            # Restore state
            self.env.space.set_state(current_state)
            
            # Set package
            self.env.next_box = list(pkg[:3])
            self.env.next_den = pkg[3] if len(pkg) > 3 else 1.0
            
            # Get valid placements
            leaf_nodes = self.env.get_possible_position()
            valid_mask = leaf_nodes[:, 8] > 0
            
            if valid_mask.sum() == 0:
                # No valid placement
                values.append(-1.0)
                continue
            
            # Build observation
            next_box_sorted = sorted(list(pkg[:3]))
            next_den = pkg[3] if len(pkg) > 3 else 1.0
            
            next_box_vec = np.zeros((1, 9))
            next_box_vec[:, 3:6] = next_box_sorted
            next_box_vec[:, 0] = next_den
            next_box_vec[:, -1] = 1
            
            observation = np.concatenate([
                self.env.space.box_vec.reshape(-1),
                leaf_nodes.reshape(-1),
                next_box_vec.reshape(-1)
            ])
            
            # Get observation dimensions from environment
            internal_holder = self.env.internal_node_holder
            leaf_holder = self.env.leaf_node_holder
            next_holder = 1
            
            obs_tensor = torch.FloatTensor(observation).reshape(
                1, internal_holder + leaf_holder + next_holder, 9
            ).to(self.device)
            
            # Get Critic value
            normFactor = self.env.normFactor if hasattr(self.env, 'normFactor') else 1.0
            _, _, _, value = self.pct_policy(obs_tensor, deterministic=True)
            values.append(value.item())
        
        # Restore state
        self.env.space.set_state(current_state)
        
        # Convert to numpy
        values = np.array(values)
        
        # Handle all invalid case
        if np.all(values < 0):
            return 0, np.ones(K) / K
        
        # Replace invalid with minimum valid value
        valid_mask = values >= 0
        if not np.all(valid_mask):
            min_valid = values[valid_mask].min() if valid_mask.sum() > 0 else 0
            values[~valid_mask] = min_valid - 1.0
        
        # Soft labels via temperature-scaled softmax
        values_scaled = values / self.temperature
        soft_labels = np.exp(values_scaled - values_scaled.max())
        soft_labels = soft_labels / soft_labels.sum()
        
        # Hard label
        best_idx = int(np.argmax(values))
        
        return best_idx, soft_labels
