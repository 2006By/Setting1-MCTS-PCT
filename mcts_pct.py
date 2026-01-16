"""
MCTS-PCT: Monte Carlo Tree Search with Frozen PCT Model

This module implements MCTS enhanced by a pretrained PCT model serving as
Prior (Actor) and Value (Critic) estimator. Key features:
- Wasted Space Penalty (WSP) for reward modification
- PUCT-based selection (AlphaZero style)
- Lookahead horizon for multi-step planning

Paper reference: Learning Efficient Online 3D Bin Packing on Packing Configuration Trees
"""

import torch
import numpy as np
import math
import copy
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field


# ============================================================================
# Hyperparameters (Paper Section 4.2)
# ============================================================================
@dataclass
class MCTSConfig:
    """Configuration for MCTS-PCT algorithm."""
    lookahead_horizon: int = 4      # N: Max tree depth / lookahead window size
    n_simulations: int = 100        # n: MCTS simulations per decision
    discount_factor: float = 1.0    # γ: Discount (1.0 for finite tasks)
    wsp_weight: float = 0.5         # λ: Wasted Space Penalty weight
    c_puct: float = 1.0             # Exploration constant for PUCT


# ============================================================================
# MCTS Node Data Structure
# ============================================================================
class MCTSNodePCT:
    """
    Node in MCTS tree for bin packing with PCT integration.
    
    Each node represents a state after placing items, with statistics
    for PUCT selection and WSP-modified rewards.
    """
    
    def __init__(self, 
                 bin_state: Dict[str, Any],
                 lookahead_items: List[Tuple],
                 parent: Optional['MCTSNodePCT'] = None,
                 action_from_parent: Optional[int] = None,
                 prior: float = 0.0):
        """
        Args:
            bin_state: Snapshot of Space object state (from space.get_state())
            lookahead_items: Queue of upcoming items [(l,w,h), ...]
            parent: Parent node in tree
            action_from_parent: Leaf node index that led to this state
            prior: Prior probability P(a|s) from PCT Actor
        """
        # State information
        self.bin_state = bin_state
        self.lookahead_items = lookahead_items.copy() if lookahead_items else []
        
        # Tree structure
        self.parent = parent
        self.children: Dict[int, 'MCTSNodePCT'] = {}
        self.action_from_parent = action_from_parent
        
        # MCTS statistics
        self.N = 0                    # Visit count
        self.W = 0.0                  # Total value (sum of backpropagated values)
        self.P = prior                # Prior probability from PCT Actor
        
        # WSP reward (cached when node is created via expansion)
        self.immediate_reward = 0.0   # r' = R_vol - λ * R_p
        
        # Terminal flag cache
        self._is_terminal: Optional[bool] = None
        self._depth: Optional[int] = None
    
    @property
    def Q(self) -> float:
        """Mean action value Q(s,a) = W/N."""
        if self.N == 0:
            return 0.0
        return self.W / self.N
    
    @property
    def depth(self) -> int:
        """Depth of this node in the tree."""
        if self._depth is None:
            if self.parent is None:
                self._depth = 0
            else:
                self._depth = self.parent.depth + 1
        return self._depth
    
    def is_terminal(self, max_depth: int = 4) -> bool:
        """
        Check if node is terminal.
        
        Terminal conditions:
        1. Reached max lookahead depth
        2. No more items in lookahead queue
        3. Bin is full (no valid placements)
        """
        if self._is_terminal is None:
            self._is_terminal = (
                self.depth >= max_depth or
                len(self.lookahead_items) == 0
            )
        return self._is_terminal
    
    def is_fully_expanded(self) -> bool:
        """Check if all children have been created."""
        return len(self.children) > 0
    
    def ucb_score(self, c_puct: float, parent_visits: int) -> float:
        """
        Calculate PUCT score for selection.
        
        Formula: Q(s,a) + c_puct * P(s,a) * sqrt(N_parent) / (1 + N(s,a))
        """
        exploitation = self.Q
        exploration = c_puct * self.P * math.sqrt(parent_visits) / (1 + self.N)
        return exploitation + exploration
    
    def select_best_child(self, c_puct: float) -> 'MCTSNodePCT':
        """Select child with highest PUCT score."""
        if not self.children:
            return self
        
        parent_visits = sum(child.N for child in self.children.values())
        if parent_visits == 0:
            parent_visits = 1
            
        best_child = None
        best_score = float('-inf')
        
        for child in self.children.values():
            score = child.ucb_score(c_puct, parent_visits)
            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child
    
    def get_path_rewards(self) -> float:
        """Get cumulative immediate rewards from root to this node."""
        total = 0.0
        node = self
        while node is not None:
            total += node.immediate_reward
            node = node.parent
        return total


# ============================================================================
# Wasted Space Penalty (WSP) Calculator
# ============================================================================
def calculate_wsp(space_boxes: List[Dict], 
                  placed_box: Dict, 
                  bin_size: Tuple[int, int, int],
                  lambda_wsp: float = 0.5) -> Tuple[float, float]:
    """
    Calculate Wasted Space Penalty for a placed box.
    
    WSP measures the volume of unusable space created beneath a placed item.
    This penalizes "floating" placements that waste space.
    
    Args:
        space_boxes: List of existing boxes in bin (from state['boxes_data'])
        placed_box: Dict with keys {x, y, z, lx, ly, lz} for the new box
        bin_size: (L, W, H) dimensions of container
        lambda_wsp: Penalty weight
        
    Returns:
        Tuple of (modified_reward r', raw_volume_reward R_vol)
    """
    box_x, box_y, box_z = placed_box['x'], placed_box['y'], placed_box['z']
    box_lx, box_ly, box_lz = placed_box['lx'], placed_box['ly'], placed_box['lz']
    
    box_base_area = box_x * box_y
    bin_volume = bin_size[0] * bin_size[1] * bin_size[2]
    
    # Volume reward (normalized)
    R_vol = (box_x * box_y * box_z) / bin_volume
    
    # If on floor, no wasted space
    if box_lz == 0:
        return R_vol, R_vol
    
    # Calculate contact area with supporting boxes
    total_contact_area = 0.0
    wasted_volume = 0.0
    
    for support in space_boxes:
        support_top_z = support['lz'] + support['z']
        
        # Check if this box could be supporting
        if support_top_z <= box_lz:
            # Calculate 2D intersection
            x1 = max(box_lx, support['lx'])
            y1 = max(box_ly, support['ly'])
            x2 = min(box_lx + box_x, support['lx'] + support['x'])
            y2 = min(box_ly + box_y, support['ly'] + support['y'])
            
            if x2 > x1 and y2 > y1:
                contact_area = (x2 - x1) * (y2 - y1)
                total_contact_area += contact_area
                
                # Gap between support top and placed box bottom
                gap_height = box_lz - support_top_z
                wasted_volume += contact_area * gap_height
    
    # Unsupported area: wasted from floor to box bottom
    unsupported_area = max(0, box_base_area - total_contact_area)
    wasted_volume += unsupported_area * box_lz
    
    # Normalize penalty
    R_p = wasted_volume / bin_volume
    
    # Modified reward
    r_prime = R_vol - lambda_wsp * R_p
    
    return r_prime, R_vol


# ============================================================================
# MCTS Planner with PCT Integration
# ============================================================================
class MCTSPlannerPCT:
    """
    MCTS Planner using frozen PCT as Prior and Value estimator.
    
    The PCT Actor provides action prior P(s,a) during expansion.
    The PCT Critic provides state value V(s) during evaluation.
    WSP modifies rewards to penalize poor placements.
    """
    
    def __init__(self,
                 pct_policy,
                 env,
                 config: MCTSConfig = None,
                 device: str = 'cuda'):
        """
        Args:
            pct_policy: Frozen DRL_GAT model (Actor-Critic)
            env: PackingDiscrete environment instance
            config: MCTS hyperparameters
            device: 'cuda' or 'cpu'
        """
        self.pct_policy = pct_policy
        self.env = env
        self.config = config or MCTSConfig()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Ensure model is in eval mode
        self.pct_policy.eval()
        
        # Cache environment parameters
        self.bin_size = env.bin_size
        self.internal_node_holder = env.internal_node_holder
        self.leaf_node_holder = env.leaf_node_holder
        self.next_holder = env.next_holder
        self.setting = env.setting
    
    def search(self, 
               current_state: Dict,
               lookahead_items: List[Tuple]) -> Tuple[int, Dict]:
        """
        Run MCTS search to find best action (leaf node index).
        
        Args:
            current_state: Space state snapshot from env.space.get_state()
            lookahead_items: Upcoming items in window [(l,w,h), ...]
            
        Returns:
            Tuple of (best_action_idx, search_stats)
        """
        # Create root node
        root = MCTSNodePCT(
            bin_state=current_state,
            lookahead_items=lookahead_items
        )
        
        # Run simulations
        for sim_idx in range(self.config.n_simulations):
            # 1. Selection: traverse to leaf via PUCT
            leaf = self._select(root)
            
            # 2. Expansion: create children with PCT priors
            if not leaf.is_terminal(self.config.lookahead_horizon):
                if not leaf.is_fully_expanded():
                    leaf = self._expand(leaf)
            
            # 3. Evaluation: get value from PCT Critic
            value = self._evaluate(leaf)
            
            # 4. Backup: propagate value up the tree
            self._backup(leaf, value)
        
        # Select best action by visit count
        if not root.children:
            return 0, {'error': 'No valid actions found'}
        
        best_child = max(root.children.values(), key=lambda n: n.N)
        
        # Collect statistics
        stats = {
            'n_simulations': self.config.n_simulations,
            'root_visits': sum(c.N for c in root.children.values()),
            'best_action': best_child.action_from_parent,
            'best_visits': best_child.N,
            'best_q': best_child.Q,
            'action_distribution': {
                action: child.N for action, child in root.children.items()
            }
        }
        
        return best_child.action_from_parent, stats
    
    def _select(self, node: MCTSNodePCT) -> MCTSNodePCT:
        """Selection phase: traverse tree using PUCT until leaf."""
        while node.is_fully_expanded() and not node.is_terminal(self.config.lookahead_horizon):
            node = node.select_best_child(self.config.c_puct)
        return node
    
    def _expand(self, node: MCTSNodePCT) -> MCTSNodePCT:
        """
        Expansion phase: create children with PCT Actor priors.
        
        Uses PCT to get action probabilities and applies action mask
        for invalid placements.
        """
        if len(node.lookahead_items) == 0:
            return node
        
        # Get current item to place
        current_item = node.lookahead_items[0]
        remaining_items = node.lookahead_items[1:]
        
        # Restore environment state
        self.env.space.restore_state(node.bin_state)
        
        # Get valid placements (leaf nodes) for current item
        self.env.next_box = list(current_item[:3])  # (l, w, h)
        if len(current_item) > 3:
            self.env.next_den = current_item[3]  # density if available
        else:
            self.env.next_den = 1.0
        
        leaf_node_vec = self.env.get_possible_position()
        
        # Get valid action mask
        valid_mask = leaf_node_vec[:, 8] > 0  # Flag column
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) == 0:
            # No valid placements, mark as terminal
            node._is_terminal = True
            return node
        
        # Get PCT observation and prior probabilities
        observation = self.env.cur_observation()
        obs_tensor = torch.FloatTensor(observation).reshape(
            1, self.internal_node_holder + self.leaf_node_holder + self.next_holder, 9
        ).to(self.device)
        
        with torch.no_grad():
            # PCT forward pass
            _, _, _, _, dist = self.pct_policy.actor(
                obs_tensor, 
                evaluate_action=True,
                normFactor=self.env.normFactor if hasattr(self.env, 'normFactor') else 1.0
            )
            priors = dist.probs.cpu().numpy()[0]  # (leaf_node_holder,)
        
        # Apply mask and renormalize
        masked_priors = np.zeros_like(priors)
        masked_priors[valid_indices] = priors[valid_indices]
        prior_sum = masked_priors.sum()
        if prior_sum > 0:
            masked_priors /= prior_sum
        else:
            # Uniform if all priors are zero
            masked_priors[valid_indices] = 1.0 / len(valid_indices)
        
        # Create child nodes for valid actions
        for action_idx in valid_indices:
            if action_idx in node.children:
                continue
            
            # Simulate action to get next state
            leaf_node = leaf_node_vec[action_idx]
            action, next_box = self.env.LeafNode2Action(leaf_node)
            
            # Execute virtual placement
            self.env.space.restore_state(node.bin_state)
            success = self.env.space.drop_box(
                next_box, 
                (action[1], action[2]), 
                action[0],
                self.env.next_den,
                self.setting
            )
            
            if not success:
                continue
            
            # Get new state
            new_state = self.env.space.get_state()
            
            # Calculate WSP reward
            placed_box = {
                'x': next_box[0], 'y': next_box[1], 'z': next_box[2],
                'lx': action[1], 'ly': action[2], 
                'lz': self.env.space.boxes[-1].lz if self.env.space.boxes else 0
            }
            # Get actual lz from the placed box
            if len(self.env.space.boxes) > 0:
                last_box = self.env.space.boxes[-1]
                placed_box['lz'] = last_box.lz
            
            r_prime, _ = calculate_wsp(
                new_state['boxes_data'],
                placed_box,
                self.bin_size,
                self.config.wsp_weight
            )
            
            # Create child node
            child = MCTSNodePCT(
                bin_state=new_state,
                lookahead_items=remaining_items,
                parent=node,
                action_from_parent=int(action_idx),
                prior=masked_priors[action_idx]
            )
            child.immediate_reward = r_prime
            node.children[int(action_idx)] = child
        
        # Return a child for simulation (prefer unvisited)
        for child in node.children.values():
            if child.N == 0:
                return child
        
        return node.select_best_child(self.config.c_puct) if node.children else node
    
    def _evaluate(self, node: MCTSNodePCT) -> float:
        """
        Evaluation phase: get value from PCT Critic.
        
        Returns cumulative path reward + Critic value estimate.
        """
        if node.is_terminal(self.config.lookahead_horizon):
            # Terminal: use current space ratio as value
            self.env.space.restore_state(node.bin_state)
            return node.get_path_rewards()
        
        # Get Critic value estimate
        self.env.space.restore_state(node.bin_state)
        
        if len(node.lookahead_items) > 0:
            current_item = node.lookahead_items[0]
            self.env.next_box = list(current_item[:3])
            self.env.next_den = current_item[3] if len(current_item) > 3 else 1.0
        
        observation = self.env.cur_observation()
        obs_tensor = torch.FloatTensor(observation).reshape(
            1, self.internal_node_holder + self.leaf_node_holder + self.next_holder, 9
        ).to(self.device)
        
        with torch.no_grad():
            _, _, _, values = self.pct_policy(obs_tensor, deterministic=True)
            critic_value = values.item()
        
        # Path value = cumulative rewards + Critic estimate
        path_rewards = node.get_path_rewards()
        return path_rewards + critic_value
    
    def _backup(self, node: MCTSNodePCT, value: float):
        """Backup phase: propagate value up to root."""
        current = node
        while current is not None:
            current.N += 1
            current.W += value
            current = current.parent


# ============================================================================
# Utility Functions
# ============================================================================
def get_action_probs_from_root(root: MCTSNodePCT, 
                               temperature: float = 1.0) -> np.ndarray:
    """
    Get action probability distribution from root node visit counts.
    
    Args:
        root: Root node after MCTS search
        temperature: Temperature for softmax (0 = deterministic)
        
    Returns:
        Probability array over all actions
    """
    if not root.children:
        return np.array([])
    
    # Collect visit counts
    max_action = max(root.children.keys()) + 1
    visits = np.zeros(max_action)
    
    for action_idx, child in root.children.items():
        visits[action_idx] = child.N
    
    if temperature == 0:
        # Deterministic: all mass on best action
        probs = np.zeros_like(visits)
        probs[np.argmax(visits)] = 1.0
    else:
        # Temperature-scaled
        visits = visits ** (1.0 / temperature)
        total = visits.sum()
        if total > 0:
            probs = visits / total
        else:
            probs = np.ones_like(visits) / len(visits)
    
    return probs
