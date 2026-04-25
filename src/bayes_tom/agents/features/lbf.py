"""
Behavior Fingerprint Extraction for Level-Based Foraging (LBF)
Inspired by ReCoLLAB paper's approach for teammate modeling
"""

import jax
import jax.numpy as jnp
import chex
from typing import Dict, List, Tuple
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class BehaviorFingerprint:
    """Container for extracted behavioral features from LBF trajectories"""
    
    # Action statistics
    action_histogram: chex.Array  # Shape: (6,) - counts of each action type
    action_entropy: float  # Entropy of action distribution
    
    # Movement patterns
    avg_displacement: float  # Average distance moved per step
    grid_coverage: float  # Percentage of unique positions visited
    
    # Food interaction patterns
    num_load_attempts: int  # Number of times agent attempted to load food
    successful_loads: int  # Number of successful food loads
    load_success_rate: float  # successful_loads / num_load_attempts
    time_to_first_load: int  # Steps until first load attempt
    
    # Spatial behavior
    dwell_time_by_quadrant: chex.Array  # Shape: (4,) - time spent in each quadrant
    avg_distance_to_nearest_food: float  # Average distance to closest food
    avg_distance_to_partner: float  # Average distance to teammate
    
    # Cooperation indicators
    num_adjacent_to_food_with_partner: int  # Times both agents near same food
    num_blocked_moves: int  # Times agent couldn't move (collision)
    coordination_events: int  # Times both agents loaded same food
    
    # Performance metrics
    cumulative_reward: float  # Total reward accumulated
    avg_reward_per_step: float  # Average reward per timestep
    
    # Temporal features
    steps_recorded: int  # Number of steps in this fingerprint


class LBFBehaviorExtractor:
    """Extracts behavior fingerprints from LBF environment trajectories"""
    
    def __init__(self, grid_size: int = 7, num_agents: int = 2):
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.num_actions = 6  # noop, up, down, left, right, load
        
    def extract_fingerprint(
        self,
        observations: List[chex.Array],
        actions: List[chex.Array],
        rewards: List[chex.Array],
        states: List = None,
        partner_idx: int = 1,
        probe_length: int = None
    ) -> BehaviorFingerprint:
        """
        Extract behavior fingerprint for a partner agent from trajectory data.
        
        Args:
            observations: List of observation arrays from environment
            actions: List of action arrays (shape: num_agents,)
            rewards: List of reward arrays (shape: num_agents,)
            states: List of State objects (optional, for detailed analysis)
            partner_idx: Index of the partner agent to analyze (0 or 1)
            probe_length: If provided, only analyze first N steps
            
        Returns:
            BehaviorFingerprint object containing extracted features
        """
        if probe_length is not None:
            observations = observations[:probe_length]
            actions = actions[:probe_length]
            rewards = rewards[:probe_length]
            if states is not None:
                states = states[:probe_length]
        
        num_steps = len(actions)
        
        # Extract partner actions
        partner_actions = jnp.array([act[partner_idx] for act in actions])
        
        # Action statistics
        action_histogram = self._compute_action_histogram(partner_actions)
        action_entropy = self._compute_entropy(action_histogram)
        
        # Movement and spatial features
        positions = self._extract_positions(observations, states, partner_idx)
        avg_displacement = self._compute_avg_displacement(positions)
        grid_coverage = self._compute_grid_coverage(positions)
        dwell_time_by_quadrant = self._compute_quadrant_dwell_time(positions)
        
        # Food interaction features
        load_attempts = jnp.sum(partner_actions == 5)  # Action 5 is 'load'
        food_features = self._analyze_food_interactions(
            observations, actions, states, partner_idx
        )
        
        # Cooperation features
        coop_features = self._analyze_cooperation(
            observations, actions, states, partner_idx
        )
        
        # Performance metrics
        partner_rewards = jnp.array([r[partner_idx] for r in rewards])
        cumulative_reward = float(jnp.sum(partner_rewards))
        avg_reward = float(jnp.mean(partner_rewards))
        
        return BehaviorFingerprint(
            action_histogram=action_histogram,
            action_entropy=float(action_entropy),
            avg_displacement=float(avg_displacement),
            grid_coverage=float(grid_coverage),
            num_load_attempts=int(load_attempts),
            successful_loads=food_features['successful_loads'],
            load_success_rate=food_features['load_success_rate'],
            time_to_first_load=food_features['time_to_first_load'],
            dwell_time_by_quadrant=dwell_time_by_quadrant,
            avg_distance_to_nearest_food=food_features['avg_dist_to_food'],
            avg_distance_to_partner=coop_features['avg_dist_to_partner'],
            num_adjacent_to_food_with_partner=coop_features['adjacent_to_food'],
            num_blocked_moves=coop_features['blocked_moves'],
            coordination_events=coop_features['coordination_events'],
            cumulative_reward=cumulative_reward,
            avg_reward_per_step=avg_reward,
            steps_recorded=num_steps
        )
    
    def _compute_action_histogram(self, actions: chex.Array) -> chex.Array:
        """Compute normalized histogram of actions"""
        hist = jnp.bincount(actions, length=self.num_actions)
        return hist / jnp.sum(hist)
    
    def _compute_entropy(self, distribution: chex.Array) -> float:
        """Compute entropy of a probability distribution"""
        # Avoid log(0) by adding small epsilon
        eps = 1e-10
        return -jnp.sum(distribution * jnp.log(distribution + eps))
    
    def _extract_positions(
        self, 
        observations: List, 
        states: List, 
        partner_idx: int
    ) -> chex.Array:
        """Extract partner positions from observations or states"""
        if states is not None and len(states) > 0:
            # Extract from states if available
            positions = jnp.array([s.agents.position[partner_idx] for s in states])
        else:
            # Try to extract from vector observations
            # In VectorObserver, agent position is at indices [3*num_food : 3*num_food+2]
            positions = []
            for obs in observations:
                agent_view = obs
                # Assume 2 food items: position at indices 6:8
                num_food = 3  # This should match environment config
                pos = agent_view[3*num_food : 3*num_food + 2]
                positions.append(pos)
            positions = jnp.array(positions)
        
        return positions
    
    def _compute_avg_displacement(self, positions: chex.Array) -> float:
        """Compute average Euclidean distance between consecutive positions"""
        if len(positions) < 2:
            return 0.0
        displacements = jnp.linalg.norm(positions[1:] - positions[:-1], axis=1)
        return float(jnp.mean(displacements))
    
    def _compute_grid_coverage(self, positions: chex.Array) -> float:
        """Compute percentage of unique grid cells visited"""
        # Convert positions to unique cell indices
        unique_positions = jnp.unique(positions, axis=0)
        coverage = len(unique_positions) / (self.grid_size ** 2)
        return float(coverage)
    
    def _compute_quadrant_dwell_time(self, positions: chex.Array) -> chex.Array:
        """Compute time spent in each quadrant of the grid"""
        mid = self.grid_size / 2
        quadrants = jnp.zeros(4)
        
        for pos in positions:
            x, y = pos[0], pos[1]
            if x < mid and y < mid:
                quadrant = 0  # Top-left
            elif x >= mid and y < mid:
                quadrant = 1  # Top-right
            elif x < mid and y >= mid:
                quadrant = 2  # Bottom-left
            else:
                quadrant = 3  # Bottom-right
            quadrants = quadrants.at[quadrant].add(1)
        
        # Normalize
        return quadrants / jnp.sum(quadrants)
    
    def _analyze_food_interactions(
        self,
        observations: List,
        actions: List,
        states: List,
        partner_idx: int
    ) -> Dict:
        """Analyze partner's interactions with food items"""
        partner_actions = jnp.array([act[partner_idx] for act in actions])
        load_attempts = jnp.sum(partner_actions == 5)
        
        # Find time to first load attempt
        load_indices = jnp.where(partner_actions == 5, size=len(partner_actions), fill_value=-1)[0]
        time_to_first = int(load_indices[0]) if load_indices[0] != -1 else len(partner_actions)
        
        # Estimate successful loads from reward spikes
        partner_rewards = jnp.array([r[partner_idx] for r in 
                                    ([jnp.zeros(self.num_agents)] + 
                                     [obs for obs in observations])[:-1]])
        successful_loads = int(jnp.sum(partner_rewards > 0))
        
        load_success_rate = (successful_loads / load_attempts 
                           if load_attempts > 0 else 0.0)
        
        # Compute average distance to nearest food
        avg_dist_to_food = self._compute_avg_distance_to_food(observations, states, partner_idx)
        
        return {
            'successful_loads': successful_loads,
            'load_success_rate': float(load_success_rate),
            'time_to_first_load': time_to_first,
            'avg_dist_to_food': avg_dist_to_food
        }
    
    def _compute_avg_distance_to_food(
        self,
        observations: List,
        states: List,
        partner_idx: int
    ) -> float:
        """Compute average distance to nearest visible food"""
        if states is None or len(states) == 0:
            return 0.0
        
        distances = []
        for state in states:
            partner_pos = state.agents.position[partner_idx]
            # Only consider uneaten food
            food_positions = state.food_items.position[~state.food_items.eaten]
            
            if len(food_positions) > 0:
                dists = jnp.linalg.norm(food_positions - partner_pos, axis=1)
                distances.append(jnp.min(dists))
        
        return float(jnp.mean(jnp.array(distances))) if distances else 0.0
    
    def _analyze_cooperation(
        self,
        observations: List,
        actions: List,
        states: List,
        partner_idx: int
    ) -> Dict:
        """Analyze cooperative behaviors between agents"""
        other_idx = 1 - partner_idx
        
        if states is None or len(states) == 0:
            return {
                'avg_dist_to_partner': 0.0,
                'adjacent_to_food': 0,
                'blocked_moves': 0,
                'coordination_events': 0
            }
        
        # Distance to partner
        distances_to_partner = []
        adjacent_to_food_count = 0
        coordination_events = 0
        
        for i, state in enumerate(states):
            partner_pos = state.agents.position[partner_idx]
            other_pos = state.agents.position[other_idx]
            
            dist = jnp.linalg.norm(partner_pos - other_pos)
            distances_to_partner.append(dist)
            
            # Check if both agents are adjacent to same food
            food_positions = state.food_items.position[~state.food_items.eaten]
            for food_pos in food_positions:
                partner_dist = jnp.linalg.norm(partner_pos - food_pos)
                other_dist = jnp.linalg.norm(other_pos - food_pos)
                if partner_dist <= 1.0 and other_dist <= 1.0:
                    adjacent_to_food_count += 1
            
            # Check for coordination (both loading at same time)
            if i < len(actions):
                if actions[i][partner_idx] == 5 and actions[i][other_idx] == 5:
                    coordination_events += 1
        
        # Estimate blocked moves from repeated positions
        partner_positions = jnp.array([s.agents.position[partner_idx] for s in states])
        blocked_moves = int(jnp.sum(
            jnp.all(partner_positions[1:] == partner_positions[:-1], axis=1)
        ))
        
        return {
            'avg_dist_to_partner': float(jnp.mean(jnp.array(distances_to_partner))),
            'adjacent_to_food': adjacent_to_food_count,
            'blocked_moves': blocked_moves,
            'coordination_events': coordination_events
        }
    
    def fingerprint_to_dict(self, fp: BehaviorFingerprint) -> Dict:
        """Convert fingerprint to dictionary for easy serialization"""
        return {
            'action_histogram': fp.action_histogram.tolist(),
            'action_entropy': fp.action_entropy,
            'avg_displacement': fp.avg_displacement,
            'grid_coverage': fp.grid_coverage,
            'num_load_attempts': fp.num_load_attempts,
            'successful_loads': fp.successful_loads,
            'load_success_rate': fp.load_success_rate,
            'time_to_first_load': fp.time_to_first_load,
            'dwell_time_by_quadrant': fp.dwell_time_by_quadrant.tolist(),
            'avg_distance_to_nearest_food': fp.avg_distance_to_nearest_food,
            'avg_distance_to_partner': fp.avg_distance_to_partner,
            'num_adjacent_to_food_with_partner': fp.num_adjacent_to_food_with_partner,
            'num_blocked_moves': fp.num_blocked_moves,
            'coordination_events': fp.coordination_events,
            'cumulative_reward': fp.cumulative_reward,
            'avg_reward_per_step': fp.avg_reward_per_step,
            'steps_recorded': fp.steps_recorded
        }
    
    def describe_fingerprint(self, fp: BehaviorFingerprint) -> str:
        """Generate natural language description of behavior fingerprint"""
        action_names = ['noop', 'up', 'down', 'left', 'right', 'load']
        top_actions = jnp.argsort(fp.action_histogram)[-3:][::-1]
        
        description = f"""Behavior Summary ({fp.steps_recorded} steps):

Movement Pattern:
- Average displacement per step: {fp.avg_displacement:.2f}
- Grid coverage: {fp.grid_coverage*100:.1f}%
- Action diversity (entropy): {fp.action_entropy:.2f}
- Top actions: {', '.join([action_names[i] for i in top_actions])}

Food Interaction:
- Load attempts: {fp.num_load_attempts}
- Successful loads: {fp.successful_loads}
- Load success rate: {fp.load_success_rate*100:.1f}%
- Time to first load: {fp.time_to_first_load} steps
- Avg distance to nearest food: {fp.avg_distance_to_nearest_food:.2f}

Cooperation:
- Avg distance to partner: {fp.avg_distance_to_partner:.2f}
- Times adjacent to food with partner: {fp.num_adjacent_to_food_with_partner}
- Coordination events (simultaneous loads): {fp.coordination_events}
- Blocked moves: {fp.num_blocked_moves}

Performance:
- Cumulative reward: {fp.cumulative_reward:.3f}
- Average reward per step: {fp.avg_reward_per_step:.3f}
"""
        return description

    def describe_for_llm_prompt(self, fp: BehaviorFingerprint) -> str:
        """
        Generate concise, LLM-friendly description of behavior fingerprint.
        Optimized for teammate type classification prompts.
        """
        action_names = ['noop', 'up', 'down', 'left', 'right', 'load']
        
        # Get dominant actions (those with >10% of total)
        significant_actions = []
        for i, prob in enumerate(fp.action_histogram):
            if prob > 0.1:
                significant_actions.append(f"{action_names[i]} ({prob*100:.0f}%)")
        
        # Determine quadrant preference
        quadrant_names = ['top-left', 'top-right', 'bottom-left', 'bottom-right']
        preferred_quadrant = quadrant_names[int(jnp.argmax(fp.dwell_time_by_quadrant))]
        quadrant_time = fp.dwell_time_by_quadrant[jnp.argmax(fp.dwell_time_by_quadrant)]
        
        # Classify cooperation style
        if fp.coordination_events > 2:
            coop_style = "highly coordinated"
        elif fp.avg_distance_to_partner < 2.0:
            coop_style = "stays close to partner"
        elif fp.avg_distance_to_partner > 4.0:
            coop_style = "independent"
        else:
            coop_style = "moderately cooperative"
        
        # Classify food-seeking behavior
        if fp.avg_distance_to_nearest_food < 1.5:
            food_behavior = "stays near food"
        elif fp.time_to_first_load < 5:
            food_behavior = "immediately engages with food"
        elif fp.time_to_first_load > 15:
            food_behavior = "explores before engaging food"
        else:
            food_behavior = "moderately food-focused"
        
        description = f"""Observed Behavior (first {fp.steps_recorded} steps):

Actions: {', '.join(significant_actions)}. Action diversity: {fp.action_entropy:.2f}.

Movement: Covers {fp.grid_coverage*100:.0f}% of grid, average displacement {fp.avg_displacement:.2f} per step. Spends {quadrant_time*100:.0f}% of time in {preferred_quadrant} quadrant.

Food Strategy: {fp.num_load_attempts} load attempts with {fp.load_success_rate*100:.0f}% success rate. First load at step {fp.time_to_first_load}. Maintains average {fp.avg_distance_to_nearest_food:.1f} distance to nearest food. Classification: {food_behavior}.

Cooperation: Maintains {fp.avg_distance_to_partner:.1f} average distance from partner. {fp.coordination_events} simultaneous load events, {fp.num_adjacent_to_food_with_partner} times both agents near same food. {fp.num_blocked_moves} blocked movements. Style: {coop_style}.

Performance: {fp.cumulative_reward:.3f} cumulative reward, {fp.avg_reward_per_step:.3f} per step."""
        
        return description


# Example usage
if __name__ == "__main__":
    # Mock data for demonstration
    extractor = LBFBehaviorExtractor(grid_size=8, num_agents=2)
    
    # Simulate some trajectory data
    num_steps = 20
    mock_observations = [jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])] * num_steps  # Would be actual observations
    mock_actions = [jnp.array([1, 2])] * num_steps  # Random actions
    mock_rewards = [jnp.array([0.0, 0.0])] * num_steps
    
    # Extract fingerprint
    fingerprint = extractor.extract_fingerprint(
        observations=mock_observations,
        actions=mock_actions,
        rewards=mock_rewards,
        partner_idx=0,
        probe_length=20
    )
    
    print(extractor.describe_fingerprint(fingerprint))
    print("Behavior extractor initialized successfully!")