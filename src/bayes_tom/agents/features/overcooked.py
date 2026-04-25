"""
Behavior Fingerprint Extraction for Overcooked Environment (JaxMARL)
Inspired by ReCoLLAB paper's approach for teammate modeling
"""

import jax
import jax.numpy as jnp
import chex
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import numpy as np


# Constants from Overcooked environment
OBJECT_TO_INDEX = {
    "unseen": 0,
    "empty": 1,
    "wall": 2,
    "onion": 3,
    "onion_pile": 4,
    "plate": 5,
    "plate_pile": 6,
    "goal": 7,
    "pot": 8,
    "dish": 9,
    "agent": 10,
}

ACTION_NAMES = ['up', 'down', 'right', 'left', 'stay', 'interact']
POT_EMPTY_STATUS = 23
POT_FULL_STATUS = 20
POT_READY_STATUS = 0


@dataclass
class BehaviorFingerprint:
    """Container for extracted behavioral features from Overcooked trajectories"""
    
    # Action statistics
    action_histogram: chex.Array  # Shape: (6,) - counts of each action type
    action_entropy: float  # Entropy of action distribution
    interact_frequency: float  # Percentage of interact actions
    movement_frequency: float  # Percentage of movement actions
    
    # Movement patterns
    avg_displacement: float  # Average distance moved per step
    grid_coverage: float  # Percentage of unique positions visited
    
    # Station interaction patterns
    num_onion_pickups: int  # Estimated pickups from onion pile
    num_plate_pickups: int  # Estimated pickups from plate pile
    num_pot_interactions: int  # Interactions with pots
    num_deliveries: int  # Successful soup deliveries
    
    # Spatial behavior (dwell time at key locations)
    dwell_time_near_pot: float  # Time spent adjacent to pot
    dwell_time_near_onion_pile: float  # Time spent at onion pile
    dwell_time_near_plate_pile: float  # Time spent at plate pile
    dwell_time_near_goal: float  # Time spent at delivery station
    
    # Cooperation indicators
    avg_distance_to_partner: float  # Average distance to teammate
    num_collisions: int  # Times agents collided or blocked each other
    num_handoffs: int  # Estimated item handoffs (counter exchanges)
    coordination_score: float  # Derived coordination metric
    
    # Timing features
    time_to_first_interact: int  # Steps until first interaction
    time_to_first_pot_interact: int  # Steps until first pot interaction
    time_to_first_delivery: int  # Steps until first delivery
    
    # Performance metrics
    cumulative_reward: float  # Total reward accumulated
    cumulative_shaped_reward: float  # Total shaped reward
    avg_reward_per_step: float  # Average reward per timestep
    
    # Item holding patterns
    time_holding_onion: int  # Steps spent holding an onion
    time_holding_plate: int  # Steps spent holding a plate
    time_holding_dish: int  # Steps spent holding a soup
    time_holding_nothing: int  # Steps spent with empty inventory
    
    # Temporal features
    steps_recorded: int  # Number of steps in this fingerprint


class OvercookedBehaviorExtractor:
    """Extracts behavior fingerprints from Overcooked environment trajectories"""
    
    def __init__(self, layout_height: int = 4, layout_width: int = 5):
        self.layout_height = layout_height
        self.layout_width = layout_width
        self.num_actions = 6  # up, down, right, left, stay, interact
        self.num_agents = 2
        
    def extract_fingerprint(
        self,
        observations: List[chex.Array],
        actions: List[chex.Array],
        rewards: List[chex.Array],
        partner_idx: int = 1,
        probe_length: Optional[int] = None
    ) -> BehaviorFingerprint:
        """
        Extract behavior fingerprint for a partner agent from trajectory data.
        
        Args:
            observations: List of observation dicts from environment
            actions: List of action dicts {agent_id: action}
            rewards: List of reward dicts {agent_id: reward}
            infos: List of info dicts containing shaped rewards
            partner_id: ID of the partner agent to analyze ("agent_0" or "agent_1")
            probe_length: If provided, only analyze first N steps
            
        Returns:
            BehaviorFingerprint object containing extracted features
        """
        if probe_length is not None:
            observations = observations[:probe_length]
            actions = actions[:probe_length]
            rewards = rewards[:probe_length]
            # if infos is not None:
            #     infos = infos[:probe_length]
        
        observations = [obs.reshape(self.layout_height, self.layout_width, -1) for obs in observations]

        num_steps = len(actions)
        other_idx = int(partner_idx + 1) % self.num_agents
        
        # Extract partner actions
        partner_actions = jnp.array([act[partner_idx] for act in actions])

        # Action statistics
        action_histogram = self._compute_action_histogram(partner_actions)
        action_entropy = self._compute_entropy(action_histogram)
        interact_freq = float(action_histogram[5])  # interact is action 5
        movement_freq = float(np.sum(action_histogram[:4]))  # up/down/right/left
        
        # Movement and spatial features
        positions = self._extract_positions(observations, partner_idx)
        avg_displacement = self._compute_avg_displacement(positions)
        grid_coverage = self._compute_grid_coverage(positions)
        
        # Station locations and dwell times
        station_features = self._analyze_station_interactions(
            observations, positions, partner_idx
        )
        
        # Item interaction features
        interaction_features = self._analyze_interactions(
            observations, actions, rewards, partner_idx
        )
        
        # Cooperation features
        coop_features = self._analyze_cooperation(
            observations, actions, partner_idx, other_idx
        )
        
        # Inventory analysis
        inventory_features = self._analyze_inventory(observations, partner_idx)
        
        # Performance metrics
        partner_rewards = np.array([r[partner_idx] for r in rewards])
        cumulative_reward = float(np.sum(partner_rewards))
        avg_reward = float(np.mean(partner_rewards))
        
        cumulative_shaped_reward = 0.0
        # if infos is not None and len(infos) > 0:
        #     shaped_rewards = [info.get('shaped_reward', {}).get(partner_id, 0.0) 
        #                     for info in infos]
        #     cumulative_shaped_reward = float(np.sum(shaped_rewards))
        
        return BehaviorFingerprint(
            action_histogram=action_histogram,
            action_entropy=float(action_entropy),
            interact_frequency=interact_freq,
            movement_frequency=movement_freq,
            avg_displacement=float(avg_displacement),
            grid_coverage=float(grid_coverage),
            num_onion_pickups=interaction_features['onion_pickups'],
            num_plate_pickups=interaction_features['plate_pickups'],
            num_pot_interactions=interaction_features['pot_interactions'],
            num_deliveries=interaction_features['deliveries'],
            dwell_time_near_pot=station_features['pot_dwell'],
            dwell_time_near_onion_pile=station_features['onion_pile_dwell'],
            dwell_time_near_plate_pile=station_features['plate_pile_dwell'],
            dwell_time_near_goal=station_features['goal_dwell'],
            avg_distance_to_partner=coop_features['avg_dist_to_partner'],
            num_collisions=coop_features['collisions'],
            num_handoffs=coop_features['handoffs'],
            coordination_score=coop_features['coordination_score'],
            time_to_first_interact=interaction_features['time_to_first_interact'],
            time_to_first_pot_interact=interaction_features['time_to_first_pot_interact'],
            time_to_first_delivery=interaction_features['time_to_first_delivery'],
            cumulative_reward=cumulative_reward,
            cumulative_shaped_reward=cumulative_shaped_reward,
            avg_reward_per_step=avg_reward,
            time_holding_onion=inventory_features['holding_onion'],
            time_holding_plate=inventory_features['holding_plate'],
            time_holding_dish=inventory_features['holding_dish'],
            time_holding_nothing=inventory_features['holding_nothing'],
            steps_recorded=num_steps
        )
    
    def _compute_action_histogram(self, actions: np.ndarray) -> np.ndarray:
        """Compute normalized histogram of actions"""
        hist = np.bincount(actions, minlength=self.num_actions)
        return hist / np.sum(hist)
    
    def _compute_entropy(self, distribution: np.ndarray) -> float:
        """Compute entropy of a probability distribution"""
        eps = 1e-10
        return -np.sum(distribution * np.log(distribution + eps))
    
    def _extract_positions(
        self, 
        observations: List[Dict[str, chex.Array]], 
        agent_idx: int
    ) -> np.ndarray:
        """
        Extract agent positions from observations.
        Position is encoded in channel 0 for agent_0, channel 1 for agent_1.
        """
        positions = []
        channel_idx = 0 if agent_idx == 0 else 1
        
        for obs in observations:
            # agent_obs = obs[agent_idx]  # Shape: (height, width, 26)
            agent_layer = obs[:, :, channel_idx]  # Get agent position layer
            
            # Find where the agent is (value = 1)
            pos = np.argwhere(agent_layer == 1)
            if len(pos) > 0:
                positions.append(pos[0])  # [y, x]
            else:
                # Fallback: use previous position or center
                if positions:
                    positions.append(positions[-1])
                else:
                    positions.append([self.layout_height // 2, self.layout_width // 2])
        
        return np.array(positions)
    
    def _compute_avg_displacement(self, positions: np.ndarray) -> float:
        """Compute average Euclidean distance between consecutive positions"""
        if len(positions) < 2:
            return 0.0
        displacements = np.linalg.norm(positions[1:] - positions[:-1], axis=1)
        return float(np.mean(displacements))
    
    def _compute_grid_coverage(self, positions: np.ndarray) -> float:
        """Compute percentage of unique grid cells visited"""
        unique_positions = np.unique(positions, axis=0)
        total_cells = self.layout_height * self.layout_width
        coverage = len(unique_positions) / total_cells
        return float(coverage)
    
    def _analyze_station_interactions(
        self,
        observations: List[Dict[str, chex.Array]],
        positions: np.ndarray,
        agent_idx: int
    ) -> Dict:
        """Analyze time spent near key stations"""
        
        # Extract station locations from first observation
        first_obs = observations[0]
        
        # Channel 10: pot locations
        # Channel 12: onion pile locations  
        # Channel 14: plate pile locations
        # Channel 15: goal (delivery) locations
        pot_locs = np.argwhere(first_obs[:, :, 10] == 1)
        onion_locs = np.argwhere(first_obs[:, :, 12] == 1)
        plate_locs = np.argwhere(first_obs[:, :, 14] == 1)
        goal_locs = np.argwhere(first_obs[:, :, 15] == 1)
        
        def is_adjacent(pos, target_locs, distance=1.5):
            """Check if position is adjacent to any target location"""
            if len(target_locs) == 0:
                return False
            dists = np.linalg.norm(target_locs - pos, axis=1)
            return np.any(dists <= distance)
        
        pot_dwell = sum(is_adjacent(pos, pot_locs) for pos in positions)
        onion_dwell = sum(is_adjacent(pos, onion_locs) for pos in positions)
        plate_dwell = sum(is_adjacent(pos, plate_locs) for pos in positions)
        goal_dwell = sum(is_adjacent(pos, goal_locs) for pos in positions)
        
        total = len(positions)
        
        return {
            'pot_dwell': pot_dwell / total if total > 0 else 0.0,
            'onion_pile_dwell': onion_dwell / total if total > 0 else 0.0,
            'plate_pile_dwell': plate_dwell / total if total > 0 else 0.0,
            'goal_dwell': goal_dwell / total if total > 0 else 0.0,
        }
    
    def _analyze_interactions(
        self,
        observations: List[chex.Array],
        actions: List[chex.Array],
        rewards: List[chex.Array],
        agent_idx: int
    ) -> Dict:
        """Analyze interactions with objects (pickups, pot interactions, deliveries)"""
        
        partner_actions = [act[agent_idx] for act in actions]
        interact_actions = [act == 5 for act in partner_actions]  # action 5 is interact
        
        # Count pot interactions (when adjacent to pot and interacting)
        pot_interactions = 0
        first_pot_interact = len(partner_actions)
        
        for i, obs in enumerate(observations):
            if i >= len(interact_actions):
                break
            if not interact_actions[i]:
                continue
                
            agent_obs = obs
            pot_layer = agent_obs[:, :, 10]  # Channel 10: pot locations
            agent_channel = 0 if agent_idx == 0 else 1
            agent_layer = agent_obs[:, :, agent_channel]
            
            # Check if agent is adjacent to pot
            agent_pos = np.argwhere(agent_layer == 1)
            pot_pos = np.argwhere(pot_layer == 1)
            
            if len(agent_pos) > 0 and len(pot_pos) > 0:
                for pp in pot_pos:
                    dist = np.linalg.norm(agent_pos[0] - pp)
                    if dist <= 1.5:
                        pot_interactions += 1
                        if first_pot_interact == len(partner_actions):
                            first_pot_interact = i
                        break
        
        # Estimate pickups from shaped rewards if available
        onion_pickups = 0
        plate_pickups = 0
        
        # Count deliveries from sparse rewards (delivery = +20 reward)
        deliveries = sum(1 for r in rewards if r[agent_idx] >= 15)
        first_delivery = next((i for i, r in enumerate(rewards) 
                              if r[agent_idx] >= 15), len(rewards))
        
        # Time to first interact
        first_interact = next((i for i, act in enumerate(partner_actions) 
                              if act == 5), len(partner_actions))
        
        return {
            'onion_pickups': onion_pickups,
            'plate_pickups': plate_pickups,
            'pot_interactions': pot_interactions,
            'deliveries': deliveries,
            'time_to_first_interact': first_interact,
            'time_to_first_pot_interact': first_pot_interact,
            'time_to_first_delivery': first_delivery,
        }
    
    def _analyze_cooperation(
        self,
        observations: List[Dict[str, chex.Array]],
        actions: List[Dict[str, int]],
        partner_idx: int,
        other_idx: int
    ) -> Dict:
        """Analyze cooperative behaviors between agents"""
        
        partner_positions = self._extract_positions(observations, partner_idx)
        other_positions = self._extract_positions(observations, other_idx)
        
        # Average distance to partner
        distances = np.linalg.norm(partner_positions - other_positions, axis=1)
        avg_distance = float(np.mean(distances))
        
        # Count collisions (both agents in same position)
        collisions = sum(1 for pp, op in zip(partner_positions, other_positions)
                        if np.all(pp == op))
        
        # Estimate handoffs (both agents adjacent and one has interact action)
        handoffs = 0
        for i, (pp, op) in enumerate(zip(partner_positions, other_positions)):
            if i >= len(actions):
                break
            dist = np.linalg.norm(pp - op)
            if dist <= 1.5:  # Adjacent
                if actions[i][partner_idx] == 5 or actions[i][other_idx] == 5:
                    handoffs += 1
        
        # Coordination score: inverse of average distance + interaction alignment
        # Higher when agents are close and taking complementary actions
        coordination_score = 1.0 / (1.0 + avg_distance)
        
        return {
            'avg_dist_to_partner': avg_distance,
            'collisions': collisions,
            'handoffs': handoffs,
            'coordination_score': coordination_score,
        }
    
    def _analyze_inventory(
        self,
        observations: List[chex.Array],
        agent_idx: int
    ) -> Dict:
        """Analyze what items the agent is holding over time"""
        
        holding_onion = 0
        holding_plate = 0
        holding_dish = 0
        holding_nothing = 0
        
        agent_channel = 0 if agent_idx == 0 else 1
        
        for obs in observations:
            agent_obs = obs
            agent_layer = agent_obs[:, :, agent_channel]
            
            # Find agent position
            agent_pos = np.argwhere(agent_layer == 1)
            if len(agent_pos) == 0:
                holding_nothing += 1
                continue
            
            y, x = agent_pos[0]
            
            # Check what's at agent position in various item layers
            # Channel 22: plate locations
            # Channel 23: onion locations
            # Channel 21: soup ready layer (includes held dishes)
            
            has_plate = agent_obs[y, x, 22] > 0
            has_onion = agent_obs[y, x, 23] > 0
            has_dish = agent_obs[y, x, 21] > 0 and agent_obs[y, x, 10] == 0  # soup but not in pot
            
            if has_dish:
                holding_dish += 1
            elif has_plate:
                holding_plate += 1
            elif has_onion:
                holding_onion += 1
            else:
                holding_nothing += 1
        
        return {
            'holding_onion': holding_onion,
            'holding_plate': holding_plate,
            'holding_dish': holding_dish,
            'holding_nothing': holding_nothing,
        }
    
    def fingerprint_to_dict(self, fp: BehaviorFingerprint) -> Dict:
        """Convert fingerprint to dictionary for easy serialization"""
        return {
            'action_histogram': fp.action_histogram.tolist(),
            'action_entropy': fp.action_entropy,
            'interact_frequency': fp.interact_frequency,
            'movement_frequency': fp.movement_frequency,
            'avg_displacement': fp.avg_displacement,
            'grid_coverage': fp.grid_coverage,
            'num_onion_pickups': fp.num_onion_pickups,
            'num_plate_pickups': fp.num_plate_pickups,
            'num_pot_interactions': fp.num_pot_interactions,
            'num_deliveries': fp.num_deliveries,
            'dwell_time_near_pot': fp.dwell_time_near_pot,
            'dwell_time_near_onion_pile': fp.dwell_time_near_onion_pile,
            'dwell_time_near_plate_pile': fp.dwell_time_near_plate_pile,
            'dwell_time_near_goal': fp.dwell_time_near_goal,
            'avg_distance_to_partner': fp.avg_distance_to_partner,
            'num_collisions': fp.num_collisions,
            'num_handoffs': fp.num_handoffs,
            'coordination_score': fp.coordination_score,
            'time_to_first_interact': fp.time_to_first_interact,
            'time_to_first_pot_interact': fp.time_to_first_pot_interact,
            'time_to_first_delivery': fp.time_to_first_delivery,
            'cumulative_reward': fp.cumulative_reward,
            'cumulative_shaped_reward': fp.cumulative_shaped_reward,
            'avg_reward_per_step': fp.avg_reward_per_step,
            'time_holding_onion': fp.time_holding_onion,
            'time_holding_plate': fp.time_holding_plate,
            'time_holding_dish': fp.time_holding_dish,
            'time_holding_nothing': fp.time_holding_nothing,
            'steps_recorded': fp.steps_recorded
        }
    
    def describe_fingerprint(self, fp: BehaviorFingerprint) -> str:
        """
        Generate concise, LLM-friendly description of behavior fingerprint.
        Optimized for teammate type classification prompts.
        """
        
        # Get dominant actions
        action_names = ACTION_NAMES
        significant_actions = []
        for i, prob in enumerate(fp.action_histogram):
            if prob > 0.1:
                significant_actions.append(f"{action_names[i]} ({prob*100:.0f}%)")
        
        # Classify role based on dwell times
        dwell_scores = {
            'pot-focused': fp.dwell_time_near_pot,
            'onion-gatherer': fp.dwell_time_near_onion_pile,
            'plate-gatherer': fp.dwell_time_near_plate_pile,
            'server': fp.dwell_time_near_goal
        }
        primary_role = max(dwell_scores.items(), key=lambda x: x[1])[0]
        
        # Classify cooperation style
        if fp.avg_distance_to_partner < 1.5:
            coop_style = "very close coordination"
        elif fp.avg_distance_to_partner < 3.0:
            coop_style = "moderate coordination"
        else:
            coop_style = "independent work"
        
        # Item holding pattern
        total_holding = (fp.time_holding_onion + fp.time_holding_plate + 
                        fp.time_holding_dish + fp.time_holding_nothing)
        if total_holding > 0:
            onion_pct = (fp.time_holding_onion / total_holding) * 100
            plate_pct = (fp.time_holding_plate / total_holding) * 100
            dish_pct = (fp.time_holding_dish / total_holding) * 100
            empty_pct = (fp.time_holding_nothing / total_holding) * 100
        else:
            onion_pct = plate_pct = dish_pct = empty_pct = 0
        
        description = f"""Observed Behavior (first {fp.steps_recorded} steps):

Actions: {', '.join(significant_actions)}. Interact rate: {fp.interact_frequency*100:.0f}%. Action diversity: {fp.action_entropy:.2f}.

Movement: Covers {fp.grid_coverage*100:.0f}% of grid, average displacement {fp.avg_displacement:.2f} per step.

Station Focus: Spends {fp.dwell_time_near_pot*100:.0f}% near pot, {fp.dwell_time_near_onion_pile*100:.0f}% near onions, {fp.dwell_time_near_plate_pile*100:.0f}% near plates, {fp.dwell_time_near_goal*100:.0f}% near delivery. Primary role: {primary_role}.

Task Execution: {fp.num_pot_interactions} pot interactions, {fp.num_deliveries} deliveries. First interact at step {fp.time_to_first_interact}, first pot interaction at step {fp.time_to_first_pot_interact}.

Inventory Usage: Holds onions {onion_pct:.0f}% of time, plates {plate_pct:.0f}%, soups {dish_pct:.0f}%, empty {empty_pct:.0f}%.

Cooperation: Maintains {fp.avg_distance_to_partner:.1f} average distance from partner. {fp.num_handoffs} potential handoffs, {fp.num_collisions} collisions. Coordination score: {fp.coordination_score:.2f}. Style: {coop_style}.

Performance: {fp.cumulative_reward:.1f} reward, {fp.cumulative_shaped_reward:.1f} shaped reward, {fp.avg_reward_per_step:.3f} per step."""
        
        return description


# Example usage
if __name__ == "__main__":
    # Mock data for demonstration
    extractor = OvercookedBehaviorExtractor(layout_height=5, layout_width=5)
    
    print("Overcooked Behavior Extractor initialized successfully!")
    print(f"Supports {extractor.num_actions} actions: {ACTION_NAMES}")
    print("\nTo use:")
    print("1. Collect trajectory: observations, actions, rewards, infos")
    print("2. Call extractor.extract_fingerprint(...)")
    print("3. Get LLM description: extractor.describe_for_llm_prompt(fingerprint)")