"""
Behavior Fingerprint Extraction for Hanabi
Inspired by ReCoLLAB paper's approach for teammate modeling
"""

import jax
import jax.numpy as jnp
import chex
from typing import Dict, List, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class HanabiBehaviorFingerprint:
    """Container for extracted behavioral features from Hanabi trajectories"""
    
    # Action statistics
    action_histogram: chex.Array  # Normalized counts of each action type
    action_type_distribution: Dict[str, float]  # Play, discard, hint_color, hint_rank ratios
    action_entropy: float  # Entropy of action distribution
    
    # Play behavior
    num_plays: int  # Number of play attempts
    num_successful_plays: int  # Number of cards that scored
    play_success_rate: float  # successful_plays / num_plays
    time_to_first_play: int  # Steps until first play attempt
    avg_play_position: float  # Which card position they tend to play (0-4)
    
    # Discard behavior
    num_discards: int  # Number of discard actions
    avg_discard_position: float  # Which card position they tend to discard
    discards_when_info_full: int  # Times discarded when info tokens were full
    
    # Hinting behavior
    num_color_hints: int  # Number of color hints given
    num_rank_hints: int  # Number of rank hints given
    hint_ratio: float  # color_hints / (color_hints + rank_hints)
    avg_cards_revealed_per_hint: float  # Average information efficiency
    time_to_first_hint: int  # Steps until first hint given
    
    # Cooperation patterns
    hints_received: int  # Number of hints received from partners
    plays_after_hint: int  # Number of plays made shortly after receiving hint
    hint_responsiveness: float  # plays_after_hint / hints_received
    
    # Risk behavior
    risky_plays: int  # Plays made without full information
    conservative_plays: int  # Plays made with strong hints
    risk_tolerance: float  # risky_plays / total_plays
    
    # Information management
    avg_info_tokens_when_hinting: float  # Avg info tokens when giving hints
    avg_info_tokens_when_playing: float  # Avg info tokens when playing
    info_tokens_depleted_count: int  # Times info tokens reached 0
    
    # Card management
    oldest_card_plays: int  # Times they played their oldest card
    newest_card_plays: int  # Times they played their newest card
    card_position_bias: float  # Tendency toward left (-1) or right (1) of hand
    
    # Performance metrics
    cumulative_reward: float  # Total reward accumulated
    avg_reward_per_action: float  # Average reward per action taken
    bombs_caused: int  # Number of times they lost a life token
    
    # Temporal features
    steps_recorded: int  # Number of steps in this fingerprint
    actions_taken: int  # Number of turns they had


class HanabiBehaviorExtractor:
    """Extracts behavior fingerprints from Hanabi environment trajectories"""
    
    def __init__(self, num_agents: int = 2, hand_size: int = 5, num_colors: int = 5, num_ranks: int = 5):
        self.num_agents = num_agents
        self.hand_size = hand_size
        self.num_colors = num_colors
        self.num_ranks = num_ranks
        
        # Calculate action space size
        self.num_moves = (
            hand_size * 2 +  # discard + play
            (num_agents - 1) * (num_colors + num_ranks) +  # hints
            1  # noop
        )
        
        # Action ranges
        self.discard_range = range(0, hand_size)
        self.play_range = range(hand_size, 2 * hand_size)
        self.color_hint_range = range(
            2 * hand_size,
            2 * hand_size + (num_agents - 1) * num_colors
        )
        self.rank_hint_range = range(
            2 * hand_size + (num_agents - 1) * num_colors,
            2 * hand_size + (num_agents - 1) * (num_colors + num_ranks)
        )
        
    def extract_fingerprint(
        self,
        observations: List[Dict],
        actions: List[chex.Array],
        rewards: List[chex.Array],
        states: List = None,
        partner_idx: int = 1,
        probe_length: int = None
    ) -> HanabiBehaviorFingerprint:
        """
        Extract behavior fingerprint for a partner agent from trajectory data.
        
        Args:
            observations: List of observation dicts from environment
            actions: List of action dicts (one action per agent per step)
            rewards: List of reward dicts
            states: List of State objects (optional, for detailed analysis)
            partner_idx: Index of the partner agent to analyze
            probe_length: If provided, only analyze first N steps
            
        Returns:
            HanabiBehaviorFingerprint object containing extracted features
        """
        if probe_length is not None:
            observations = observations[:probe_length]
            actions = actions[:probe_length]
            rewards = rewards[:probe_length]
            if states is not None:
                states = states[:probe_length]
        
        num_steps = len(actions)
        
        # # Get partner's agent key
        # agent_keys = list(actions[0].keys())
        # partner_key = agent_keys[partner_idx]
        
        # # Extract partner actions (only when they're the current player)
        # partner_actions = []
        # partner_action_steps = []
        # for step_idx, (action_dict, state) in enumerate(zip(actions, states if states else [None]*len(actions))):
        #     if state is not None:
        #         current_player = int(jnp.nonzero(state.cur_player_idx, size=1)[0][0])
        #         if current_player == partner_idx:
        #             partner_actions.append(action_dict[partner_key])
        #             partner_action_steps.append(step_idx)
        #     else:
        #         # If no state, assume round-robin turns
        #         if step_idx % self.num_agents == partner_idx:
        #             partner_actions.append(action_dict[partner_key])
        #             partner_action_steps.append(step_idx)
        
        # if len(partner_actions) == 0:
        #     # Return empty fingerprint if no actions
        #     return self._empty_fingerprint(num_steps)
        
        # partner_actions = jnp.array(partner_actions)
        
        # Extract partner actions
        partner_actions = jnp.array([act[partner_idx] for act in actions])
        partner_action_steps = [i for i in range(num_steps)]

        # Action statistics
        action_histogram = self._compute_action_histogram(partner_actions)
        action_type_dist = self._compute_action_type_distribution(partner_actions)
        action_entropy = self._compute_entropy(action_histogram)
        
        # Play behavior
        play_features = self._analyze_play_behavior(partner_actions, states, partner_idx, partner_action_steps)
        
        # Discard behavior
        discard_features = self._analyze_discard_behavior(partner_actions, states, partner_idx, partner_action_steps)
        
        # Hinting behavior
        hint_features = self._analyze_hint_behavior(partner_actions, states, partner_idx, partner_action_steps)
        
        # Cooperation features
        coop_features = self._analyze_cooperation(observations, actions, states, partner_idx, partner_action_steps)
        
        # Risk behavior
        risk_features = self._analyze_risk_behavior(partner_actions, states, partner_idx, partner_action_steps)
        
        # Information management
        info_features = self._analyze_info_management(partner_actions, states, partner_idx, partner_action_steps)
        
        # Card position preferences
        card_features = self._analyze_card_positions(partner_actions)
        
        # Performance metrics
        partner_rewards = self._extract_partner_rewards(rewards, partner_idx, partner_action_steps)
        cumulative_reward = float(jnp.sum(partner_rewards))
        avg_reward = float(jnp.mean(partner_rewards)) if len(partner_rewards) > 0 else 0.0
        
        bombs = self._count_bombs(states, partner_idx, partner_action_steps) if states else 0
        
        return HanabiBehaviorFingerprint(
            action_histogram=action_histogram,
            action_type_distribution=action_type_dist,
            action_entropy=float(action_entropy),
            num_plays=play_features['num_plays'],
            num_successful_plays=play_features['successful_plays'],
            play_success_rate=play_features['success_rate'],
            time_to_first_play=play_features['time_to_first'],
            avg_play_position=play_features['avg_position'],
            num_discards=discard_features['num_discards'],
            avg_discard_position=discard_features['avg_position'],
            discards_when_info_full=discard_features['discards_when_full'],
            num_color_hints=hint_features['color_hints'],
            num_rank_hints=hint_features['rank_hints'],
            hint_ratio=hint_features['hint_ratio'],
            avg_cards_revealed_per_hint=hint_features['avg_cards_revealed'],
            time_to_first_hint=hint_features['time_to_first'],
            hints_received=coop_features['hints_received'],
            plays_after_hint=coop_features['plays_after_hint'],
            hint_responsiveness=coop_features['responsiveness'],
            risky_plays=risk_features['risky_plays'],
            conservative_plays=risk_features['conservative_plays'],
            risk_tolerance=risk_features['risk_tolerance'],
            avg_info_tokens_when_hinting=info_features['avg_info_when_hint'],
            avg_info_tokens_when_playing=info_features['avg_info_when_play'],
            info_tokens_depleted_count=info_features['depleted_count'],
            oldest_card_plays=card_features['oldest_plays'],
            newest_card_plays=card_features['newest_plays'],
            card_position_bias=card_features['position_bias'],
            cumulative_reward=cumulative_reward,
            avg_reward_per_action=avg_reward,
            bombs_caused=bombs,
            steps_recorded=num_steps,
            actions_taken=len(partner_actions)
        )
    
    def _empty_fingerprint(self, num_steps: int) -> HanabiBehaviorFingerprint:
        """Return an empty fingerprint when no actions are available"""
        return HanabiBehaviorFingerprint(
            action_histogram=jnp.zeros(self.num_moves),
            action_type_distribution={'play': 0, 'discard': 0, 'hint_color': 0, 'hint_rank': 0},
            action_entropy=0.0,
            num_plays=0, num_successful_plays=0, play_success_rate=0.0,
            time_to_first_play=num_steps, avg_play_position=0.0,
            num_discards=0, avg_discard_position=0.0, discards_when_info_full=0,
            num_color_hints=0, num_rank_hints=0, hint_ratio=0.0,
            avg_cards_revealed_per_hint=0.0, time_to_first_hint=num_steps,
            hints_received=0, plays_after_hint=0, hint_responsiveness=0.0,
            risky_plays=0, conservative_plays=0, risk_tolerance=0.0,
            avg_info_tokens_when_hinting=0.0, avg_info_tokens_when_playing=0.0,
            info_tokens_depleted_count=0,
            oldest_card_plays=0, newest_card_plays=0, card_position_bias=0.0,
            cumulative_reward=0.0, avg_reward_per_action=0.0, bombs_caused=0,
            steps_recorded=num_steps, actions_taken=0
        )
    
    def _compute_action_histogram(self, actions: chex.Array) -> chex.Array:
        """Compute normalized histogram of actions"""
        hist = jnp.bincount(actions, length=self.num_moves)
        return hist / jnp.sum(hist) if jnp.sum(hist) > 0 else hist
    
    def _compute_action_type_distribution(self, actions: chex.Array) -> Dict[str, float]:
        """Compute distribution over action types"""
        total = len(actions)
        if total == 0:
            return {'play': 0, 'discard': 0, 'hint_color': 0, 'hint_rank': 0}
        
        plays = jnp.sum(jnp.isin(actions, jnp.array(list(self.play_range))))
        discards = jnp.sum(jnp.isin(actions, jnp.array(list(self.discard_range))))
        color_hints = jnp.sum(jnp.isin(actions, jnp.array(list(self.color_hint_range))))
        rank_hints = jnp.sum(jnp.isin(actions, jnp.array(list(self.rank_hint_range))))
        
        return {
            'play': float(plays / total),
            'discard': float(discards / total),
            'hint_color': float(color_hints / total),
            'hint_rank': float(rank_hints / total)
        }
    
    def _compute_entropy(self, distribution: chex.Array) -> float:
        """Compute entropy of a probability distribution"""
        eps = 1e-10
        return float(-jnp.sum(distribution * jnp.log(distribution + eps)))
    
    def _analyze_play_behavior(
        self,
        actions: chex.Array,
        states: List,
        partner_idx: int,
        action_steps: List[int]
    ) -> Dict:
        """Analyze play action patterns"""
        play_mask = jnp.isin(actions, jnp.array(list(self.play_range)))
        num_plays = int(jnp.sum(play_mask))
        
        # Find first play
        play_indices = jnp.where(play_mask, size=len(actions), fill_value=-1)[0]
        first_play_step = int(action_steps[play_indices[0]]) if play_indices[0] != -1 else len(action_steps)
        
        # Count successful plays (those that scored)
        successful_plays = 0
        if states is not None:
            for i, (action, step) in enumerate(zip(actions, action_steps)):
                if action in self.play_range and i < len(states) - 1:
                    # Check if score increased
                    if states[step + 1].score > states[step].score:
                        successful_plays += 1
        
        success_rate = successful_plays / num_plays if num_plays > 0 else 0.0
        
        # Average play position
        play_actions = actions[play_mask]
        if len(play_actions) > 0:
            play_positions = play_actions - self.hand_size
            avg_position = float(jnp.mean(play_positions))
        else:
            avg_position = 0.0
        
        return {
            'num_plays': num_plays,
            'successful_plays': successful_plays,
            'success_rate': float(success_rate),
            'time_to_first': first_play_step,
            'avg_position': avg_position
        }
    
    def _analyze_discard_behavior(
        self,
        actions: chex.Array,
        states: List,
        partner_idx: int,
        action_steps: List[int]
    ) -> Dict:
        """Analyze discard action patterns"""
        discard_mask = jnp.isin(actions, jnp.array(list(self.discard_range)))
        num_discards = int(jnp.sum(discard_mask))
        
        # Average discard position
        discard_actions = actions[discard_mask]
        avg_position = float(jnp.mean(discard_actions)) if len(discard_actions) > 0 else 0.0
        
        # Count discards when info tokens were full
        discards_when_full = 0
        if states is not None:
            for action, step in zip(actions, action_steps):
                if action in self.discard_range:
                    info_tokens = jnp.sum(states[step].info_tokens)
                    if info_tokens >= 8:  # max_info_tokens
                        discards_when_full += 1
        
        return {
            'num_discards': num_discards,
            'avg_position': avg_position,
            'discards_when_full': discards_when_full
        }
    
    def _analyze_hint_behavior(
        self,
        actions: chex.Array,
        states: List,
        partner_idx: int,
        action_steps: List[int]
    ) -> Dict:
        """Analyze hinting patterns"""
        color_hint_mask = jnp.isin(actions, jnp.array(list(self.color_hint_range)))
        rank_hint_mask = jnp.isin(actions, jnp.array(list(self.rank_hint_range)))
        
        num_color_hints = int(jnp.sum(color_hint_mask))
        num_rank_hints = int(jnp.sum(rank_hint_mask))
        
        total_hints = num_color_hints + num_rank_hints
        hint_ratio = num_color_hints / total_hints if total_hints > 0 else 0.5
        
        # Find first hint
        hint_mask = color_hint_mask | rank_hint_mask
        hint_indices = jnp.where(hint_mask, size=len(actions), fill_value=-1)[0]
        first_hint_step = int(action_steps[hint_indices[0]]) if hint_indices[0] != -1 else len(action_steps)
        
        # Average cards revealed per hint (would need state inspection for exact count)
        # For now, use a rough estimate
        avg_cards_revealed = 2.0  # Placeholder
        
        return {
            'color_hints': num_color_hints,
            'rank_hints': num_rank_hints,
            'hint_ratio': float(hint_ratio),
            'avg_cards_revealed': avg_cards_revealed,
            'time_to_first': first_hint_step
        }
    
    def _analyze_cooperation(
        self,
        observations: List[Dict],
        actions: List[chex.Array],
        states: List,
        partner_idx: int,
        action_steps: List[int]
    ) -> Dict:
        """Analyze cooperative behaviors"""
        # Count hints received (when other players hint and partner is target)
        hints_received = 0
        
        # Count plays shortly after receiving hint
        plays_after_hint = 0
        last_hint_step = -10

        for step_idx, action in enumerate(actions):
            # Check if partner received a hint
            partner_action = action[partner_idx]
            other_action = action[(partner_idx + 1) % self.num_agents]
            if other_action in self.color_hint_range or other_action in self.rank_hint_range:
                hints_received += 1
                last_hint_step = step_idx

            # Check if partner played shortly after hint
            if step_idx in action_steps:
                if partner_action in self.play_range and step_idx - last_hint_step <= 3:
                    plays_after_hint += 1
        
        responsiveness = plays_after_hint / hints_received if hints_received > 0 else 0.0
        
        return {
            'hints_received': hints_received,
            'plays_after_hint': plays_after_hint,
            'responsiveness': float(responsiveness)
        }
    
    def _analyze_risk_behavior(
        self,
        actions: chex.Array,
        states: List,
        partner_idx: int,
        action_steps: List[int]
    ) -> Dict:
        """Analyze risk-taking patterns"""
        risky_plays = 0
        conservative_plays = 0
        
        if states is not None:
            for action, step in zip(actions, action_steps):
                if action in self.play_range:
                    # Check if player had strong hints on this card
                    card_idx = action - self.hand_size
                    colors_revealed = states[step].colors_revealed[partner_idx][card_idx]
                    ranks_revealed = states[step].ranks_revealed[partner_idx][card_idx]
                    
                    has_hint = jnp.any(colors_revealed) or jnp.any(ranks_revealed)
                    
                    if has_hint:
                        conservative_plays += 1
                    else:
                        risky_plays += 1
        
        total_plays = risky_plays + conservative_plays
        risk_tolerance = risky_plays / total_plays if total_plays > 0 else 0.0
        
        return {
            'risky_plays': risky_plays,
            'conservative_plays': conservative_plays,
            'risk_tolerance': float(risk_tolerance)
        }
    
    def _analyze_info_management(
        self,
        actions: chex.Array,
        states: List,
        partner_idx: int,
        action_steps: List[int]
    ) -> Dict:
        """Analyze information token management"""
        info_when_hint = []
        info_when_play = []
        depleted_count = 0
        
        if states is not None:
            for action, step in zip(actions, action_steps):
                info_tokens = int(jnp.sum(states[step].info_tokens))
                
                if action in self.color_hint_range or action in self.rank_hint_range:
                    info_when_hint.append(info_tokens)
                elif action in self.play_range:
                    info_when_play.append(info_tokens)
                
                if info_tokens == 0:
                    depleted_count += 1
        
        return {
            'avg_info_when_hint': float(jnp.mean(jnp.array(info_when_hint))) if info_when_hint else 0.0,
            'avg_info_when_play': float(jnp.mean(jnp.array(info_when_play))) if info_when_play else 0.0,
            'depleted_count': depleted_count
        }
    
    def _analyze_card_positions(self, actions: chex.Array) -> Dict:
        """Analyze card position preferences"""
        play_actions = actions[jnp.isin(actions, jnp.array(list(self.play_range)))]
        
        oldest_plays = 0
        newest_plays = 0
        
        if len(play_actions) > 0:
            play_positions = play_actions - self.hand_size
            oldest_plays = int(jnp.sum(play_positions == 0))
            newest_plays = int(jnp.sum(play_positions == self.hand_size - 1))
            
            # Position bias: -1 (left) to 1 (right)
            normalized_positions = (play_positions / (self.hand_size - 1)) * 2 - 1
            position_bias = float(jnp.mean(normalized_positions))
        else:
            position_bias = 0.0
        
        return {
            'oldest_plays': oldest_plays,
            'newest_plays': newest_plays,
            'position_bias': position_bias
        }
    
    def _extract_partner_rewards(
        self,
        rewards: List[Dict],
        partner_key: str,
        action_steps: List[int]
    ) -> chex.Array:
        """Extract rewards at steps when partner acted"""
        partner_rewards = []
        for step in action_steps:
            if step < len(rewards):
                partner_rewards.append(rewards[step][partner_key])
        return jnp.array(partner_rewards) if partner_rewards else jnp.array([0.0])
    
    def _count_bombs(
        self,
        states: List,
        partner_idx: int,
        action_steps: List[int]
    ) -> int:
        """Count number of bombs (life tokens lost) caused by partner"""
        bombs = 0
        for i, step in enumerate(action_steps):
            if step < len(states) - 1:
                lives_before = jnp.sum(states[step].life_tokens)
                lives_after = jnp.sum(states[step + 1].life_tokens)
                if lives_after < lives_before:
                    bombs += 1
        return bombs
    
    def describe_fingerprint(self, fp: HanabiBehaviorFingerprint) -> str:
        """
        Generate concise, LLM-friendly description of behavior fingerprint.
        Optimized for teammate type classification prompts.
        """
        # Determine play style
        if fp.play_success_rate > 0.8:
            play_style = "highly successful plays"
        elif fp.play_success_rate > 0.6:
            play_style = "moderately successful plays"
        elif fp.play_success_rate > 0.3:
            play_style = "risky plays with mixed success"
        else:
            play_style = "frequently unsuccessful plays"
        
        # Determine hint preference
        if fp.num_color_hints + fp.num_rank_hints == 0:
            hint_style = "no hints given"
        elif fp.hint_ratio > 0.7:
            hint_style = "strongly prefers color hints"
        elif fp.hint_ratio < 0.3:
            hint_style = "strongly prefers rank hints"
        else:
            hint_style = "balanced hint usage"
        
        # Determine risk style
        if fp.risk_tolerance > 0.6:
            risk_style = "aggressive (plays without strong hints)"
        elif fp.risk_tolerance > 0.3:
            risk_style = "moderately risk-taking"
        else:
            risk_style = "conservative (waits for hints)"
        
        # Determine cooperation style
        if fp.hint_responsiveness > 0.7:
            coop_style = "highly responsive to hints"
        elif fp.hint_responsiveness > 0.3:
            coop_style = "moderately responsive"
        else:
            coop_style = "less responsive to teammate hints"
        
        action_dist = fp.action_type_distribution
        description = f"""Observed Behavior ({fp.actions_taken} actions over {fp.steps_recorded} steps):

Action Distribution: {action_dist['play']*100:.0f}% plays, {action_dist['discard']*100:.0f}% discards, {action_dist['hint_color']*100:.0f}% color hints, {action_dist['hint_rank']*100:.0f}% rank hints. Diversity: {fp.action_entropy:.2f}.

Play Strategy: {fp.num_plays} play attempts with {fp.play_success_rate*100:.0f}% success rate. First play at step {fp.time_to_first_play}. Prefers position {fp.avg_play_position:.1f} in hand. Classification: {play_style}.

Discard Behavior: {fp.num_discards} discards, average position {fp.avg_discard_position:.1f}. Discarded {fp.discards_when_info_full} times when info tokens were full.

Hint Behavior: {fp.num_color_hints} color hints, {fp.num_rank_hints} rank hints (ratio: {fp.hint_ratio:.2f}). First hint at step {fp.time_to_first_hint}. Style: {hint_style}.

Risk Profile: {fp.risky_plays} risky plays vs {fp.conservative_plays} conservative plays. Risk tolerance: {fp.risk_tolerance:.2f}. Classification: {risk_style}.

Cooperation: Received {fp.hints_received} hints, played {fp.plays_after_hint} times shortly after (responsiveness: {fp.hint_responsiveness:.2f}). Style: {coop_style}.

Information Management: Avg {fp.avg_info_tokens_when_hinting:.1f} tokens when hinting, {fp.avg_info_tokens_when_playing:.1f} when playing. Info depleted {fp.info_tokens_depleted_count} times.

Card Preferences: {fp.oldest_card_plays} oldest card plays, {fp.newest_card_plays} newest card plays. Position bias: {fp.card_position_bias:.2f} (left=-1, right=1).

Performance: {fp.cumulative_reward:.1f} total reward, {fp.avg_reward_per_action:.2f} per action. Caused {fp.bombs_caused} bombs."""
        
        return description


# Example usage
if __name__ == "__main__":
    print("Hanabi Behavior Extractor initialized successfully!")
    
    # Example of how to use:
    # extractor = HanabiBehaviorExtractor(num_agents=2, hand_size=5)
    # 
    # During episode collection:
    # observations, actions, rewards, states = [], [], [], []
    # 
    # After episode:
    # fingerprint = extractor.extract_fingerprint(
    #     observations=observations,
    #     actions=actions,
    #     rewards=rewards,
    #     states=states,
    #     partner_idx=0,
    #     probe_length=20
    # )
    # 
    # description = extractor.describe_for_llm_prompt(fingerprint)
    # print(description)