import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple

import numpy as np


class ProbeTrace:
    def __init__(self, episode_idx: int = 0, probe_length: int = 20):
        self.episode_idx = episode_idx
        self.probe_length = probe_length

        self.ego_obs: List[np.ndarray] = []
        self.partner_obs: List[np.ndarray] = []
        self.ego_actions: List[int] = []
        self.partner_actions: List[int] = []
        self.ego_avail_actions: List[float] = []
        self.partner_avail_actions: List[float] = []
        self.ego_rewards: List[float] = []
        self.partner_rewards: List[float] = []

    def is_ready(self):
        return len(self.partner_actions) == self.probe_length

    def insert(
        self,
        last_ego_obs,
        last_partner_obs,
        last_ego_actions,
        last_partner_actions,
        last_ego_avail_actions,
        last_partner_avail_actions,
        last_ego_rewards,
        last_partner_rewards
    ):
        self.ego_obs.append(last_ego_obs)
        self.partner_obs.append(last_partner_obs)
        self.ego_actions.append(last_ego_actions)
        self.partner_actions.append(last_partner_actions)
        self.ego_avail_actions.append(last_ego_avail_actions)
        self.partner_avail_actions.append(last_partner_avail_actions)
        self.ego_rewards.append(last_ego_rewards)
        self.partner_rewards.append(last_partner_rewards)
