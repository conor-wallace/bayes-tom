import random
from functools import partial

import jax
import jax.numpy as jnp

from bayes_tom.agents.policies.population_interface import AgentPopulation


class RandomAgent:
    population: AgentPopulation

    def __init__(self, population):
        self.method = "random"
        self.population = population

    def init_hstate(self, batch_size=1, aux_info=None):
        return self.population.policy_cls.init_hstate(1, aux_info={"agent_id": 0})

    def reset(self):
        pass

    def get_action(self, params, partner_indices, obs, done, avail_actions, hstate, rng,
                   aux_obs=None, env_state=None, test_mode=False, trace=None):

        random_policy_idx = random.randint(0, self.population.pop_size-1)
        random_policy_idx = jnp.array([random_policy_idx])

        self.pred_partner_idx = random_policy_idx

        return self.population.get_actions(params, random_policy_idx, obs, done, avail_actions,
                                           hstate, rng, env_state, aux_obs, test_mode=True)