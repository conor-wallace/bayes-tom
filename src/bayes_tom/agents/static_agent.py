from functools import partial

import jax
import jax.numpy as jnp

from bayes_tom.agents.policies.population_interface import AgentPopulation


class StaticAgent:
    population: AgentPopulation

    def __init__(self, population, static_idx):
        self.method = "static"
        self.population = population
        self.static_idx = jnp.array([static_idx])

    def init_hstate(self, batch_size=1, aux_info=None):
        return self.population.policy_cls.init_hstate(1, aux_info={"agent_id": 0})

    def reset(self):
        pass

    def get_action(self, params, partner_indices, obs, done, avail_actions, hstate, rng,
                   aux_obs=None, env_state=None, test_mode=False, trace=None):

        self.pred_partner_idx = self.static_idx

        # This agent always uses the {self.static_idx} best-respose policy
        return self.population.get_actions(params, self.static_idx, obs, done, avail_actions,
                                           hstate, rng, env_state, aux_obs, test_mode=True)