import bisect
from functools import partial
from itertools import accumulate
from typing import Any, Iterable, Sequence

import jax
import jax.numpy as jnp


def _maybe_convert_to_int(x):
    if isinstance(x, jnp.ndarray):
        return int(x[0])
    return x


class AgentPopulation:
    '''Base class for a population of homogeneous agents
    TODO: develop more complex population classes that can handle heterogeneous agents
    '''
    def __init__(self, pop_size, policy_cls, params):
        '''
        Args:
            pop_size: int, number of agents in the population
            policy_cls: an instance of the AgentPolicy class. The policy class for the population of agents
            params: initial parameters for the population
        '''
        self.pop_size = pop_size
        self.policy_cls = policy_cls # AgentPolicy class
        self.params = params # pytree of parameters for the population of agents, with shape (pop_size, ...)

    def sample_agent_indices(self, n, rng):
        '''Sample n indices from the population, with replacement.'''
        return jax.random.randint(rng, (n,), 0, self.pop_size)
    
    def gather_agent_params(self, agent_indices):
        '''Gather the parameters of the agents specified by agent_indices.

        Args:
            pop_params: pytree of parameters for the population of agents of shape (pop_size, ...).
            agent_indices: indices with shape (num_envs,), each in [0, pop_size)
        '''
        def gather_leaf(leaf):
            # leaf shape: (num_envs,  ...)
            return jax.vmap(lambda idx: leaf[idx])(agent_indices)
        return jax.tree.map(gather_leaf, self.params)
    
    def get_actions(self, agent_indices, obs, done, avail_actions, hstate, rng, 
                    env_state=None, aux_obs=None, test_mode=False):
        '''
        Get the actions of the agents specified by agent_indices. 
        
        Args:
            agent_indices: indices with shape (num_envs,), each in [0, pop_size)
            obs: observations with shape (num_envs, ...) 
            done: done flags with shape (num_envs,)
            avail_actions: available actions with shape (num_envs, num_actions)
            hstate: hidden state with shape (num_envs, ...) or None if policy doesn't use hidden state
            rng: random key
            env_state: environment state with shape (num_envs, ...) or None if policy doesn't use env state
            aux_obs: an optional auxiliary vector to append to the observation
        Returns:
            actions: actions with shape (num_envs,)
            new_hstate: new hidden state with shape (num_envs, ...) or None
        '''
        # print("In Get Actions")
        # print("Agent indices: ", agent_indices)
        gathered_params = self.gather_agent_params(agent_indices)
        agent_indices = jnp.atleast_1d(agent_indices)
        num_envs = agent_indices.shape[0]
        rngs_batched = jax.random.split(rng, num_envs)
        vmapped_get_action = jax.vmap(partial(self.policy_cls.get_action, 
                                              aux_obs=aux_obs, 
                                              env_state=env_state, 
                                              test_mode=test_mode))
        actions, new_hstate = vmapped_get_action(
            gathered_params, obs, done, avail_actions, hstate, 
            rngs_batched)
        return actions, new_hstate

    def get_action_value_policies(self, agent_indices, obs, done, avail_actions, hstate, rng, 
                    env_state=None, aux_obs=None, test_mode=False):
        '''
        Get the actions of the agents specified by agent_indices. 
        
        Args:
            agent_indices: indices with shape (num_envs,), each in [0, pop_size)
            obs: observations with shape (num_envs, ...) 
            done: done flags with shape (num_envs,)
            avail_actions: available actions with shape (num_envs, num_actions)
            hstate: hidden state with shape (num_envs, ...) or None if policy doesn't use hidden state
            rng: random key
            env_state: environment state with shape (num_envs, ...) or None if policy doesn't use env state
            aux_obs: an optional auxiliary vector to append to the observation
        Returns:
            actions: actions with shape (num_envs,)
            new_hstate: new hidden state with shape (num_envs, ...) or None
        '''
        # print("In Get Action Value Policies")
        # print("Agent indices: ", agent_indices)
        gathered_params = self.gather_agent_params(agent_indices)
        agent_indices = jnp.atleast_1d(agent_indices)
        num_envs = agent_indices.shape[0]
        rngs_batched = jax.random.split(rng, num_envs)
        vmapped_get_action_value_policy = jax.vmap(partial(self.policy_cls.get_action_value_policy, 
                                              aux_obs=aux_obs, 
                                              env_state=env_state))

        actions, values, probs, new_hstate = vmapped_get_action_value_policy(
            gathered_params, obs, done, avail_actions, hstate, 
            rngs_batched
        )
        return actions, values, probs, new_hstate

    def init_hstate(self, n: int, aux_info: dict=None):
        '''Initialize the hidden state for n members of the population.'''
        return self.policy_cls.init_hstate(n, aux_info)


class NestedAgentPopulation:
    '''A population of AgentPopulation objects.

    This wrapper keeps the child-population API intact while adding a parent
    selection dimension. Each parent selects one AgentPopulation, and then the
    child agent index is resolved within that selected population.
    '''

    def __init__(self, populations: Sequence[AgentPopulation]):
        if len(populations) == 0:
            raise ValueError("populations must not be empty")

        self.populations = list(populations)
        self.pop_size = sum([pop.pop_size for pop in self.populations])
        self.sub_pop_sizes = [pop.pop_size for pop in self.populations]
        self.cumulative_sizes = list(accumulate(self.sub_pop_sizes))

    def init_hstate(self, n: int, aux_info: dict=None):
        '''Initialize the hidden state for n members of the population.'''
        return self.populations[0].policy_cls.init_hstate(n, aux_info)

    def sample_parent_indices(self, n, rng):
        '''Sample n parent indices with replacement.'''
        return jax.random.randint(rng, (n,), 0, self.pop_size)

    def sample_agent_indices(self, n, rng):
        '''Sample n (parent_idx, child_idx) pairs with replacement.'''
        rng, parent_rng, child_rng = jax.random.split(rng, 3)
        parent_indices = self.sample_parent_indices(n, parent_rng)
        child_indices = jnp.zeros((n,), dtype=jnp.int32)

        for i in range(n):
            parent_idx = int(parent_indices[i])
            child_indices = child_indices.at[i].set(
                jax.random.randint(child_rng, (), 0, self.child_pop_sizes[parent_idx])
            )
            child_rng, _ = jax.random.split(child_rng)

        return parent_indices, child_indices

    def _normalize_indices(self, agent_idx):
        agent_idx = _maybe_convert_to_int(agent_idx)

        pop_idx = bisect.bisect_right(self.cumulative_sizes, agent_idx)
        sub_pop_idx = agent_idx - (self.cumulative_sizes[pop_idx - 1] if pop_idx > 0 else 0)

        return pop_idx, jnp.array([sub_pop_idx])

    def gather_agent_params(self, parent_indices, agent_indices):
        '''Gather parameters for the selected parent/child pairs.'''
        parent_indices, agent_indices = self._normalize_indices(parent_indices, agent_indices)

        gathered_params = []
        for parent_idx, agent_idx in zip(parent_indices.tolist(), agent_indices.tolist()):
            population = self.parent_populations[int(parent_idx)]
            params = self.parent_params[int(parent_idx)]
            selected_params = population.gather_agent_params(params, jnp.array([int(agent_idx)]))
            gathered_params.append(selected_params)

        return jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *gathered_params)

    def get_actions(self, agent_indices, obs, done, avail_actions, hstate, rng,
                    env_state=None, aux_obs=None, test_mode=False):
        '''Get actions for the selected parent/child pairs.'''
        sub_pop_index, sub_pop_agent_indices = self._normalize_indices(agent_indices)

        return self.populations[sub_pop_index].get_actions(
            sub_pop_agent_indices, obs, done, avail_actions, hstate, rng, env_state, aux_obs, test_mode
        )

    def get_action_value_policies(self, parent_indices, agent_indices, obs, done, avail_actions, hstate, rng,
                    env_state=None, aux_obs=None, test_mode=False):
        '''Get action/value/policy outputs for selected parent/child pairs.'''
        parent_indices, agent_indices = self._normalize_indices(parent_indices, agent_indices)
        gathered_params = self.gather_agent_params(parent_indices, agent_indices)

        num_envs = parent_indices.shape[0]
        rngs_batched = jax.random.split(rng, num_envs)
        vmapped_get_action_value_policy = jax.vmap(partial(self.policy_cls.get_action_value_policy,
                                              aux_obs=aux_obs,
                                              env_state=env_state))

        actions, values, probs, new_hstate = vmapped_get_action_value_policy(
            gathered_params,
            obs,
            done,
            avail_actions,
            hstate,
            rngs_batched,
        )
        return actions, values, probs, new_hstate

    # def init_hstate(self, n: int, aux_info: dict=None):
    #     '''Initialize hidden state for n members of the nested population.'''
    #     return self.policy_cls.init_hstate(n, aux_info)