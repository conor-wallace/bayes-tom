import jax
import jax.numpy as jnp
import chex
from jumanji.environments.routing.lbf.generator import RandomGenerator
from jumanji.environments.routing.lbf.types import Agent, Food, State


class DifferentLevelsGenerator(RandomGenerator):
    """LBF generator with a fixed food-level distribution.

    All agents are fixed at level 1. Food levels are set to one-third level-1
    (solo-collectible) and two-thirds level-2 (requires both agents), shuffled
    each episode. For 6 foods this yields [1,1,2,2,2,2].
    """

    def __call__(self, key: chex.PRNGKey) -> State:
        key_food, key_agents, key_shuffle, key = jax.random.split(key, 4)

        food_positions = self.sample_food(key_food)

        mask = jnp.ones((self.grid_size, self.grid_size), dtype=bool)
        mask = mask.at[food_positions].set(False)
        mask = mask.ravel()
        agent_positions = self.sample_agents(key=key_agents, mask=mask)

        agent_levels = jnp.ones(self.num_agents, dtype=jnp.int32)

        num_ones = self.num_food // 3
        num_twos = self.num_food - num_ones
        food_levels_base = jnp.concatenate([
            jnp.ones(num_ones, dtype=jnp.int32),
            jnp.full(num_twos, 2, dtype=jnp.int32),
        ])
        food_levels = jax.random.permutation(key_shuffle, food_levels_base)

        agents = jax.vmap(Agent)(
            id=jnp.arange(self.num_agents),
            position=agent_positions,
            level=agent_levels,
            loading=jnp.zeros(self.num_agents, dtype=bool),
        )
        food_items = jax.vmap(Food)(
            id=jnp.arange(self.num_food),
            position=food_positions,
            level=food_levels,
            eaten=jnp.zeros(self.num_food, dtype=bool),
        )

        return State(
            key=key,
            step_count=jnp.array(0, jnp.int32),
            agents=agents,
            food_items=food_items,
        )
