from typing import Any, NamedTuple

import jax
import jax.numpy as jnp


class BufferState(NamedTuple):
    params: Any   # pytree with leading dim (max_pop_size, ...)
    count: jnp.ndarray  # scalar int, number of agents written so far


class BufferedPopulation:
    """A JAX-compatible fixed-capacity population buffer.

    Unlike AgentPopulation (which holds params inside the object), the params
    live in a separate BufferState pytree so they can be threaded through
    jax.lax.scan without capturing mutable Python state.
    """

    def __init__(self, max_pop_size: int, policy_cls):
        self.max_pop_size = max_pop_size
        self.policy_cls = policy_cls

    def reset_buffer(self, init_params) -> BufferState:
        """Allocate a zeroed buffer with capacity max_pop_size."""
        params = jax.tree.map(
            lambda x: jnp.zeros((self.max_pop_size,) + x.shape, x.dtype),
            init_params,
        )
        return BufferState(params=params, count=jnp.array(0, dtype=jnp.int32))

    def add_agent(self, buffer: BufferState, new_params) -> BufferState:
        """Write new_params into slot buffer.count and increment the counter."""
        new_params_stored = jax.tree.map(
            lambda buf, p: buf.at[buffer.count].set(p),
            buffer.params,
            new_params,
        )
        return BufferState(params=new_params_stored, count=buffer.count + 1)

    def gather_agent_params(self, buffer: BufferState, agent_indices):
        """Gather params at agent_indices. Returns pytree with leading dim len(agent_indices)."""
        agent_indices = jnp.atleast_1d(agent_indices)
        return jax.tree.map(
            lambda p: jax.vmap(lambda idx: p[idx])(agent_indices),
            buffer.params,
        )
