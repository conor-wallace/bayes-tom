from typing import NamedTuple

import jax
import jax.numpy as jnp


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: dict
    avail_actions: jnp.ndarray


def unbatchify(x: jnp.ndarray, agent_list, num_envs: int, num_agents: int) -> dict:
    """Split a flat (num_agents * num_envs, ...) array into a per-agent dict."""
    x = x.reshape((num_agents, num_envs) + x.shape[1:])
    return {a: x[i] for i, a in enumerate(agent_list)}


def _create_minibatches(traj_batch, advantages, targets, hstate,
                        num_envs: int, num_minibatches: int, rng):
    """Flatten, shuffle, and split a trajectory batch into minibatches.

    Args:
        traj_batch: Transition pytree with leaves shaped (rollout_len, num_envs, ...).
        advantages: array shaped (rollout_len, num_envs).
        targets: array shaped (rollout_len, num_envs).
        hstate: ignored (pass None for MLP policies).
        num_envs: number of parallel environments.
        num_minibatches: how many minibatches to create.
        rng: JAX random key for shuffling.

    Returns:
        4-tuple (hstate_mb, traj_mb, adv_mb, targets_mb) where each element
        has a leading dimension of num_minibatches suitable for jax.lax.scan.
    """
    rollout_len = jax.tree.leaves(traj_batch.obs)[0].shape[0]
    batch_size = rollout_len * num_envs
    mb_size = batch_size // num_minibatches

    permutation = jax.random.permutation(rng, batch_size)

    flat_traj = jax.tree.map(
        lambda x: x.reshape((batch_size,) + x.shape[2:]), traj_batch
    )
    flat_adv = advantages.reshape(batch_size)
    flat_targets = targets.reshape(batch_size)

    flat_traj = jax.tree.map(lambda x: x[permutation], flat_traj)
    flat_adv = flat_adv[permutation]
    flat_targets = flat_targets[permutation]

    traj_mb = jax.tree.map(
        lambda x: x.reshape((num_minibatches, mb_size) + x.shape[1:]), flat_traj
    )
    adv_mb = flat_adv.reshape((num_minibatches, mb_size))
    targets_mb = flat_targets.reshape((num_minibatches, mb_size))

    return (None, traj_mb, adv_mb, targets_mb)
