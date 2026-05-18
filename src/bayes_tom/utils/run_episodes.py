import jax
import jax.numpy as jnp


def run_episodes(rng, env, agent_0_param, agent_0_policy,
                 agent_1_param, agent_1_policy,
                 max_episode_steps: int, num_eps: int) -> dict:
    """Run num_eps episodes in parallel and return episode statistics.

    Both agents use get_action_value_policy; values and log-probs are discarded.
    The env must be a LogWrapper so that info contains 'returned_episode_returns'.

    Returns:
        dict with key 'returned_episode_returns' of shape (num_eps, num_agents).
    """

    def run_one_episode(rng):
        rng, reset_rng = jax.random.split(rng)
        obs, state = env.reset(reset_rng)

        init_done = {k: jnp.zeros((), dtype=bool) for k in env.agents + ["__all__"]}

        def step_fn(carry, _):
            state, obs, done, rng = carry
            rng, rng0, rng1, step_rng = jax.random.split(rng, 4)

            avail_actions = env.get_avail_actions(state.env_state)
            avail_0 = avail_actions["agent_0"].astype(jnp.float32)
            avail_1 = avail_actions["agent_1"].astype(jnp.float32)

            act_0, _, _, _ = agent_0_policy.get_action_value_policy(
                params=agent_0_param,
                obs=obs["agent_0"][None, None, :],
                done=done["agent_0"][None, None],
                avail_actions=avail_0,
                hstate=None,
                rng=rng0,
            )
            act_1, _, _, _ = agent_1_policy.get_action_value_policy(
                params=agent_1_param,
                obs=obs["agent_1"][None, None, :],
                done=done["agent_1"][None, None],
                avail_actions=avail_1,
                hstate=None,
                rng=rng1,
            )

            actions = {
                "agent_0": act_0.squeeze(),
                "agent_1": act_1.squeeze(),
            }
            obs_next, state_next, _, done_next, info = env.step(step_rng, state, actions)
            return (state_next, obs_next, done_next, rng), info

        _, info = jax.lax.scan(step_fn, (state, obs, init_done, rng), None, max_episode_steps)
        # info["returned_episode_returns"] shape: (max_episode_steps, num_agents)
        # LogWrapper sets returned_episode_returns only at the done step then zeros it; take max.
        return {"returned_episode_returns": info["returned_episode_returns"].max(axis=0)}

    rngs = jax.random.split(rng, num_eps)
    all_info = jax.vmap(run_one_episode)(rngs)
    # Shape: (num_eps, num_agents)
    return all_info
