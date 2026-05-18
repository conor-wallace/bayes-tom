"""Independent PPO (IPPO) with parameter sharing for 2-agent environments."""
from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from bayes_tom.agents.policies.initialize_policies import (
    initialize_mlp_agent,
    initialize_actor_with_double_critic,
    initialize_actor_with_conditional_critic,
    initialize_pseudo_actor_with_double_critic,
    initialize_pseudo_actor_with_conditional_critic,
)
from bayes_tom.marl.ppo_utils import Transition, unbatchify, _create_minibatches


def _init_policy(config, env, rng):
    actor_type = config.get("ACTOR_TYPE", "mlp")
    dispatch = {
        "mlp": initialize_mlp_agent,
        "actor_with_double_critic": initialize_actor_with_double_critic,
        "actor_with_conditional_critic": initialize_actor_with_conditional_critic,
        "pseudo_actor_with_double_critic": initialize_pseudo_actor_with_double_critic,
        "pseudo_actor_with_conditional_critic": initialize_pseudo_actor_with_conditional_critic,
    }
    if actor_type not in dispatch:
        raise ValueError(f"Unknown ACTOR_TYPE: {actor_type}")
    return dispatch[actor_type](config, env, rng)


def make_train(config: dict, env):
    """Return a JIT-able train function for IPPO with parameter sharing.

    The returned function takes a single JAX random key and returns a dict
    with at minimum 'final_params'.

    Required config keys:
        TOTAL_TIMESTEPS, NUM_ENVS, ROLLOUT_LENGTH, LR, ANNEAL_LR,
        MAX_GRAD_NORM, NUM_MINIBATCHES, UPDATE_EPOCHS, CLIP_EPS,
        VF_COEF, ENT_COEF, GAMMA, GAE_LAMBDA, NUM_GAME_AGENTS (default 2).
    """
    num_agents = config.get("NUM_GAME_AGENTS", 2)
    num_envs = config["NUM_ENVS"]
    rollout_len = config["ROLLOUT_LENGTH"]
    num_updates = int(config["TOTAL_TIMESTEPS"] // (num_agents * rollout_len * num_envs))

    def train(rng):
        rng, policy_rng, reset_rng = jax.random.split(rng, 3)
        policy, init_params = _init_policy(config, env, policy_rng)

        def linear_schedule(count):
            frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / num_updates
            return config["LR"] * frac

        if config.get("ANNEAL_LR", False):
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )

        train_state = TrainState.create(
            apply_fn=policy.network.apply,
            params=init_params,
            tx=tx,
        )

        reset_rngs = jax.random.split(reset_rng, num_envs)
        obsv, env_state = jax.vmap(env.reset)(reset_rngs)
        init_done = {k: jnp.zeros((num_envs,), dtype=bool) for k in env.agents + ["__all__"]}

        def _env_step(runner_state, _):
            train_state, env_state, last_obs, last_done, rng = runner_state
            rng, act_rng_0, act_rng_1, step_rng = jax.random.split(rng, 4)

            avail_actions = jax.vmap(env.get_avail_actions)(env_state.env_state)
            avail_0 = avail_actions["agent_0"].astype(jnp.float32)
            avail_1 = avail_actions["agent_1"].astype(jnp.float32)

            act_0, val_0, pi_0, _ = policy.get_action_value_policy(
                params=train_state.params,
                obs=last_obs["agent_0"].reshape(1, num_envs, -1),
                done=last_done["agent_0"].reshape(1, num_envs),
                avail_actions=avail_0,
                hstate=None,
                rng=act_rng_0,
            )
            act_1, val_1, pi_1, _ = policy.get_action_value_policy(
                params=train_state.params,
                obs=last_obs["agent_1"].reshape(1, num_envs, -1),
                done=last_done["agent_1"].reshape(1, num_envs),
                avail_actions=avail_1,
                hstate=None,
                rng=act_rng_1,
            )

            act_0, val_0 = act_0.squeeze(), val_0.squeeze()
            act_1, val_1 = act_1.squeeze(), val_1.squeeze()
            logp_0 = pi_0.log_prob(act_0.reshape(1, num_envs)).squeeze()
            logp_1 = pi_1.log_prob(act_1.reshape(1, num_envs)).squeeze()

            combined = jnp.concatenate([act_0, act_1], axis=0)
            env_act = unbatchify(combined, env.agents, num_envs, num_agents)
            env_act = {k: v.flatten() for k, v in env_act.items()}

            step_rngs = jax.random.split(step_rng, num_envs)
            obs_next, env_state_next, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0))(
                step_rngs, env_state, env_act
            )
            info_0 = jax.tree.map(lambda x: x[:, 0], info)
            info_1 = jax.tree.map(lambda x: x[:, 1], info)

            tr_0 = Transition(
                done=done["agent_0"], action=act_0, value=val_0,
                reward=reward["agent_0"], log_prob=logp_0,
                obs=last_obs["agent_0"], info=info_0, avail_actions=avail_0,
            )
            tr_1 = Transition(
                done=done["agent_1"], action=act_1, value=val_1,
                reward=reward["agent_1"], log_prob=logp_1,
                obs=last_obs["agent_1"], info=info_1, avail_actions=avail_1,
            )
            return (train_state, env_state_next, obs_next, done, rng), (tr_0, tr_1)

        def _calculate_gae(traj_batch, last_val, gamma, gae_lambda):
            def _get_advantages(carry, transition):
                gae, next_val = carry
                done, value, reward = transition.done, transition.value, transition.reward
                delta = reward + gamma * next_val * (1 - done) - value
                gae = delta + gamma * gae_lambda * (1 - done) * gae
                return (gae, value), gae

            _, advantages = jax.lax.scan(
                _get_advantages,
                (jnp.zeros_like(last_val), last_val),
                traj_batch,
                reverse=True,
                unroll=16,
            )
            return advantages, advantages + traj_batch.value

        def _update_epoch(update_state, _):
            train_state, traj_0, traj_1, adv_0, adv_1, tgt_0, tgt_1, rng = update_state
            rng, perm_0, perm_1 = jax.random.split(rng, 3)

            mb_0 = _create_minibatches(traj_0, adv_0, tgt_0, None, num_envs, config["NUM_MINIBATCHES"], perm_0)
            mb_1 = _create_minibatches(traj_1, adv_1, tgt_1, None, num_envs, config["NUM_MINIBATCHES"], perm_1)

            def _update_minibatch(ts, batch):
                (_, tb0, a0, r0), (_, tb1, a1, r1) = batch

                def loss_fn(params):
                    _, v0, pi0, _ = policy.get_action_value_policy(
                        params=params, obs=tb0.obs, done=tb0.done,
                        avail_actions=tb0.avail_actions, hstate=None, rng=jax.random.PRNGKey(0),
                    )
                    _, v1, pi1, _ = policy.get_action_value_policy(
                        params=params, obs=tb1.obs, done=tb1.done,
                        avail_actions=tb1.avail_actions, hstate=None, rng=jax.random.PRNGKey(0),
                    )

                    def pg_loss(pi, tb, adv):
                        lp = pi.log_prob(tb.action)
                        ratio = jnp.exp(lp - tb.log_prob)
                        norm_adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                        return -jnp.mean(jnp.minimum(
                            ratio * norm_adv,
                            jnp.clip(ratio, 1 - config["CLIP_EPS"], 1 + config["CLIP_EPS"]) * norm_adv,
                        ))

                    def vf_loss(v, tb, tgt):
                        v_clip = tb.value + jnp.clip(v - tb.value, -config["CLIP_EPS"], config["CLIP_EPS"])
                        return jnp.maximum(jnp.square(v - tgt), jnp.square(v_clip - tgt)).mean()

                    loss = (
                        pg_loss(pi0, tb0, a0) + pg_loss(pi1, tb1, a1)
                        + config["VF_COEF"] * (vf_loss(v0, tb0, r0) + vf_loss(v1, tb1, r1))
                        - config["ENT_COEF"] * (pi0.entropy().mean() + pi1.entropy().mean())
                    )
                    return loss

                grads = jax.grad(loss_fn)(ts.params)
                return ts.apply_gradients(grads=grads), None

            train_state, _ = jax.lax.scan(_update_minibatch, train_state, (mb_0, mb_1))
            return (train_state, traj_0, traj_1, adv_0, adv_1, tgt_0, tgt_1, rng), None

        def _update_step(runner_state, _):
            train_state, env_state, last_obs, last_done, rng = runner_state

            runner_state, (traj_0, traj_1) = jax.lax.scan(
                _env_step, runner_state, None, rollout_len
            )
            train_state, env_state, last_obs, last_done, rng = runner_state

            avail_last = jax.vmap(env.get_avail_actions)(env_state.env_state)
            _, last_val_0, _, _ = policy.get_action_value_policy(
                params=train_state.params,
                obs=last_obs["agent_0"].reshape(1, num_envs, -1),
                done=last_done["agent_0"].reshape(1, num_envs),
                avail_actions=avail_last["agent_0"].astype(jnp.float32),
                hstate=None, rng=jax.random.PRNGKey(0),
            )
            _, last_val_1, _, _ = policy.get_action_value_policy(
                params=train_state.params,
                obs=last_obs["agent_1"].reshape(1, num_envs, -1),
                done=last_done["agent_1"].reshape(1, num_envs),
                avail_actions=avail_last["agent_1"].astype(jnp.float32),
                hstate=None, rng=jax.random.PRNGKey(0),
            )

            adv_0, tgt_0 = _calculate_gae(traj_0, last_val_0.squeeze(), config["GAMMA"], config["GAE_LAMBDA"])
            adv_1, tgt_1 = _calculate_gae(traj_1, last_val_1.squeeze(), config["GAMMA"], config["GAE_LAMBDA"])

            rng, epoch_rng = jax.random.split(rng)
            update_state = (train_state, traj_0, traj_1, adv_0, adv_1, tgt_0, tgt_1, epoch_rng)
            (train_state, *_), _ = jax.lax.scan(_update_epoch, update_state, None, config["UPDATE_EPOCHS"])

            metric = traj_0.info
            metric["update_steps"] = jnp.array(0)
            return (train_state, env_state, last_obs, last_done, rng), metric

        runner_state = (train_state, env_state, obsv, init_done, rng)
        runner_state, metrics = jax.lax.scan(_update_step, runner_state, None, num_updates)
        final_train_state = runner_state[0]

        return {
            "final_params": final_train_state.params,
            "metrics": metrics,
        }

    return train
