'''Implementation of the LBRDiv teammate generation algorithm (Rahman et al., AAAI 2024)
https://ojs.aaai.org/index.php/AAAI/article/view/29702

Limitations: does not support recurrent actors.
'''
import logging
import os
import shutil
import time
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from flax.training.train_state import TrainState

from bayes_tom.agents.policies.mlp_actor_critic_policy import ActorWithConditionalCriticPolicy
from bayes_tom.agents.policies.population_interface import AgentPopulation
from bayes_tom.envs import make_env
from bayes_tom.envs.log_wrapper import LogWrapper
from bayes_tom.marl.ppo_utils import unbatchify, _create_minibatches
from bayes_tom.utils.plot_utils import get_metric_names
from bayes_tom.utils.run_episodes import run_episodes
from bayes_tom.utils.save_load_utils import save_train_run

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------------
# Utilities shared by (L)BRDiv
# ---------------------------------------------------------------------------

class XPTransition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    self_onehot_id: jnp.ndarray
    oppo_onehot_id: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: dict
    avail_actions: jnp.ndarray


def _get_all_ids(pop_size: int):
    """Return all (conf_id, br_id) pairs as numpy arrays of shape (pop_size^2,)."""
    ids = np.arange(pop_size)
    conf_ids, br_ids = np.meshgrid(ids, ids)
    return conf_ids.ravel(), br_ids.ravel()


def gather_params(params, indices):
    """Index into the leading population axis of a params pytree."""
    return jax.tree.map(lambda x: x[indices], params)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_lbrdiv_partners(train_rng, env, config, conf_policy, br_policy):
    num_agents = env.num_agents
    assert num_agents == 2, "LBRDiv requires exactly 2 agents."

    config["NUM_GAME_AGENTS"] = num_agents
    config["NUM_CONF_ACTORS"] = config["NUM_ENVS"]
    config["NUM_BR_ACTORS"] = config["NUM_ENVS"]
    config["NUM_UPDATES"] = int(config["TOTAL_TIMESTEPS"]) // (
        config["ROLLOUT_LENGTH"] * config["NUM_ENVS"]
    )

    def make_lbrdiv_agents(config):
        def linear_schedule(count):
            frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
            return config["LR"] * frac

        def train(rng):
            rng, init_conf_rng, init_br_rng = jax.random.split(rng, 3)
            all_conf_init_rngs = jax.random.split(init_conf_rng, config["PARTNER_POP_SIZE"])
            all_br_init_rngs = jax.random.split(init_br_rng, config["PARTNER_POP_SIZE"])
            identity_matrix = jnp.eye(config["PARTNER_POP_SIZE"])

            def init_train_states(rng_agents, rng_brs):
                def init_single_pair(rng_agent, rng_br):
                    return conf_policy.init_params(rng_agent), br_policy.init_params(rng_br)

                all_conf_params, all_br_params = jax.vmap(init_single_pair)(rng_agents, rng_brs)

                tx = optax.chain(
                    optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                    optax.adam(
                        learning_rate=linear_schedule if config["ANNEAL_LR"] else config["LR"],
                        eps=1e-5,
                    ),
                )
                tx_br = optax.chain(
                    optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                    optax.adam(
                        learning_rate=linear_schedule if config["ANNEAL_LR"] else config["LR"],
                        eps=1e-5,
                    ),
                )
                train_state_conf = TrainState.create(
                    apply_fn=conf_policy.network.apply,
                    params=all_conf_params,
                    tx=tx,
                )
                train_state_br = TrainState.create(
                    apply_fn=br_policy.network.apply,
                    params=all_br_params,
                    tx=tx_br,
                )
                return train_state_conf, train_state_br

            all_conf_optims, all_br_optims = init_train_states(all_conf_init_rngs, all_br_init_rngs)

            def forward_pass_conf(params, obs, id, done, avail_actions, hstate, rng):
                act, val, pi, new_h = conf_policy.get_action_value_policy(
                    params=params,
                    obs=obs[jnp.newaxis, ...],
                    done=done[jnp.newaxis, ...],
                    avail_actions=avail_actions,
                    hstate=hstate,
                    rng=rng,
                    aux_obs=id[jnp.newaxis, ...],
                )
                return act, val, pi, new_h

            def forward_pass_br(params, obs, id, done, avail_actions, hstate, rng):
                act, val, pi, new_h = br_policy.get_action_value_policy(
                    params=params,
                    obs=obs[jnp.newaxis, ...],
                    done=done[jnp.newaxis, ...],
                    avail_actions=avail_actions,
                    hstate=hstate,
                    rng=rng,
                    aux_obs=id[jnp.newaxis, ...],
                )
                return act, val, pi, new_h

            def _env_step(runner_state, unused):
                (
                    all_ts_conf, all_ts_br, last_conf_ids, last_br_ids,
                    env_state, last_obs, last_done, last_conf_h, last_br_h, rng,
                ) = runner_state
                rng, act0_rng, act1_rng, step_rng, conf_samp_rng, br_samp_rng = jax.random.split(rng, 6)

                needs_resample = last_done["__all__"]
                new_conf_ids = jax.random.randint(conf_samp_rng, (config["NUM_ENVS"],), 0, config["PARTNER_POP_SIZE"])
                new_br_ids = jax.random.randint(br_samp_rng, (config["NUM_ENVS"],), 0, config["PARTNER_POP_SIZE"])
                conf_ids = jnp.where(needs_resample, new_conf_ids, last_conf_ids)
                br_ids = jnp.where(needs_resample, new_br_ids, last_br_ids)

                # hstates are None for MLP policies
                conf_h = last_conf_h
                br_h = last_br_h

                conf_params = gather_params(all_ts_conf.params, conf_ids)
                br_params = gather_params(all_ts_br.params, br_ids)
                conf_onehots = identity_matrix[conf_ids]
                br_onehots = identity_matrix[br_ids]

                avail_actions = jax.vmap(env.get_avail_actions)(env_state.env_state)
                avail_actions = jax.lax.stop_gradient(avail_actions)
                avail_0 = avail_actions["agent_0"].astype(jnp.float32)
                avail_1 = avail_actions["agent_1"].astype(jnp.float32)

                act0_rngs = jax.random.split(act0_rng, config["NUM_ENVS"])
                act_0, val_0, pi_0, new_conf_h = jax.vmap(forward_pass_conf)(
                    conf_params, last_obs["agent_0"], br_onehots,
                    last_done["agent_0"], avail_0, conf_h, act0_rngs,
                )
                logp_0 = pi_0.log_prob(act_0)
                act_0, val_0, logp_0 = act_0.squeeze(), val_0.squeeze(), logp_0.squeeze()

                act1_rngs = jax.random.split(act1_rng, config["NUM_ENVS"])
                act_1, val_1, pi_1, new_br_h = jax.vmap(forward_pass_br)(
                    br_params, last_obs["agent_1"], conf_onehots,
                    last_done["agent_1"], avail_1, br_h, act1_rngs,
                )
                logp_1 = pi_1.log_prob(act_1)
                act_1, val_1, logp_1 = act_1.squeeze(), val_1.squeeze(), logp_1.squeeze()

                combined = jnp.concatenate([act_0, act_1], axis=0)
                env_act = unbatchify(combined, env.agents, config["NUM_ENVS"], num_agents)
                env_act = {k: v.flatten() for k, v in env_act.items()}

                step_rngs = jax.random.split(step_rng, config["NUM_ENVS"])
                obs_next, env_state_next, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0))(
                    step_rngs, env_state, env_act
                )
                info_0 = jax.tree.map(lambda x: x[:, 0], info)
                info_1 = jax.tree.map(lambda x: x[:, 1], info)

                trans_0 = XPTransition(
                    done=done["agent_0"],
                    action=act_0,
                    value=val_0,
                    self_onehot_id=conf_onehots,
                    oppo_onehot_id=br_onehots,
                    reward=reward["agent_1"],
                    log_prob=logp_0,
                    obs=last_obs["agent_0"],
                    info=info_0,
                    avail_actions=avail_0,
                )
                trans_1 = XPTransition(
                    done=done["agent_1"],
                    action=act_1,
                    value=val_1,
                    self_onehot_id=br_onehots,
                    oppo_onehot_id=conf_onehots,
                    reward=reward["agent_1"],
                    log_prob=logp_1,
                    obs=last_obs["agent_1"],
                    info=info_1,
                    avail_actions=avail_1,
                )
                new_runner = (
                    all_ts_conf, all_ts_br, conf_ids, br_ids,
                    env_state_next, obs_next, done, new_conf_h, new_br_h, rng,
                )
                return new_runner, (trans_0, trans_1)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_val, transition):
                    gae, next_val = gae_and_next_val
                    delta = transition.reward + config["GAMMA"] * next_val * (1 - transition.done) - transition.value
                    gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - transition.done) * gae
                    return (gae, transition.value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            def run_all_episodes(rng, train_state_conf, train_state_br):
                conf_ids, br_ids = _get_all_ids(config["PARTNER_POP_SIZE"])
                gathered_conf = gather_params(train_state_conf.params, conf_ids)
                gathered_br = gather_params(train_state_br.params, br_ids)

                rng, eval_rng = jax.random.split(rng)

                def run_pair(conf_param, br_param):
                    return run_episodes(
                        eval_rng, env,
                        conf_param, conf_policy,
                        br_param, br_policy,
                        config["ROLLOUT_LENGTH"],
                        config["NUM_EVAL_EPISODES"],
                    )

                return jax.vmap(run_pair)(gathered_conf, gathered_br)

            def _update_epoch(update_state, unused):
                def _update_minbatch(all_train_states, all_data):
                    train_state_conf, train_state_br = all_train_states
                    minbatch_conf, minbatch_br, lms_vertical, lms_horizontal = all_data

                    def _loss_fn(param, agent_policy, minbatch, agent_id, lms_v, lms_h):
                        init_hstate, traj_batch, gae, target_v = minbatch
                        squeezed_param = jax.tree.map(lambda x: jnp.squeeze(x, 0), param)
                        _, value, pi, _ = agent_policy.get_action_value_policy(
                            params=squeezed_param,
                            obs=traj_batch.obs,
                            done=traj_batch.done,
                            avail_actions=traj_batch.avail_actions,
                            hstate=init_hstate,
                            rng=jax.random.PRNGKey(0),
                            aux_obs=traj_batch.oppo_onehot_id,
                        )
                        log_prob = pi.log_prob(traj_batch.action)

                        # Select transitions relevant to this agent
                        is_relevant = jnp.equal(
                            jnp.argmax(traj_batch.self_onehot_id, axis=-1), agent_id
                        )
                        loss_weights = jnp.where(is_relevant, 1, 0).astype(jnp.float32)
                        int_self_id = jnp.argmax(traj_batch.self_onehot_id, axis=-1)
                        int_oppo_id = jnp.argmax(traj_batch.oppo_onehot_id, axis=-1)

                        def _gather_sp_weights(ids):
                            s_id, _ = ids
                            return jnp.sum(lms_v, axis=-1)[s_id], jnp.sum(lms_h, axis=-1)[s_id]

                        def _gather_xp_weights(ids):
                            s_id, o_id = ids
                            return -lms_v[s_id][o_id], -lms_h[o_id][s_id]

                        def _get_weights(s_id, o_id):
                            return jax.lax.cond(
                                jnp.equal(s_id, o_id),
                                _gather_sp_weights,
                                _gather_xp_weights,
                                (s_id, o_id),
                            )

                        # Value loss
                        value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
                            -config["CLIP_EPS"], config["CLIP_EPS"]
                        )
                        value_losses = jnp.square(value - target_v)
                        value_losses_clipped = jnp.square(value_pred_clipped - target_v)
                        value_loss = jax.lax.cond(
                            loss_weights.sum() == 0,
                            lambda x: jnp.zeros_like(x).astype(jnp.float32),
                            lambda x: x,
                            (loss_weights * jnp.maximum(value_losses, value_losses_clipped)).sum()
                            / (loss_weights.sum() + 1e-8),
                        )

                        # Lagrange-weighted policy gradient loss
                        n = config["PARTNER_POP_SIZE"]
                        is_sp = jnp.equal(int_self_id, int_oppo_id)
                        # Single vmap (our _create_minibatches flattens rollout×envs into one dim)
                        weights1, weights2 = jax.vmap(_get_weights)(int_self_id, int_oppo_id)
                        actor_weights_sp = (weights1 + weights2) * (n / 2)
                        actor_weights_xp = (weights1 + weights2) * (n / (2 * (n - 1)))
                        actor_weights = jnp.where(is_sp, actor_weights_sp, actor_weights_xp)

                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae_norm = (gae - gae.mean()) / (gae.std() + 1e-8)
                        pg_loss_1 = ratio * actor_weights * gae_norm
                        pg_loss_2 = jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * actor_weights * gae_norm
                        pg_loss = jax.lax.cond(
                            loss_weights.sum() == 0,
                            lambda x: jnp.zeros_like(x).astype(jnp.float32),
                            lambda x: x,
                            -(loss_weights * jnp.minimum(pg_loss_1, pg_loss_2)).sum()
                            / (loss_weights.sum() + 1e-8),
                        )

                        # Entropy loss weighted by SP Lagrange multipliers
                        all_sp_w1, all_sp_w2 = jax.vmap(_gather_sp_weights)((int_self_id, int_self_id))
                        entropy_scaler = jnp.maximum(all_sp_w1, all_sp_w2)
                        entropy = jax.lax.cond(
                            loss_weights.sum() == 0,
                            lambda x: jnp.zeros_like(x).astype(jnp.float32),
                            lambda x: x,
                            (loss_weights * entropy_scaler * pi.entropy()).sum()
                            / (loss_weights.sum() + 1e-8),
                        )

                        total_loss = pg_loss + config["VF_COEF"] * value_loss - config["ENT_COEF"] * entropy
                        return total_loss, (value_loss, pg_loss, entropy)

                    possible_agent_ids = jnp.expand_dims(jnp.arange(config["PARTNER_POP_SIZE"]), 1)
                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)

                    def conf_grad(agent_id):
                        pv = gather_params(train_state_conf.params, agent_id)
                        return grad_fn(pv, conf_policy, minbatch_conf, agent_id,
                                       jnp.transpose(lms_vertical), jnp.transpose(lms_horizontal))

                    def br_grad(agent_id):
                        pv = gather_params(train_state_br.params, agent_id)
                        return grad_fn(pv, br_policy, minbatch_br, agent_id, lms_vertical, lms_horizontal)

                    (loss_conf, aux_conf), grads_conf = jax.vmap(conf_grad)(possible_agent_ids)
                    (loss_br, aux_br), grads_br = jax.vmap(br_grad)(possible_agent_ids)

                    grads_conf = jax.tree.map(lambda x: jnp.squeeze(x, 1), grads_conf)
                    grads_br = jax.tree.map(lambda x: jnp.squeeze(x, 1), grads_br)
                    train_state_conf = train_state_conf.apply_gradients(grads=grads_conf)
                    train_state_br = train_state_br.apply_gradients(grads=grads_br)
                    return (train_state_conf, train_state_br), ((loss_conf, aux_conf), (loss_br, aux_br))

                (
                    train_state_conf, train_state_br,
                    traj_batch_conf, traj_batch_br,
                    adv_conf, adv_br, tgt_conf, tgt_br,
                    rng, lms_vertical, lms_horizontal,
                ) = update_state
                rng, perm_rng_conf, perm_rng_br = jax.random.split(rng, 3)

                mbs_conf = _create_minibatches(
                    traj_batch_conf, adv_conf, tgt_conf, None,
                    config["NUM_CONF_ACTORS"], config["NUM_MINIBATCHES"], perm_rng_conf,
                )
                mbs_br = _create_minibatches(
                    traj_batch_br, adv_br, tgt_br, None,
                    config["NUM_BR_ACTORS"], config["NUM_MINIBATCHES"], perm_rng_br,
                )

                num_mbs = mbs_br[1].obs.shape[0]
                rep_lms_v = lms_vertical[jnp.newaxis, ...].repeat(num_mbs, axis=0)
                rep_lms_h = lms_horizontal[jnp.newaxis, ...].repeat(num_mbs, axis=0)

                (train_state_conf, train_state_br), all_losses = jax.lax.scan(
                    _update_minbatch,
                    (train_state_conf, train_state_br),
                    (mbs_conf, mbs_br, rep_lms_v, rep_lms_h),
                )

                update_state = (
                    train_state_conf, train_state_br,
                    traj_batch_conf, traj_batch_br,
                    adv_conf, adv_br, tgt_conf, tgt_br,
                    rng, lms_vertical, lms_horizontal,
                )
                return update_state, all_losses

            def _update_step(update_runner_state, unused):
                (
                    all_ts_conf, all_ts_br,
                    last_env_state, last_obs, last_done, last_conf_h, last_br_h,
                    rng, update_steps, lms_vertical, lms_horizontal,
                ) = update_runner_state

                rng, conf_samp_rng, br_samp_rng = jax.random.split(rng, 3)
                conf_ids = jax.random.randint(conf_samp_rng, (config["NUM_ENVS"],), 0, config["PARTNER_POP_SIZE"])
                br_ids = jax.random.randint(br_samp_rng, (config["NUM_ENVS"],), 0, config["PARTNER_POP_SIZE"])

                runner_state = (
                    all_ts_conf, all_ts_br, conf_ids, br_ids,
                    last_env_state, last_obs, last_done, last_conf_h, last_br_h, rng,
                )
                runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, config["ROLLOUT_LENGTH"])
                (all_ts_conf, all_ts_br, last_conf_ids, last_br_ids,
                 last_env_state, last_obs, last_done, last_conf_h, last_br_h, rng) = runner_state

                traj_batch_conf, traj_batch_br = traj_batch

                # GAE for conf
                last_conf_params = gather_params(all_ts_conf.params, last_conf_ids)
                last_br_onehots = identity_matrix[last_br_ids]
                avail_0 = jax.vmap(env.get_avail_actions)(last_env_state.env_state)["agent_0"].astype(jnp.float32)
                _, last_val_conf, _, _ = jax.vmap(forward_pass_conf)(
                    last_conf_params, last_obs["agent_0"], last_br_onehots,
                    last_done["agent_0"], avail_0, last_conf_h,
                    jax.random.split(jax.random.PRNGKey(0), config["NUM_ENVS"]),
                )
                adv_conf, tgt_conf = _calculate_gae(traj_batch_conf, last_val_conf.squeeze())

                # GAE for br
                last_br_params = gather_params(all_ts_br.params, last_br_ids)
                last_conf_onehots = identity_matrix[last_conf_ids]
                avail_1 = jax.vmap(env.get_avail_actions)(last_env_state.env_state)["agent_1"].astype(jnp.float32)
                _, last_val_br, _, _ = jax.vmap(forward_pass_br)(
                    last_br_params, last_obs["agent_1"], last_conf_onehots,
                    last_done["agent_1"], avail_1, last_br_h,
                    jax.random.split(jax.random.PRNGKey(0), config["NUM_ENVS"]),
                )
                adv_br, tgt_br = _calculate_gae(traj_batch_br, last_val_br.squeeze())

                # PPO epochs
                rng, update_rng = jax.random.split(rng)
                update_state = (
                    all_ts_conf, all_ts_br,
                    traj_batch_conf, traj_batch_br,
                    adv_conf, adv_br, tgt_conf, tgt_br,
                    update_rng, lms_vertical, lms_horizontal,
                )
                update_state, all_losses = jax.lax.scan(_update_epoch, update_state, None, config["UPDATE_EPOCHS"])
                all_ts_conf, all_ts_br = update_state[:2]
                lms_vertical, lms_horizontal = update_state[-2:]

                # Lagrange multiplier updates
                def compute_lagrange_grads_same(params_br, batch, target_value, ids):
                    conf_id, br_id = ids
                    all_target_value = jnp.reshape(target_value, (-1, 1))
                    repeated_value_sp = jnp.repeat(
                        jnp.reshape(all_target_value, (1, -1)), config["PARTNER_POP_SIZE"], axis=0
                    )

                    relevant_conf_params = gather_params(params_br, jnp.reshape(conf_id, (1,)))
                    relevant_conf_params = jax.tree.map(lambda x: jnp.squeeze(x, 0), relevant_conf_params)

                    def _get_value_xp_vary_conf(param, agent_onehot_id):
                        ts, bs = batch.obs.shape[:2]
                        aid = agent_onehot_id[jnp.newaxis, jnp.newaxis, ...].repeat(ts, axis=0).repeat(bs, axis=1)
                        _, val, _, _ = br_policy.get_action_value_policy(
                            params=param, obs=batch.obs, done=batch.done,
                            avail_actions=batch.avail_actions, hstate=None,
                            rng=jax.random.PRNGKey(0), aux_obs=aid,
                        )
                        return val.reshape(ts * bs)

                    all_val_xp_vary_conf = jax.vmap(
                        lambda aid: _get_value_xp_vary_conf(relevant_conf_params, aid)
                    )(jnp.eye(config["PARTNER_POP_SIZE"]))
                    all_val_xp_vary_conf = all_val_xp_vary_conf.at[conf_id].set(repeated_value_sp[conf_id])

                    tol = config["TOLERANCE_FACTOR"]
                    offsets = jnp.zeros_like(repeated_value_sp)
                    offsets = offsets.at[conf_id].set(tol * jnp.ones_like(offsets[conf_id]))
                    grad_sp_vary_conf = repeated_value_sp + offsets - (all_val_xp_vary_conf + tol)

                    relevant_all_params = gather_params(params_br, jnp.arange(config["PARTNER_POP_SIZE"]))

                    def _get_value_xp_vary_br(param):
                        ts, bs = batch.obs.shape[:2]
                        conf_oh = jnp.eye(config["PARTNER_POP_SIZE"])[conf_id]
                        conf_oh = conf_oh[jnp.newaxis, jnp.newaxis, ...].repeat(ts, axis=0).repeat(bs, axis=1)
                        _, val, _, _ = br_policy.get_action_value_policy(
                            params=param, obs=batch.obs, done=batch.done,
                            avail_actions=batch.avail_actions, hstate=None,
                            rng=jax.random.PRNGKey(0), aux_obs=conf_oh,
                        )
                        return val.reshape(ts * bs)

                    all_val_xp_vary_br = jax.vmap(_get_value_xp_vary_br)(relevant_all_params)
                    all_val_xp_vary_br = jnp.reshape(all_val_xp_vary_br, (config["PARTNER_POP_SIZE"], -1))
                    all_val_xp_vary_br = all_val_xp_vary_br.at[conf_id].set(repeated_value_sp[conf_id])
                    grad_sp_vary_br = repeated_value_sp + offsets - (all_val_xp_vary_br + tol)

                    all_self = jnp.reshape(batch.self_onehot_id, (-1, batch.self_onehot_id.shape[-1])).argmax(-1)
                    all_oppo = jnp.reshape(batch.oppo_onehot_id, (-1, batch.oppo_onehot_id.shape[-1])).argmax(-1)
                    self_is_conf = jnp.equal(all_self, conf_id).astype(jnp.float32)
                    oppo_is_conf = jnp.equal(all_oppo, conf_id).astype(jnp.float32)
                    lw = self_is_conf * oppo_is_conf
                    rep_lw = jnp.repeat(jnp.expand_dims(lw, 0), config["PARTNER_POP_SIZE"], axis=0)

                    v_grads = jnp.sum(grad_sp_vary_conf * rep_lw, axis=-1) / (lw.sum() + 1e-8)
                    h_grads = jnp.sum(grad_sp_vary_br * rep_lw, axis=-1) / (lw.sum() + 1e-8)

                    out_v = jnp.zeros((config["PARTNER_POP_SIZE"], config["PARTNER_POP_SIZE"])).at[conf_id].set(v_grads)
                    out_h = jnp.zeros((config["PARTNER_POP_SIZE"], config["PARTNER_POP_SIZE"])).at[conf_id].set(h_grads)
                    return out_v, out_h

                def compute_lagrange_grads_diff(params_br, batch, target_returns, ids):
                    conf_id, br_id = ids
                    param_conf = jax.tree.map(
                        lambda x: jnp.squeeze(x, 0), gather_params(params_br, jnp.reshape(conf_id, (1,)))
                    )
                    param_br = jax.tree.map(
                        lambda x: jnp.squeeze(x, 0), gather_params(params_br, jnp.reshape(br_id, (1,)))
                    )

                    all_self = jnp.reshape(batch.self_onehot_id, (-1, batch.self_onehot_id.shape[-1])).argmax(-1)
                    all_oppo = jnp.reshape(batch.oppo_onehot_id, (-1, batch.oppo_onehot_id.shape[-1])).argmax(-1)
                    all_targets = jnp.reshape(target_returns, (-1,))

                    oppo_is_conf = jnp.equal(all_oppo, conf_id).astype(jnp.float32)
                    self_is_br = jnp.equal(all_self, br_id).astype(jnp.float32)
                    lw = oppo_is_conf * self_is_br

                    ts, bs = batch.obs.shape[:2]
                    conf_oh = jnp.eye(config["PARTNER_POP_SIZE"])[conf_id]
                    conf_oh = conf_oh[jnp.newaxis, jnp.newaxis, ...].repeat(ts, axis=0).repeat(bs, axis=1)
                    br_oh = jnp.eye(config["PARTNER_POP_SIZE"])[br_id]
                    br_oh = br_oh[jnp.newaxis, jnp.newaxis, ...].repeat(ts, axis=0).repeat(bs, axis=1)

                    _, val_br, _, _ = br_policy.get_action_value_policy(
                        params=param_br, obs=batch.obs, done=batch.done,
                        avail_actions=batch.avail_actions, hstate=None,
                        rng=jax.random.PRNGKey(0), aux_obs=br_oh,
                    )
                    val_br = val_br.reshape(bs * ts)

                    _, val_conf, _, _ = br_policy.get_action_value_policy(
                        params=param_conf, obs=batch.obs, done=batch.done,
                        avail_actions=batch.avail_actions, hstate=None,
                        rng=jax.random.PRNGKey(0), aux_obs=conf_oh,
                    )
                    val_conf = val_conf.reshape(bs * ts)

                    tol = config["TOLERANCE_FACTOR"]
                    v_diff = val_br - all_targets - tol
                    h_diff = val_conf - all_targets - tol

                    total_v = (lw * v_diff).sum() / (lw.sum() + 1e-8)
                    total_h = (lw * h_diff).sum() / (lw.sum() + 1e-8)

                    out_v = jnp.zeros((config["PARTNER_POP_SIZE"], config["PARTNER_POP_SIZE"])).at[br_id, conf_id].set(total_v)
                    out_h = jnp.zeros((config["PARTNER_POP_SIZE"], config["PARTNER_POP_SIZE"])).at[conf_id, br_id].set(total_h)
                    return out_v, out_h

                diag_ids = np.arange(config["PARTNER_POP_SIZE"])
                diag_grads = jax.vmap(
                    lambda ci, bi: compute_lagrange_grads_same(
                        all_ts_br.params, traj_batch_br, tgt_br, (ci, bi)
                    )
                )(diag_ids, diag_ids)

                all_conf_ids_np, all_br_ids_np = _get_all_ids(config["PARTNER_POP_SIZE"])
                off_mask = all_conf_ids_np != all_br_ids_np
                off_conf = all_conf_ids_np[off_mask]
                off_br = all_br_ids_np[off_mask]
                off_grads = jax.vmap(
                    lambda ci, bi: compute_lagrange_grads_diff(
                        all_ts_br.params, traj_batch_br, tgt_br, (ci, bi)
                    )
                )(off_conf, off_br)

                avg_grad_v = jnp.sum(diag_grads[0], axis=0) + jnp.sum(off_grads[0], axis=0)
                avg_grad_h = jnp.sum(diag_grads[1], axis=0) + jnp.sum(off_grads[1], axis=0)

                lms_vertical = jnp.maximum(
                    lms_vertical - config["LAGRANGE_LR"] * avg_grad_v,
                    0.5 * jnp.eye(config["PARTNER_POP_SIZE"]),
                )
                lms_vertical = jnp.fill_diagonal(
                    lms_vertical,
                    0.5 * jnp.ones(config["PARTNER_POP_SIZE"], dtype=jnp.float32),
                    inplace=False,
                )
                lms_horizontal = jnp.maximum(
                    lms_horizontal - config["LAGRANGE_LR"] * avg_grad_h,
                    0.5 * jnp.eye(config["PARTNER_POP_SIZE"]),
                )
                lms_horizontal = jnp.fill_diagonal(
                    lms_horizontal,
                    0.5 * jnp.ones(config["PARTNER_POP_SIZE"], dtype=jnp.float32),
                    inplace=False,
                )

                (_, (val_loss_conf, pg_loss_conf, ent_conf)), (_, (val_loss_br, pg_loss_br, ent_br)) = all_losses

                def mask_and_mean(x, mask):
                    return jnp.where(mask, x, 0).sum() / jnp.maximum(1, mask.sum())

                mask = traj_batch_conf.info.get(
                    "returned_episode", jnp.ones_like(traj_batch_conf.reward)
                )
                metric = jax.tree.map(lambda x: mask_and_mean(x, mask), traj_batch_conf.info)
                metric["lms_vertical"] = lms_vertical
                metric["lms_horizontal"] = lms_horizontal
                metric["update_steps"] = update_steps
                metric["value_loss_conf_agent"] = val_loss_conf.mean(axis=(0, 1))
                metric["value_loss_br_agent"] = val_loss_br.mean(axis=(0, 1))
                metric["pg_loss_conf_agent"] = pg_loss_conf.mean(axis=(0, 1))
                metric["pg_loss_br_agent"] = pg_loss_br.mean(axis=(0, 1))
                metric["entropy_conf"] = ent_conf.mean(axis=(0, 1))
                metric["entropy_br"] = ent_br.mean(axis=(0, 1))

                new_runner_state = (
                    all_ts_conf, all_ts_br,
                    last_env_state, last_obs, last_done, last_conf_h, last_br_h,
                    rng, update_steps + 1, lms_vertical, lms_horizontal,
                )
                return new_runner_state, metric

            # Checkpoint + eval loop
            ckpt_and_eval_interval = config["NUM_UPDATES"] // max(1, config["NUM_CHECKPOINTS"] - 1)
            num_ckpts = config["NUM_CHECKPOINTS"]

            def init_ckpt_array(params):
                return jax.tree.map(lambda x: jnp.zeros((num_ckpts,) + x.shape, x.dtype), params)

            def _update_step_with_ckpt(state_with_ckpt, unused):
                (update_runner_state, ckpt_arr_conf, ckpt_arr_br, ckpt_idx, eval_info) = state_with_ckpt

                new_runner_state, metric = _update_step(update_runner_state, None)
                (ts_conf, ts_br, last_env_state, last_obs, last_done, last_conf_h, last_br_h,
                 rng, update_steps, lms_v, lms_h) = new_runner_state

                to_store = jnp.logical_or(
                    jnp.equal(jnp.mod(update_steps - 1, ckpt_and_eval_interval), 0),
                    jnp.equal(update_steps, config["NUM_UPDATES"]),
                )

                def store_and_eval(args):
                    (ckpt_c, ckpt_b, _), rng, cidx = args
                    new_ckpt_c = jax.tree.map(lambda c, p: c.at[cidx].set(p), ckpt_c, ts_conf.params)
                    new_ckpt_b = jax.tree.map(lambda c, p: c.at[cidx].set(p), ckpt_b, ts_br.params)
                    rng, eval_rng = jax.random.split(rng)
                    ep_info = jax.tree.map(
                        lambda x: x.mean(axis=(-2, -1)),
                        run_all_episodes(eval_rng, ts_conf, ts_br),
                    )
                    return (new_ckpt_c, new_ckpt_b, ep_info), rng, cidx + 1

                def skip_ckpt(args):
                    return args

                (ckpt_arrs, rng, ckpt_idx) = jax.lax.cond(
                    to_store,
                    store_and_eval,
                    skip_ckpt,
                    ((ckpt_arr_conf, ckpt_arr_br, eval_info), rng, ckpt_idx),
                )
                ckpt_arr_conf, ckpt_arr_br, eval_ep_last_info = ckpt_arrs
                metric["eval_ep_last_info"] = eval_ep_last_info

                return (
                    (ts_conf, ts_br, last_env_state, last_obs, last_done, last_conf_h, last_br_h,
                     rng, update_steps, lms_v, lms_h),
                    ckpt_arr_conf, ckpt_arr_br, ckpt_idx, eval_ep_last_info,
                ), metric

            ckpt_arr_conf = init_ckpt_array(all_conf_optims.params)
            ckpt_arr_br = init_ckpt_array(all_br_optims.params)
            ckpt_idx = 0
            update_steps = 0

            rng, reset_rng, eval_rng = jax.random.split(rng, 3)
            eval_ep_last_info = jax.tree.map(
                lambda x: x.mean(axis=(-2, -1)),
                run_all_episodes(eval_rng, all_conf_optims, all_br_optims),
            )

            reset_rngs = jax.random.split(reset_rng, config["NUM_ENVS"])
            init_obs, init_env_state = jax.vmap(env.reset)(reset_rngs)
            init_done = {k: jnp.zeros((config["NUM_ENVS"],), dtype=bool) for k in env.agents + ["__all__"]}

            lms_v = 0.5 * jnp.eye(config["PARTNER_POP_SIZE"])
            lms_h = 0.5 * jnp.eye(config["PARTNER_POP_SIZE"])

            update_runner_state = (
                all_conf_optims, all_br_optims,
                init_env_state, init_obs, init_done, None, None,
                rng, update_steps, lms_v, lms_h,
            )

            state_with_ckpt = (update_runner_state, ckpt_arr_conf, ckpt_arr_br, ckpt_idx, eval_ep_last_info)
            state_with_ckpt, metrics = jax.lax.scan(
                _update_step_with_ckpt, state_with_ckpt, xs=None, length=config["NUM_UPDATES"]
            )

            (final_runner_state, ckpt_arr_conf, ckpt_arr_br, _, all_ep_infos) = state_with_ckpt
            return {
                "final_params_conf": final_runner_state[0].params,
                "final_params_br": final_runner_state[1].params,
                "checkpoints_conf": ckpt_arr_conf,
                "checkpoints_br": ckpt_arr_br,
                "metrics": metrics,
                "all_pair_returns": all_ep_infos,
            }

        return train

    train_fn = make_lbrdiv_agents(config)
    return train_fn(train_rng)


def get_lbrdiv_population(config, out, env):
    pop_size = config["algorithm"]["PARTNER_POP_SIZE"]
    partner_params = out["final_params_conf"]

    partner_policy = ActorWithConditionalCriticPolicy(
        action_dim=env.action_space(env.agents[1]).n,
        obs_dim=env.observation_space(env.agents[1]).shape[0],
        pop_size=pop_size,
        activation=config["algorithm"].get("ACTIVATION", "tanh"),
    )

    num_seeds = jax.tree.leaves(partner_params)[0].shape[0]
    flat_params = jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), partner_params)

    partner_population = AgentPopulation(
        pop_size=num_seeds * pop_size,
        policy_cls=partner_policy,
        params=flat_params,
    )
    return partner_params, partner_population


def run_lbrdiv(config, wandb_logger):
    algorithm_config = dict(config["algorithm"])

    env = make_env(algorithm_config["ENV_NAME"], algorithm_config["ENV_KWARGS"])
    env = LogWrapper(env)

    log.info("Starting LBRDiv training...")
    start = time.time()

    rng = jax.random.PRNGKey(algorithm_config["TRAIN_SEED"])
    rngs = jax.random.split(rng, algorithm_config["NUM_SEEDS"])

    conf_policy = ActorWithConditionalCriticPolicy(
        action_dim=env.action_space(env.agents[0]).n,
        obs_dim=env.observation_space(env.agents[0]).shape[0],
        pop_size=algorithm_config["PARTNER_POP_SIZE"],
        activation=algorithm_config.get("ACTIVATION", "tanh"),
    )
    br_policy = ActorWithConditionalCriticPolicy(
        action_dim=env.action_space(env.agents[0]).n,
        obs_dim=env.observation_space(env.agents[0]).shape[0],
        pop_size=algorithm_config["PARTNER_POP_SIZE"],
        activation=algorithm_config.get("ACTIVATION", "tanh"),
    )

    with jax.disable_jit(False):
        vmapped_train_fn = jax.jit(
            jax.vmap(
                partial(
                    train_lbrdiv_partners,
                    env=env, config=algorithm_config,
                    conf_policy=conf_policy, br_policy=br_policy,
                )
            )
        )
        out = vmapped_train_fn(rngs)

    end = time.time()
    log.info(f"LBRDiv training complete in {end - start:.1f}s")

    metric_names = get_metric_names(algorithm_config["ENV_NAME"])
    log_metrics(config, out, wandb_logger, metric_names)

    return get_lbrdiv_population(config, out, env)


def log_metrics(config, outs, logger, metric_names: tuple):
    metrics = outs["metrics"]
    num_seeds, num_updates, pop_size = metrics["pg_loss_conf_agent"].shape

    # Eval return curves
    all_returns = np.asarray(metrics["eval_ep_last_info"]["returned_episode_returns"])
    xs = list(range(num_updates))

    all_conf_ids, all_br_ids = _get_all_ids(pop_size)
    sp_mask = all_conf_ids == all_br_ids
    sp_returns = all_returns[:, :, sp_mask].mean(axis=(0, 2))
    xp_returns = all_returns[:, :, ~sp_mask].mean(axis=(0, 2))

    for step in range(num_updates):
        logger.log_item("Eval/AvgSPReturnCurve", sp_returns[step], train_step=step)
        logger.log_item("Eval/AvgXPReturnCurve", xp_returns[step], train_step=step)
    logger.commit()

    # Final XP matrix
    last_returns = all_returns[:, -1].mean(axis=0).reshape(pop_size, pop_size)
    logger.log_xp_matrix("Eval/LastXPMatrix", last_returns)

    # Loss curves (wandb only)
    use_wandb = config.get("logger", {}).get("use_wandb", False)
    if use_wandb:
        processed_losses = {
            "ConfPGLoss": np.asarray(metrics["pg_loss_conf_agent"]).mean(axis=0).T,
            "BRPGLoss": np.asarray(metrics["pg_loss_br_agent"]).mean(axis=0).T,
            "ConfValLoss": np.asarray(metrics["value_loss_conf_agent"]).mean(axis=0).T,
            "BRValLoss": np.asarray(metrics["value_loss_br_agent"]).mean(axis=0).T,
            "ConfEntropy": np.asarray(metrics["entropy_conf"]).mean(axis=0).T,
            "BREntropy": np.asarray(metrics["entropy_br"]).mean(axis=0).T,
        }
        keys = [f"pair {i}" for i in range(pop_size)]
        for loss_name, loss_data in processed_losses.items():
            logger.log_item(
                f"Losses/{loss_name}",
                wandb.plot.line_series(xs=xs, ys=loss_data, keys=keys, title=loss_name, xname="train_step"),
            )

        lm_keys = [f"{i},{j}" for i in range(pop_size) for j in range(pop_size)]
        lm_h = np.asarray(metrics["lms_horizontal"]).mean(axis=0).reshape(num_updates, -1).T
        lm_v = np.asarray(metrics["lms_vertical"]).mean(axis=0).reshape(num_updates, -1).T
        for name, data in [("LMs_Horizontal", lm_h), ("LMs_Vertical", lm_v)]:
            logger.log_item(
                f"Losses/{name}",
                wandb.plot.line_series(xs=xs, ys=data, keys=lm_keys, title=name, xname="train_step"),
            )
        logger.commit()

    # Save artifacts
    run_timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    savedir = os.path.join(config.get("output_dir", "outputs/train_parents"), run_timestamp)
    out_savepath = save_train_run(outs, savedir, savename="saved_train_run")
    if config["logger"]["log_train_out"]:
        logger.log_artifact(name="saved_train_run", path=out_savepath, type_name="train_run")
    if not config["local_logger"]["save_train_out"]:
        shutil.rmtree(out_savepath)
