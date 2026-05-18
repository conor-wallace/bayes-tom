#!/usr/bin/env python
"""
Diagnose behavioral diversity of a saved LBRDiv (or CoMeDi) partner population.

Three checks:
  1. Return matrix   — SP pairs (conf_i, br_i) should outperform XP pairs (conf_i, br_j, i≠j)
  2. Action-prob JSD — partner policies should predict measurably different action distributions
                       given the same probe observations
  3. Behavioral mode — pairwise action-choice agreement rate and per-policy entropy
                       (diverse policies disagree on which action to take)

Usage:
    uv run python scripts/diagnose_diversity.py <path/to/saved_train_run> [options]

Example:
    uv run python scripts/diagnose_diversity.py \\
        src/bayes_tom/outputs/train_parents/lbrdiv_lbf/2026-05-17_21-53-52/saved_train_run \\
        --env lbf --alg lbrdiv
"""
import argparse

import jax
import jax.numpy as jnp
import numpy as np
from prettytable import PrettyTable
from scipy.spatial.distance import jensenshannon

from bayes_tom.agents.policies.mlp_actor_critic_policy import ActorWithConditionalCriticPolicy
from bayes_tom.envs import make_env
from bayes_tom.envs.log_wrapper import LogWrapper
from bayes_tom.utils.run_episodes import run_episodes
from bayes_tom.utils.save_load_utils import load_train_run


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pop_params(all_params, seed_idx):
    """Extract params for all policies in one seed: leaves go from (S, N, ...) → (N, ...)."""
    return jax.tree.map(lambda x: x[seed_idx], all_params)


def _single_params(all_params, seed_idx, policy_idx):
    """Extract params for a single (seed, policy) pair: leaves → (...)."""
    return jax.tree.map(lambda x: x[seed_idx, policy_idx], all_params)


def _action_probs(policy, params, obs, avail, conditioning):
    """Return action probability array of shape (n_obs, n_actions).

    conditioning: (n_obs, pop_size) one-hot opponent ID passed as aux_obs.
    """
    _, _, pi, _ = policy.get_action_value_policy(
        params=params,
        obs=jnp.array(obs),
        done=jnp.zeros(obs.shape[0], dtype=bool),
        avail_actions=jnp.array(avail),
        hstate=None,
        rng=jax.random.PRNGKey(0),
        aux_obs=jnp.array(conditioning),
    )
    return np.array(pi.probs)  # (n_obs, n_actions)


def _collect_probe_obs(env, n_obs, seed):
    """Collect observations and available-action masks via a random-action rollout."""
    rng = jax.random.PRNGKey(seed)
    rng, reset_rng = jax.random.split(rng)
    obs_dict, state = env.reset(reset_rng)

    all_obs, all_avail = [], []
    for _ in range(n_obs):
        raw_avail = env.get_avail_actions(state.env_state)  # dict of bool arrays
        all_obs.append(np.array(obs_dict["agent_0"]))
        all_avail.append(np.array(raw_avail["agent_0"].astype(jnp.float32)))

        rng, step_rng, a0_rng, a1_rng = jax.random.split(rng, 4)
        n_act = env.action_space("agent_0").n
        actions = {
            "agent_0": jax.random.randint(a0_rng, (), 0, n_act),
            "agent_1": jax.random.randint(a1_rng, (), 0, n_act),
        }
        obs_dict, state, _, _, _ = env.step(step_rng, state, actions)

    return np.stack(all_obs), np.stack(all_avail)  # (n_obs, obs_dim), (n_obs, n_actions)


# ─────────────────────────────────────────────────────────────────────────────
# Check 1: SP vs XP return matrix
# ─────────────────────────────────────────────────────────────────────────────

def check_1_returns(env, conf_params, br_params, conf_policy, br_policy, pop_size, n_eps, base_rng):
    print("\n══ Check 1: SP vs XP Return Matrix ══")
    matrix = np.zeros((pop_size, pop_size))

    for i in range(pop_size):
        conf_p = jax.tree.map(lambda x: x[i], conf_params)
        for j in range(pop_size):
            br_p = jax.tree.map(lambda x: x[j], br_params)
            rng = jax.random.fold_in(base_rng, i * pop_size + j)
            info = run_episodes(
                rng, env,
                agent_0_param=conf_p, agent_0_policy=conf_policy,
                agent_1_param=br_p,   agent_1_policy=br_policy,
                max_episode_steps=200,
                num_eps=n_eps,
            )
            matrix[i, j] = float(info["returned_episode_returns"].mean())

    t = PrettyTable(["conf╲br"] + [f"br_{j}" for j in range(pop_size)])
    for i in range(pop_size):
        t.add_row([f"conf_{i}"] + [
            f"{'→' if i == j else ' '}{matrix[i, j]:.3f}" for j in range(pop_size)
        ])
    print(t)
    print("  (→ marks SP pairs on the diagonal)")

    sp_mask = np.eye(pop_size, dtype=bool)
    sp_mean = matrix[sp_mask].mean()
    xp_mean = matrix[~sp_mask].mean() if pop_size > 1 else float("nan")
    print(f"\n  Mean SP return (diagonal):     {sp_mean:.3f}")
    print(f"  Mean XP return (off-diagonal): {xp_mean:.3f}")
    print(f"  SP − XP gap:                   {sp_mean - xp_mean:.3f}")
    return matrix


# ─────────────────────────────────────────────────────────────────────────────
# Check 2: Action-distribution divergence (JSD)
# ─────────────────────────────────────────────────────────────────────────────

def check_2_action_divergence(br_params, br_policy, pop_size, probe_obs, probe_avail):
    print("\n══ Check 2: Action-Distribution Divergence (JSD) ══")
    print(f"  Probe set: {probe_obs.shape[0]} observations")
    print("  Each BR policy is evaluated under its own SP conditioning (one-hot self-ID).")

    identity = np.eye(pop_size)
    n_obs = probe_obs.shape[0]

    # Compute action-probability matrix for each BR policy under its SP conditioning
    all_probs = []
    for i in range(pop_size):
        p = jax.tree.map(lambda x: x[i], br_params)
        cond = np.tile(identity[i], (n_obs, 1))                    # (n_obs, pop_size)
        probs = _action_probs(br_policy, p, probe_obs, probe_avail, cond)
        all_probs.append(probs)
        mean_ent = float(-(probs * np.log(probs + 1e-9)).sum(axis=-1).mean())
        print(f"  br_{i}: mean action entropy = {mean_ent:.3f}")

    # Pairwise JSD (averaged over probe observations)
    jsd_matrix = np.zeros((pop_size, pop_size))
    for i in range(pop_size):
        for j in range(pop_size):
            per_obs_jsd = np.array([
                jensenshannon(all_probs[i][k], all_probs[j][k])
                for k in range(n_obs)
            ])
            jsd_matrix[i, j] = per_obs_jsd.mean()

    t = PrettyTable(["br_i╲br_j"] + [f"br_{j}" for j in range(pop_size)])
    for i in range(pop_size):
        t.add_row([f"br_{i}"] + [f"{jsd_matrix[i, j]:.3f}" for j in range(pop_size)])
    print(t)
    print("  (JSD ∈ [0, 1]:  0 = identical distributions,  1 = maximally different)")

    if pop_size > 1:
        off = jsd_matrix[~np.eye(pop_size, dtype=bool)]
        print(f"\n  Mean off-diagonal JSD: {off.mean():.3f}")
        print(f"  Min  off-diagonal JSD: {off.min():.3f}")
    return jsd_matrix, all_probs


# ─────────────────────────────────────────────────────────────────────────────
# Check 3: Behavioral mode — action agreement and entropy
# ─────────────────────────────────────────────────────────────────────────────

def check_3_behavioral_mode(all_probs, pop_size, probe_obs):
    print("\n══ Check 3: Behavioral Mode Diversity ══")

    # Greedy (mode) action for each policy on each probe observation
    mode_actions = [np.argmax(p, axis=-1) for p in all_probs]  # list of (n_obs,)

    # Action-name labels for LBF: noop up down left right load
    action_names = ["noop", "up", "down", "left", "right", "load"]
    n_actions = all_probs[0].shape[-1]
    labels = action_names[:n_actions] if n_actions <= len(action_names) else [str(a) for a in range(n_actions)]

    # Per-policy: fraction of time each action is the greedy choice
    print("\n  Greedy action frequency per BR policy (fraction of probe obs):")
    t_freq = PrettyTable(["policy"] + labels)
    for i in range(pop_size):
        counts = np.bincount(mode_actions[i], minlength=n_actions) / len(mode_actions[i])
        t_freq.add_row([f"br_{i}"] + [f"{c:.2f}" for c in counts])
    print(t_freq)

    # Pairwise agreement rate: fraction of obs where mode actions match
    if pop_size > 1:
        agree_matrix = np.zeros((pop_size, pop_size))
        for i in range(pop_size):
            for j in range(pop_size):
                agree_matrix[i, j] = (mode_actions[i] == mode_actions[j]).mean()

        t_agree = PrettyTable(["br_i╲br_j"] + [f"br_{j}" for j in range(pop_size)])
        for i in range(pop_size):
            t_agree.add_row([f"br_{i}"] + [f"{agree_matrix[i, j]:.3f}" for j in range(pop_size)])
        print("\n  Pairwise greedy-action agreement rate (fraction of probe obs):")
        print(t_agree)
        print("  (1.0 = always same action,  0.0 = never same action)")

        off = agree_matrix[~np.eye(pop_size, dtype=bool)]
        print(f"\n  Mean off-diagonal agreement: {off.mean():.3f}")
        print(f"  Max  off-diagonal agreement: {off.max():.3f}")
        return agree_matrix


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", help="Path to the saved_train_run directory")
    parser.add_argument("--alg", choices=["lbrdiv", "comedi"], default="lbrdiv",
                        help="Algorithm that produced the checkpoint (determines which param keys to load)")
    parser.add_argument("--env", default="lbf", help="Environment name passed to make_env")
    parser.add_argument("--seed", type=int, default=0, help="Seed index to evaluate (default: 0)")
    parser.add_argument("--n-eps", type=int, default=30,
                        help="Episodes per (conf, br) pair for Check 1 (default: 30)")
    parser.add_argument("--n-probe", type=int, default=512,
                        help="Probe observations for Checks 2 & 3 (default: 512)")
    parser.add_argument("--machine-readable", action="store_true",
                        help="Print METRIC lines at the end for autoresearch parsing")
    args = parser.parse_args()

    # ── Load checkpoint ──────────────────────────────────────────────────────
    print(f"Loading checkpoint: {args.checkpoint}")
    out = load_train_run(args.checkpoint)

    conf_params_all = out["final_params_conf"]
    pop_size = jax.tree.leaves(conf_params_all)[0].shape[1]
    print(f"  Algorithm:       {args.alg}")
    print(f"  Population size: {pop_size}")
    print(f"  Seed index:      {args.seed}")

    conf_params = _pop_params(conf_params_all, args.seed)       # leaves: (pop_size, ...)

    has_br = "final_params_br" in out
    if has_br:
        br_params = _pop_params(out["final_params_br"], args.seed)
    else:
        # CoMeDi has only confederate policies; use them as both sides
        print("  No BR params found — using conf params for both sides (CoMeDi mode)")
        br_params = conf_params

    # ── Build env and policies ────────────────────────────────────────────────
    env = LogWrapper(make_env(args.env, {}))
    obs_dim = env.observation_space(env.agents[0]).shape[0]
    act_dim = env.action_space(env.agents[0]).n

    conf_policy = ActorWithConditionalCriticPolicy(
        action_dim=act_dim, obs_dim=obs_dim, pop_size=pop_size
    )
    br_policy = ActorWithConditionalCriticPolicy(
        action_dim=act_dim, obs_dim=obs_dim, pop_size=pop_size
    )

    base_rng = jax.random.PRNGKey(args.seed)

    # ── Run diagnostics ───────────────────────────────────────────────────────
    return_matrix = check_1_returns(
        env, conf_params, br_params, conf_policy, br_policy,
        pop_size, args.n_eps, base_rng,
    )

    print(f"\nCollecting {args.n_probe} probe observations via random rollout...")
    probe_obs, probe_avail = _collect_probe_obs(env, args.n_probe, seed=args.seed + 99)

    jsd_matrix, all_probs = check_2_action_divergence(
        br_params, br_policy, pop_size, probe_obs, probe_avail
    )

    agree_matrix = check_3_behavioral_mode(all_probs, pop_size, probe_obs)

    print("\n✓ Diagnostics complete.")

    if args.machine_readable:
        sp_returns = return_matrix.diagonal()
        n_collapsed = int(np.sum(sp_returns < 0.05))
        min_sp = float(sp_returns.min())
        mean_sp = float(sp_returns.mean())
        if pop_size > 1:
            off_mask = ~np.eye(pop_size, dtype=bool)
            mean_jsd = float(jsd_matrix[off_mask].mean())
            min_jsd = float(jsd_matrix[off_mask].min())
            mean_agree = float(agree_matrix[off_mask].mean()) if agree_matrix is not None else float("nan")
        else:
            mean_jsd, min_jsd, mean_agree = 0.0, 0.0, 1.0
        composite = mean_jsd * mean_sp  # rewards both diversity and avg performance; penalizes collapse via lower mean_sp
        print()
        print(f"METRIC composite={composite:.6f}")
        print(f"METRIC mean_jsd={mean_jsd:.6f}")
        print(f"METRIC min_jsd={min_jsd:.6f}")
        print(f"METRIC min_sp_return={min_sp:.6f}")
        print(f"METRIC mean_sp_return={mean_sp:.6f}")
        print(f"METRIC n_collapsed={n_collapsed}")
        print(f"METRIC mean_agreement={mean_agree:.6f}")


if __name__ == "__main__":
    main()
