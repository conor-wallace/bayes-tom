import os
import logging
import pickle
from pathlib import Path
from pprint import pprint

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import distrax
import flax.linen as nn
from tqdm import tqdm

from bayes_tom.agents import load_agent
from bayes_tom.agents.llm import ChatClient
from bayes_tom.agents.policies.population_interface import AgentPopulation, NestedAgentPopulation
from bayes_tom.envs import make_env
from bayes_tom.envs.log_wrapper import LogWrapper
from bayes_tom.utils.agent_loader_from_config import initialize_rl_agent_from_config
from bayes_tom.utils.history import ProbeTrace

logger = logging.getLogger("Assitant")
logger.setLevel(logging.ERROR)
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)


class DiscretePolicy(nn.Module):
    """Discrete action policy used by child IQL checkpoints."""
    hidden_dims: tuple
    action_dim: int
    layer_norm: bool = False

    @nn.compact
    def __call__(self, observations: jnp.ndarray, temperature: float = 1.0, avail_actions=None):
        logits = MLP(
            (*self.hidden_dims, self.action_dim),
            layer_norm=self.layer_norm,
        )(observations)
        logits = logits / jnp.maximum(temperature, 1e-8)

        # Match ActorWithConditionalCritic behavior: mask unavailable actions.
        if avail_actions is not None:
            avail_actions = jnp.asarray(avail_actions, dtype=logits.dtype)
            unavail_actions = 1.0 - avail_actions
            logits = logits - (unavail_actions * 1e10)

        return distrax.Categorical(logits=logits)


def default_init(scale: float = jnp.sqrt(2.0)):
    return nn.initializers.orthogonal(scale)


class MLP(nn.Module):
    """MLP module matching the checkpoint architecture used in child IQL training."""
    hidden_dims: tuple
    activations: callable = nn.relu
    activate_final: bool = False
    layer_norm: bool = False
    kernel_init: callable = default_init()

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for i, hidden_dim in enumerate(self.hidden_dims):
            x = nn.Dense(hidden_dim, kernel_init=self.kernel_init)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                if self.layer_norm:
                    x = nn.LayerNorm()(x)
                x = self.activations(x)
        return x


class IQLPolicyAdapter:
    """Adapter that exposes AgentPopulation-compatible methods for IQL children."""

    def __init__(self, hidden_dims=(256, 256), action_dim=6, layer_norm=True):
        self.policy_model = DiscretePolicy(
            hidden_dims=hidden_dims,
            action_dim=action_dim,
            layer_norm=layer_norm,
        )

    @staticmethod
    def _as_variables(params):
        """Normalize loaded checkpoint params into Flax variables dict."""
        if isinstance(params, dict) and "params" in params:
            return params
        return {"params": params}

    def init_hstate(self, n: int, aux_info: dict = None):
        return jnp.zeros((n, 1), dtype=jnp.float32)

    def get_action(self, params, obs, done, avail_actions, hstate, rng,
                   aux_obs=None, env_state=None, test_mode=False):
        del done, rng, aux_obs, env_state, test_mode
        obs = jnp.reshape(obs, (-1,))
        avail_actions = None if avail_actions is None else jnp.reshape(avail_actions, (-1,))
        variables = self._as_variables(params)
        dist = self.policy_model.apply(variables, obs, temperature=1.0, avail_actions=avail_actions)
        action = dist.probs.argmax(axis=-1)
        return action, hstate

    def get_action_value_policy(self, params, obs, done, avail_actions, hstate, rng,
                                aux_obs=None, env_state=None):
        del done, rng, aux_obs, env_state
        obs = jnp.reshape(obs, (-1,))
        avail_actions = None if avail_actions is None else jnp.reshape(avail_actions, (-1,))
        variables = self._as_variables(params)
        dist = self.policy_model.apply(variables, obs, temperature=1.0, avail_actions=avail_actions)
        action = dist.probs.argmax(axis=-1)
        values = jnp.zeros((1,), dtype=jnp.float32)
        return action, values, dist.probs, hstate


def _is_nested_population(population):
    return isinstance(population, NestedAgentPopulation)


def _discover_iql_child_checkpoints(base_dir: Path, num_parents: int):
    checkpoints = {i: [] for i in range(num_parents)}
    cwd = Path(__file__).resolve().parent
    ckpt_dir = cwd / base_dir
    if not ckpt_dir.exists():
        logger.warning(f"IQL checkpoint directory not found: {ckpt_dir}")
        return checkpoints

    parent_dirs = sorted(ckpt_dir.glob("ego_agent_iql_multiseed_parent_*"))
    for parent_dir in parent_dirs:
        if not parent_dir.is_dir():
            continue
        try:
            parent_id = int(parent_dir.name.split("_")[-1])
        except (ValueError, IndexError):
            continue
        if parent_id < 0 or parent_id >= num_parents:
            continue
        for seed_dir in sorted(parent_dir.glob("seed_*")):
            if not seed_dir.is_dir():
                continue
            actor_file = seed_dir / "actor_params.npy"
            if not actor_file.exists():
                continue
            try:
                seed_id = int(seed_dir.name.split("_")[1])
            except (ValueError, IndexError):
                seed_id = len(checkpoints[parent_id])
            checkpoints[parent_id].append((seed_dir, seed_id))

    return checkpoints


def _load_nested_iql_population(
    base_dir: Path,
    num_parents: int,
    action_dim: int,
    apply_obs_norm: bool = True,
    norm_clip: float = 10.0,
):
    child_checkpoints = _discover_iql_child_checkpoints(base_dir, num_parents)
    parent_populations = []
    parent_params = []
    parent_obs_means = []
    parent_obs_stds = []
    parent_obs_norm_enabled = []

    adapter = IQLPolicyAdapter(action_dim=action_dim)

    def _load_norm_stats_for_child(checkpoint_dir: Path):
        """Load normalization stats for one child checkpoint.

        Priority:
        1) seed directory: <...>/seed_X/norm_stats.npz
        2) parent directory: <...>/ego_agent_iql_multiseed_parent_Y/norm_stats.npz
        """
        candidates = [
            checkpoint_dir / "norm_stats.npz",
            checkpoint_dir.parent / "norm_stats.npz",
        ]
        for stats_path in candidates:
            if not stats_path.exists():
                continue
            try:
                with np.load(stats_path) as stats:
                    obs_mean = np.asarray(stats["obs_mean"], dtype=np.float32)
                    obs_std = np.asarray(stats["obs_std"], dtype=np.float32)
                return obs_mean, obs_std
            except Exception as e:
                logger.warning(f"Failed loading normalization stats from {stats_path}: {e}")
        return None, None

    def _load_normalize_state_flag(checkpoint_dir: Path):
        """Infer normalize_state from saved training config when available."""
        candidates = [
            checkpoint_dir / "config.txt",
            checkpoint_dir.parent / "config.txt",
        ]
        for cfg_path in candidates:
            if not cfg_path.exists():
                continue
            try:
                txt = cfg_path.read_text()
                if "normalize_state=False" in txt:
                    return False
                if "normalize_state=True" in txt:
                    return True
            except Exception:
                continue
        # Training default in IQLConfig is True.
        return True

    for parent_id in sorted(child_checkpoints.keys()):
        print(f"Discovered child checkpoints for parent {parent_id}: {len(child_checkpoints[parent_id])} seeds")
        children = child_checkpoints[parent_id]
        if not children:
            continue
        params_list = []
        obs_means = []
        obs_stds = []
        obs_norm_flags = []
        for checkpoint_dir, _seed in children:
            print(f"Loading checkpoint from {checkpoint_dir}")
            params_path = checkpoint_dir / "actor_params.npy"
            try:
                params = np.load(params_path, allow_pickle=True).item()
                params_list.append(params)
                obs_mean, obs_std = _load_norm_stats_for_child(checkpoint_dir)
                obs_means.append(obs_mean)
                obs_stds.append(obs_std)
                obs_norm_flags.append(_load_normalize_state_flag(checkpoint_dir))
            except Exception as e:
                logger.warning(f"Failed loading {params_path}: {e}")
        if not params_list:
            continue
        stacked_params = jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *params_list)
        parent_populations.append(AgentPopulation(pop_size=len(params_list), policy_cls=adapter))
        parent_params.append(stacked_params)
        parent_obs_means.append(obs_means)
        parent_obs_stds.append(obs_stds)
        parent_obs_norm_enabled.append(obs_norm_flags)

    if not parent_populations:
        return None
    nested_population = NestedAgentPopulation(parent_populations=parent_populations, parent_params=parent_params)
    # Attach normalization metadata parallel to nested parent/child indexing.
    nested_population.child_obs_means = parent_obs_means
    nested_population.child_obs_stds = parent_obs_stds
    nested_population.child_obs_norm_enabled = parent_obs_norm_enabled
    nested_population.apply_obs_norm = bool(apply_obs_norm)
    nested_population.norm_clip = float(norm_clip)
    return nested_population


def _init_partner_hstate(partner_population, partner_idx: int, child_idx: int):
    if _is_nested_population(partner_population):
        return partner_population.init_hstate(1, aux_info={"agent_id": 1})
    return partner_population.policy_cls.init_hstate(1, aux_info={"agent_id": 1})


def _partner_action(partner_population, partner_params, partner_idx_batched, child_idx_batched,
                    obs, done, avail_actions, hstate, rng, env_state):
    if _is_nested_population(partner_population):
        # Apply per-child observation normalization when available.
        try:
            if getattr(partner_population, "apply_obs_norm", True):
                parent_i = int(partner_idx_batched[0])
                child_i = int(child_idx_batched[0])
                use_norm = partner_population.child_obs_norm_enabled[parent_i][child_i]
                obs_mean = partner_population.child_obs_means[parent_i][child_i]
                obs_std = partner_population.child_obs_stds[parent_i][child_i]
                if use_norm and obs_mean is not None and obs_std is not None:
                    print(f"Applying observation normalization for parent {parent_i} child {child_i}")
                    obs_mean = jnp.asarray(obs_mean, dtype=obs.dtype).reshape(1, 1, -1)
                    obs_std = jnp.asarray(obs_std, dtype=obs.dtype).reshape(1, 1, -1)
                    # Exact IQL training-time normalization behavior:
                    # observations = (observations - obs_mean) / obs_std
                    # where obs_std already includes +1e-5 from dataset preprocessing.
                    obs = (obs - obs_mean) / obs_std
        except Exception:
            # If stats are missing or malformed, fall back to raw observation.
            pass

        return partner_population.get_actions(
            parent_indices=partner_idx_batched,
            agent_indices=child_idx_batched,
            obs=obs,
            done=done,
            avail_actions=avail_actions,
            hstate=hstate,
            rng=rng,
            aux_obs=None,
            env_state=env_state,
            test_mode=True,
        )
    return partner_population.get_actions(
        pop_params=partner_params,
        agent_indices=partner_idx_batched,
        obs=obs,
        done=done,
        avail_actions=avail_actions,
        hstate=hstate,
        rng=rng,
        aux_obs=None,
        env_state=env_state,
    )


def parse_args():
    """Parse command line arguments for partner evaluation"""
    parser = argparse.ArgumentParser(description='Partner Evaluation Script')
    
    # Agent selection
    parser.add_argument('--agent', type=str, required=True,
                        choices=['oracle', 'static', 'random', 'plastic', 'liam', 'meliba',
                                'llm_zero', 'llm_cot', 'llm_few', 'llm_ip', 'bayestom',
                                'recollab', 'collab'],
                        help='Type of ego agent to use')
    
    parser.add_argument('--static-idx', type=int, default=0,
                        help='Static policy index (only for static agent)')
    
    # Experiment modes
    parser.add_argument('--mode', type=str, default='evaluate',
                        choices=['evaluate', 'probe', 'learn', 'dataset'],
                        help='Experiment mode: evaluate, probe, learn, or dataset generation')
    
    # Evaluation parameters
    parser.add_argument('--num-episodes', type=int, default=None,
                        help='Number of evaluation episodes (defaults to config value)')
    
    parser.add_argument('--probe-lengths', type=int, nargs='+', 
                        default=[40, 80, 140, 200],
                        help='Probe lengths for probe mode (space-separated list)')
    
    # Configuration
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (overrides config)')
    
    parser.add_argument('--model-name', type=str, default=None,
                        help='LLM model name (overrides config)')
    
    # Dataset generation
    parser.add_argument('--dataset-path', type=str, default=None,
                        help='Path to save offline dataset (defaults to config results_path/offline_dataset.hdf5)')
    
    # Output
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed metrics')
    
    return parser.parse_args()


def create_ego_agent(agent_type, config, ego_population, train_partner_population, 
                     train_flattened_partner_params, env, ego_init_rng, llm_client=None, 
                     static_idx=0):
    """Factory function to create ego agent based on type"""
    
    agent_class = load_agent(agent_type)

    if agent_type == 'oracle':
        return agent_class(ego_population)
    
    elif agent_type == 'static':
        return agent_class(ego_population, static_idx=static_idx)
    
    elif agent_type == 'random':
        return agent_class(ego_population)
    
    elif agent_type == 'plastic':
        return agent_class(config, ego_population, train_partner_population, train_flattened_partner_params)
    
    elif agent_type == 'liam':
        return agent_class(config, env, ego_init_rng)
    
    elif agent_type == 'meliba':
        return agent_class(config, env, ego_init_rng)
    
    elif agent_type == 'llm_zero':
        if llm_client is None:
            raise ValueError("LLM client required for LLM agents")
        return agent_class(config=config, ego_population=ego_population, llm=llm_client)
    
    elif agent_type == 'llm_cot':
        if llm_client is None:
            raise ValueError("LLM client required for LLM agents")
        return agent_class(config=config, ego_population=ego_population, llm=llm_client)
    
    elif agent_type == 'llm_few':
        if llm_client is None:
            raise ValueError("LLM client required for LLM agents")
        return agent_class(config=config, ego_population=ego_population, llm=llm_client)
    
    elif agent_type == 'llm_ip':
        if llm_client is None:
            raise ValueError("LLM client required for LLM agents")
        return agent_class(config=config, ego_population=ego_population, llm=llm_client)
    
    elif agent_type == 'bayestom':
        if llm_client is None:
            raise ValueError("LLM client required for LLM agents")
        return agent_class(
            config=config,
            ego_population=ego_population,
            partner_population=train_partner_population,
            partner_params=train_flattened_partner_params,
            llm=llm_client
        )
    
    elif agent_type == 'recollab':
        if llm_client is None:
            raise ValueError("LLM client required for LLM agents")
        return agent_class(config=config, ego_population=ego_population, llm=llm_client)
    
    elif agent_type == 'collab':
        if llm_client is None:
            raise ValueError("LLM client required for LLM agents")
        return agent_class(config=config, ego_population=ego_population, llm=llm_client)
    
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def run_probe_phase(rng, env, ego_agent, ego_params, partner_population, partner_params, partner_idx, episode_idx, max_episode_steps, partner_child_idx=0):    
    # Reset the env.
    rng, reset_rng = jax.random.split(rng)
    obs, env_state = env.reset(reset_rng)
    done = {k: jnp.zeros((1), dtype=bool) for k in env.agents + ["__all__"]}
    act_onehot = {k: jnp.zeros((env.action_space(env.agents[i]).n)) for i, k in enumerate(env.agents)}
    reward = {k: jnp.zeros((1)) for i, k in enumerate(env.agents)}
    joint_act_onehot = jnp.concatenate((act_onehot["agent_0"].reshape(1, 1, -1),
                                             act_onehot["agent_1"].reshape(1, 1, -1)), axis=-1)

    trace = ProbeTrace(episode_idx=episode_idx, probe_length=max_episode_steps)

    # Initialize hidden states. Agent id is passed as part of the hstate initialization to support heuristic agents.
    hstate_0 = ego_agent.population.policy_cls.init_hstate(1, aux_info={"agent_id": 0})
    hstate_1 = _init_partner_hstate(partner_population, int(partner_idx[0]), partner_child_idx)
    partner_child_idx_batched = jnp.array([partner_child_idx])

    t = 0

    while t <= max_episode_steps:
        # Get available actions from environment state
        avail_actions = env.get_avail_actions(env_state.env_state)
        avail_actions = jax.lax.stop_gradient(avail_actions)
        avail_actions_0 = avail_actions["agent_0"].astype(jnp.float32).reshape(1, -1)
        avail_actions_1 = avail_actions["agent_1"].astype(jnp.float32).reshape(1, -1)

        # Update random number generators for the step
        rng, act0_rng, act1_rng, step_rng = jax.random.split(rng, 4)

        # Get ego action
        act_0, hstate_0 = ego_agent.get_action(
            params=ego_params,
            partner_indices=partner_idx,
            obs=obs["agent_0"].reshape(1, 1, -1),
            done=done["agent_0"].reshape(1, 1),
            avail_actions=avail_actions_0,
            hstate=hstate_0,
            rng=act0_rng,
            aux_obs=(act_onehot["agent_0"].reshape(1, 1, -1), joint_act_onehot, reward["agent_0"].reshape(1, 1, -1)),
            env_state=env_state,
            trace=trace
        )
        act_0 = act_0.squeeze()

        # Get partner action using the underlying policy class's get_action method directly
        act_1, hstate_1 = _partner_action(
            partner_population=partner_population,
            partner_params=partner_params,
            partner_idx_batched=partner_idx,
            child_idx_batched=partner_child_idx_batched,
            obs=obs["agent_1"].reshape(1, 1, -1),
            done=done["agent_1"].reshape(1, 1),
            avail_actions=avail_actions_1,
            hstate=hstate_1,
            rng=act1_rng,
            env_state=env_state,
        )
        act_1 = act_1.squeeze()

        act_onehot = {
            "agent_0": jax.nn.one_hot(act_0, env.action_space("agent_0").n),
            "agent_1": jax.nn.one_hot(act_1, env.action_space("agent_1").n),
        }
        joint_act_onehot = jnp.concatenate(
            (act_onehot["agent_0"].reshape(1, 1, -1),
            act_onehot["agent_1"].reshape(1, 1, -1)),
            axis=-1
        )

        both_actions = [act_0, act_1]
        env_act = {k: both_actions[i] for i, k in enumerate(env.agents)}
        env_act_onehot = {k: jax.nn.one_hot(both_actions[i], env.action_space(env.agents[i]).n) for i, k in enumerate(env.agents)}
        next_obs, env_state, reward, done, next_info = env.step(step_rng, env_state, env_act)

        trace.insert(
            last_partner_obs=obs["agent_1"],
            last_ego_obs=obs["agent_0"],
            last_partner_actions=act_1,
            last_ego_actions=act_0,
            last_partner_avail_actions=avail_actions_1,
            last_ego_avail_actions=avail_actions_0,
            last_partner_rewards=reward["agent_1"],
            last_ego_rewards=reward["agent_0"]
        )

        obs = next_obs

        t += 1

        if bool(done["__all__"]):
            terminal_info = next_info
            break

    return next_info, trace


def run_single_episode(rng, env, ego_agent, ego_params, partner_population, partner_params, partner_idx, max_episode_steps, partner_child_idx=0):    
    # Reset the env.
    rng, reset_rng = jax.random.split(rng)
    obs, env_state = env.reset(reset_rng)
    done = {k: jnp.zeros((1), dtype=bool) for k in env.agents + ["__all__"]}
    act_onehot = {k: jnp.zeros((env.action_space(env.agents[i]).n)) for i, k in enumerate(env.agents)}
    reward = {k: jnp.zeros((1)) for i, k in enumerate(env.agents)}
    joint_act_onehot = jnp.concatenate((act_onehot["agent_0"].reshape(1, 1, -1),
                                             act_onehot["agent_1"].reshape(1, 1, -1)), axis=-1)

    trace = ProbeTrace()

    # Initialize hidden states. Agent id is passed as part of the hstate initialization to support heuristic agents.
    hstate_0 = ego_agent.init_hstate(1, aux_info={"agent_id": 0})
    hstate_1 = _init_partner_hstate(partner_population, int(partner_idx[0]), partner_child_idx)
    partner_child_idx_batched = jnp.array([partner_child_idx])

    t = 0
    true_partners = []
    pred_partners = []
    
    # Per-timestep metrics
    timestep_ego_rewards = []
    timestep_partner_rewards = []
    timestep_ego_actions = []
    timestep_partner_actions = []
    timestep_predictions = []
    timestep_correct = []
    timestep_nll = []
    timestep_entropy = []

    while t <= max_episode_steps:
        # Get available actions from environment state
        avail_actions = env.get_avail_actions(env_state.env_state)
        avail_actions = jax.lax.stop_gradient(avail_actions)
        avail_actions_0 = avail_actions["agent_0"].astype(jnp.float32).reshape(1, -1)
        avail_actions_1 = avail_actions["agent_1"].astype(jnp.float32).reshape(1, -1)

        # Update random number generators for the step
        rng, act0_rng, act1_rng, step_rng = jax.random.split(rng, 4)

        # Get ego action
        act_0, hstate_0 = ego_agent.get_action(
            params=ego_params,
            partner_indices=partner_idx,
            obs=obs["agent_0"].reshape(1, 1, -1),
            done=done["agent_0"].reshape(1, 1),
            avail_actions=avail_actions_0,
            hstate=hstate_0,
            rng=act0_rng,
            aux_obs=(act_onehot["agent_0"].reshape(1, 1, -1), joint_act_onehot, reward["agent_0"].reshape(1, 1, -1)),
            env_state=env_state,
            trace=trace
        )
        act_0 = act_0.squeeze()

        # Get partner action using the underlying policy class's get_action method directly
        act_1, hstate_1 = _partner_action(
            partner_population=partner_population,
            partner_params=partner_params,
            partner_idx_batched=partner_idx,
            child_idx_batched=partner_child_idx_batched,
            obs=obs["agent_1"].reshape(1, 1, -1),
            done=done["agent_1"].reshape(1, 1),
            avail_actions=avail_actions_1,
            hstate=hstate_1,
            rng=act1_rng,
            env_state=env_state,
        )
        act_1 = act_1.squeeze()

        # print("Partner obs: ", obs["agent_1"])
        # print("Partner avail actions: ", avail_actions_1)
        # print("Partner logits: ", logits_1)
        # print("Partner action: ", act_1)

        act_onehot = {
            "agent_0": jax.nn.one_hot(act_0, env.action_space("agent_0").n),
            "agent_1": jax.nn.one_hot(act_1, env.action_space("agent_1").n),
        }
        joint_act_onehot = jnp.concatenate(
            (act_onehot["agent_0"].reshape(1, 1, -1),
            act_onehot["agent_1"].reshape(1, 1, -1)),
            axis=-1
        )

        # print("True partner: ", int(partner_idx[0]))
        # print("Pred partner: ", int(ego_agent.pred_partner_idx[0]))

        true_partners.append(int(partner_idx[0]))
        pred_partners.append(int(ego_agent.pred_partner_idx[0]))
        
        # Store per-timestep metrics
        timestep_ego_rewards.append(float(reward["agent_0"].item()))
        timestep_partner_rewards.append(float(reward["agent_1"].item()))
        timestep_ego_actions.append(int(act_0))
        timestep_partner_actions.append(int(act_1))
        timestep_predictions.append(int(ego_agent.pred_partner_idx[0]))
        timestep_correct.append(int(partner_idx[0]) == int(ego_agent.pred_partner_idx[0]))
        
        # Store NLL and entropy if available
        if hasattr(ego_agent, "nll") and ego_agent.nll is not None:
            nll_val = ego_agent.nll.item() if hasattr(ego_agent.nll, 'item') else float(ego_agent.nll)
            timestep_nll.append(nll_val)
        if hasattr(ego_agent, "entropy") and ego_agent.entropy is not None:
            entropy_val = ego_agent.entropy.item() if hasattr(ego_agent.entropy, 'item') else float(ego_agent.entropy)
            timestep_entropy.append(entropy_val)

        both_actions = [act_0, act_1]
        env_act = {k: both_actions[i] for i, k in enumerate(env.agents)}
        env_act_onehot = {k: jax.nn.one_hot(both_actions[i], env.action_space(env.agents[i]).n) for i, k in enumerate(env.agents)}
        next_obs, env_state, reward, done, next_info = env.step(step_rng, env_state, env_act)

        trace.insert(
            last_partner_obs=obs["agent_1"],
            last_ego_obs=obs["agent_0"],
            last_partner_actions=act_1,
            last_ego_actions=act_0,
            last_partner_avail_actions=avail_actions_1,
            last_ego_avail_actions=avail_actions_0,
            last_partner_rewards=reward["agent_1"],
            last_ego_rewards=reward["agent_0"]
        )

        obs = next_obs

        t += 1

        if bool(done["__all__"]):
            terminal_info = next_info
            break

    accuracy = sum(1 for pred, true in zip(pred_partners, true_partners) if pred == true) / len(true_partners)

    next_info["accuracy"] = accuracy
    next_info["episode_length"] = t
    
    # Add per-timestep metrics
    next_info["timestep_ego_rewards"] = timestep_ego_rewards
    next_info["timestep_partner_rewards"] = timestep_partner_rewards
    next_info["timestep_ego_actions"] = timestep_ego_actions
    next_info["timestep_partner_actions"] = timestep_partner_actions
    next_info["timestep_predictions"] = timestep_predictions
    next_info["timestep_correct"] = timestep_correct
    next_info["true_partner"] = int(partner_idx[0])
    
    # Add per-timestep NLL and entropy if available
    if timestep_nll:
        next_info["timestep_nll"] = timestep_nll
    if timestep_entropy:
        next_info["timestep_entropy"] = timestep_entropy

    # Add episode-level NLL and entropy
    if hasattr(ego_agent, "nll"):
        next_info["nll"] = ego_agent.nll
    if hasattr(ego_agent, "entropy"):
        next_info["entropy"] = ego_agent.entropy

    return next_info, trace


def evaluate(
    config,
    env,
    rng,
    num_episodes,
    ego_agent,
    ego_params,
    partner_population,
    partner_params,
    collect_timestep_data=False
):
    '''Evaluate ego policy switching vs partner population
    
    Args:
        collect_timestep_data: If True, store per-timestep metrics for each episode
    '''
    num_agents = env.num_agents
    assert num_agents == 2, "This eval code assumes exactly 2 agents."

    num_partner_total = partner_population.pop_size
    partner_child_idx = config.get("agent_model", {}).get("partner_child_idx", 0)
    results = []
    
    # Optional: collect all timestep data across episodes
    all_timestep_data = [] if collect_timestep_data else None

    ego_agent.is_learning = False

    pbar = tqdm(
        desc="Evaluating agent...",
        total=int(num_episodes) * int(num_partner_total)
    )
    for partner_idx in range(num_partner_total):
        # Create a partner-specific RNG from the base RNG
        rng, partner_rng = jax.random.split(rng)
        for episode_idx in range(num_episodes):
            # Wrap scalar in batch dimension for vmap compatibility
            print("Evaluating with Partner ", partner_idx)
            ego_agent.reset()
            partner_idx_batched = jnp.array([partner_idx])
            # Split RNG for each episode to ensure different random seeds
            partner_rng, episode_rng = jax.random.split(partner_rng)
            result, trace = run_single_episode(
                episode_rng, env, ego_agent, ego_params, 
                partner_population, partner_params, 
                partner_idx_batched,  # Pass as array instead of scalar
                max_episode_steps=config["ROLLOUT_LENGTH"],
                partner_child_idx=partner_child_idx,
                # max_episode_steps=config["agent_model"]["probe_steps"]+1
            )
            result["episode_idx"] = episode_idx
            result["partner_idx"] = partner_idx
            
            # Optionally collect timestep data
            if collect_timestep_data:
                timestep_info = {
                    "episode_idx": episode_idx,
                    "partner_idx": partner_idx,
                    "timestep_ego_rewards": result.pop("timestep_ego_rewards"),
                    "timestep_partner_rewards": result.pop("timestep_partner_rewards"),
                    "timestep_ego_actions": result.pop("timestep_ego_actions"),
                    "timestep_partner_actions": result.pop("timestep_partner_actions"),
                    "timestep_predictions": result.pop("timestep_predictions"),
                    "timestep_correct": result.pop("timestep_correct"),
                    "true_partner": result.get("true_partner"),
                }
                # Add NLL and entropy if available
                if "timestep_nll" in result:
                    timestep_info["timestep_nll"] = result.pop("timestep_nll")
                if "timestep_entropy" in result:
                    timestep_info["timestep_entropy"] = result.pop("timestep_entropy")
                all_timestep_data.append(timestep_info)
            
            # print('Episode length: ', result["episode_length"])
            print('Episode return: ', result["returned_episode_returns"][0])
            results.append(result)
            pbar.update(1)

    pbar.close()
    
    # Log average return per partner
    partner_returns = {}
    for result in results:
        partner_idx = result["partner_idx"]
        episode_return = result["returned_episode_returns"][0]
        if partner_idx not in partner_returns:
            partner_returns[partner_idx] = []
        partner_returns[partner_idx].append(episode_return)
    
    print("\n" + "="*60)
    print("AVERAGE RETURN PER PARTNER")
    print("="*60)
    for partner_idx in sorted(partner_returns.keys()):
        avg_return = np.mean(partner_returns[partner_idx])
        std_return = np.std(partner_returns[partner_idx])
        print(f"Partner {partner_idx}: {avg_return:.3f} ± {std_return:.3f} (n={len(partner_returns[partner_idx])})")
    print("="*60 + "\n")

    if collect_timestep_data:
        return results, all_timestep_data
    return results


def learn(config, env, rng, num_episodes, ego_agent, ego_params,
          partner_population, partner_params):
    num_partner_total = partner_population.pop_size
    partner_child_idx = config.get("agent_model", {}).get("partner_child_idx", 0)
    results = []

    ego_agent.is_learning = True

    behavior_model_dir = os.path.join(
        str(config["agent_model"]["behavior_model_path"]),
        str(config["ENV_NAME"]),
    )

    if not os.path.exists(behavior_model_dir):
        os.makedirs(behavior_model_dir)

    pbar = tqdm(
        desc="Learning from prior experience...",
        total=int(config["agent_model"]["num_episodes"]) * int(num_partner_total)
    )
    for partner_idx in range(num_partner_total):
        traces = []
        rng, partner_rng = jax.random.split(rng)
        for episode_idx in range(config["agent_model"]["num_episodes"]):
            # partner_idx = 1
            ego_agent.reset()
            partner_idx_batched = jnp.array([partner_idx])
            partner_rng, episode_rng = jax.random.split(partner_rng)
            result, trace = run_probe_phase(
                episode_rng, env, ego_agent, ego_params, 
                partner_population, partner_params, 
                partner_idx_batched,  # Pass as array instead of scalar
                episode_idx,
                # max_episode_steps=config["agent_model"]["probe_steps"],
                max_episode_steps=config["ROLLOUT_LENGTH"],
                partner_child_idx=partner_child_idx,
            )

            ego_agent.learn(
                trace=trace,
                partner_idx=partner_idx,
                episode_idx=episode_idx
            )

            traces.append(trace)
            pbar.update(1)

        all_partner_obs = np.concatenate([trace.partner_obs for trace in traces], axis=0)
        all_partner_act = np.concatenate([trace.partner_actions for trace in traces], axis=0)

        behavior_model_path = os.path.join(behavior_model_dir, f"partner_{partner_idx}_behavior_model.pkl")
        behavior_model = KNNBehaviorModel(k=50, continuous_actions=False, sigma_s=0.5)
        behavior_model.fit(all_partner_obs, all_partner_act)
        metrics = behavior_model.eval(all_partner_obs, all_partner_act)
        pprint(metrics)
        behavior_model.save(behavior_model_path)


def probe(config, env, rng, num_episodes, ego_population, ego_params,
          partner_population, partner_params, probe_lengths):
    all_results = []

    for probe_length in probe_lengths:
        print(f"Evaluating Probe Length: {probe_length}")
        config["agent_model"]["probe_steps"] = probe_length

        llm_client = ChatClient(
            model_name=config["agent_model"]["model_name"],
        )

        ego_agent = LLMInversePlanningAgent(
            config=config,
            ego_population=ego_population,
            llm=llm_client,
        )

        if hasattr(ego_agent, "learn"):
            learn(
                config, env, rng, num_episodes, 
                ego_agent, ego_params, partner_population, partner_params
            )

        results = evaluate(
            config, env, rng, num_episodes, 
            ego_agent, ego_params, partner_population, partner_params
        )

        for result in results:
            all_results.append(
                {
                    "return": result["returned_episode_returns"][0],
                    "accuracy": result["accuracy"],
                    "nll": result.get("nll", None),
                    "entropy": result.get("entropy", None),
                    "episode_index": result["episode_idx"],
                    "length": probe_length
                }
            )

    df = pd.DataFrame(data=all_results)
    df.to_csv("probe_results.csv", index=False)
    print("Probe results saved to probe_results.csv")

    return all_results


def sweep_alpha(config, env, rng, num_episodes, ego_population, ego_params,
                partner_population, partner_params, alpha_values):
    """Sweep alpha parameter for BayesToM agent
    
    Args:
        config: Configuration dictionary
        env: Environment instance
        rng: JAX random key
        num_episodes: Number of evaluation episodes
        ego_population: Ego agent population
        ego_params: Ego agent parameters
        partner_population: Partner population
        partner_params: Partner parameters
        alpha_values: List of alpha values to test
    """
    all_results = []

    for alpha in alpha_values:
        print(f"Evaluating Alpha: {alpha:.2f}")
        config["agent_model"]["alpha"] = alpha

        llm_client = ChatClient(
            model_name=config["agent_model"]["model_name"],
        )

        ego_agent = BayesToMAgent(
            config=config,
            ego_population=ego_population,
            partner_population=partner_population,
            partner_params=partner_params,
            llm=llm_client,
        )

        # if hasattr(ego_agent, "learn"):
        #     learn(
        #         config, env, rng, num_episodes, 
        #         ego_agent, ego_params, partner_population, partner_params
        #     )

        results = evaluate(
            config, env, rng, num_episodes, 
            ego_agent, ego_params, partner_population, partner_params
        )

        for result in results:
            all_results.append(
                {
                    "return": result["returned_episode_returns"][0],
                    "accuracy": result["accuracy"],
                    "nll": result.get("nll", None),
                    "entropy": result.get("entropy", None),
                    "episode_index": result["episode_idx"],
                    "alpha": alpha
                }
            )

    df = pd.DataFrame(data=all_results)
    df.to_csv("alpha_sweep_results.csv", index=False)
    print("Alpha sweep results saved to alpha_sweep_results.csv")

    return all_results


def run_partner_evaluation(config, print_metrics=False):
    '''Run partner evaluation
    
    Config parameters:
        task.agent_model.agent_type: Agent type (oracle, static, random, plastic, liam, meliba, llm_*, bayestom, recollab, collab)
        task.agent_model.mode: Mode (evaluate, probe, learn, dataset, sweep_alpha)
        task.agent_model.static_idx: Static policy index (for static agent)
        task.agent_model.probe_lengths: List of probe lengths (for probe mode)
        task.agent_model.alpha_values: List of alpha values (for sweep_alpha mode)
        task.agent_model.dataset_path: Dataset save path (for dataset mode)
    '''

    # Extract parameters from config with defaults
    agent_type = config.get("agent_model", {}).get("agent_type", "bayestom")
    mode = config.get("agent_model", {}).get("mode", "evaluate")
    static_idx = config.get("agent_model", {}).get("static_idx", 0)
    probe_lengths = config.get("agent_model", {}).get("probe_lengths", [40, 80, 140, 200])
    alpha_values = config.get("agent_model", {}).get("alpha_values", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    dataset_path = config.get("agent_model", {}).get("dataset_path", None)
    dataset_episodes = config.get("agent_model", {}).get("dataset_episodes", 100)
    num_eval_episodes = config["NUM_EVAL_EPISODES"]

    # Create only one environment instance
    env = make_env(config["ENV_NAME"], config["ENV_KWARGS"])
    env = LogWrapper(env)
    
    rng = jax.random.PRNGKey(config["EVAL_SEED"])
    rng, ego_init_rng, partner_init_rng, eval_rng = jax.random.split(rng, 4)

    # Load train/test partner populations
    train_partner_agent_config = dict(config["train_partner_agent"])
    test_partner_agent_config = dict(config["test_partner_agent"])
    
    train_partner_name = list(train_partner_agent_config.keys())[0]
    test_partner_name = list(test_partner_agent_config.keys())[0]
    train_partner_agent_config = list(train_partner_agent_config.values())[0]
    test_partner_agent_config = list(test_partner_agent_config.values())[0]

    train_partner_policy, train_partner_params, init_train_partner_params, idx_labels = initialize_rl_agent_from_config(
        train_partner_agent_config, train_partner_name, env, partner_init_rng)
    test_partner_policy, test_partner_params, init_test_partner_params, idx_labels = initialize_rl_agent_from_config(
        test_partner_agent_config, test_partner_name, env, partner_init_rng)

    train_flattened_partner_params = jax.tree.map(lambda x, y: x.reshape((-1,) + y.shape), train_partner_params, init_train_partner_params)
    test_flattened_partner_params = jax.tree.map(lambda x, y: x.reshape((-1,) + y.shape), test_partner_params, init_test_partner_params)
    train_pop_size = jax.tree.leaves(train_flattened_partner_params)[0].shape[0]
    test_pop_size = jax.tree.leaves(test_flattened_partner_params)[0].shape[0]

    train_partner_population = AgentPopulation(
        pop_size=train_pop_size,
        policy_cls=train_partner_policy
    )
    test_partner_population = AgentPopulation(
        pop_size=test_pop_size,
        policy_cls=test_partner_policy
    )

    # Optional: replace test partner population with nested IQL child checkpoints.
    use_iql_child_partners = config.get("agent_model", {}).get("use_iql_child_partners", False)
    if use_iql_child_partners:
        action_dim = env.action_space(env.agents[1]).n
        default_iql_dir = Path(__file__).resolve().parents[1] / "checkpoints" / "lbf"
        test_iql_dir = config.get("agent_model", {}).get("test_iql_checkpoint_dir", str(default_iql_dir))
        test_iql_num_parents = int(config.get("agent_model", {}).get("iql_num_parents", test_pop_size))

        nested_test_population = _load_nested_iql_population(
            base_dir=Path(test_iql_dir),
            num_parents=test_iql_num_parents,
            action_dim=action_dim,
            apply_obs_norm=bool(config.get("agent_model", {}).get("use_iql_child_obs_norm", True)),
            norm_clip=float(config.get("agent_model", {}).get("iql_child_obs_norm_clip", 10.0)),
        )

        if nested_test_population is not None:
            test_partner_population = nested_test_population
            test_flattened_partner_params = None
            print(f"Using nested IQL test partners from {test_iql_dir} (parents={test_partner_population.pop_size})")
        else:
            print("Warning: requested nested IQL test partners, but none were loaded. Falling back to default test partner population.")

    # Load best-response population
    ego_agent_config = dict(config["ego_agent"])
    
    ego0_name = list(ego_agent_config.keys())[0]
    ego0_agent_config = list(ego_agent_config.values())[0]
    ego_policy, ego_params, init_ego_params, idx_labels = initialize_rl_agent_from_config(
        ego0_agent_config, ego0_name, env, ego_init_rng)

    flattened_ego_params = jax.tree.map(lambda x, y: x.reshape((-1,) + y.shape), ego_params, init_ego_params)
    pop_size = jax.tree.leaves(flattened_ego_params)[0].shape[0]

    ego_population = AgentPopulation(
        pop_size=pop_size,
        policy_cls=ego_policy
    )

    # Create LLM client if needed
    llm_agents = ['llm_zero', 'llm_cot', 'llm_few', 'llm_ip', 'bayestom', 'recollab', 'collab']
    llm_client = None
    if agent_type in llm_agents:
        llm_client = ChatClient(
            model_name=config["agent_model"]["model_name"],
        )

    # Handle different modes
    if mode == 'probe':
        print("Running probe experiment...")
        results = probe(
            config, env, eval_rng, num_eval_episodes, 
            ego_population, flattened_ego_params, 
            test_partner_population, test_flattened_partner_params,
            probe_lengths=probe_lengths
        )
        return results

    elif mode == 'sweep_alpha':
        print("Running alpha sweep experiment...")
        results = sweep_alpha(
            config, env, eval_rng, num_eval_episodes, 
            ego_population, flattened_ego_params, 
            train_partner_population, train_flattened_partner_params,
            alpha_values=alpha_values
        )
        return results

    # For 'learn' and 'evaluate' modes
    print(f"Creating ego agent of type '{agent_type}'...")
    ego_agent = create_ego_agent(
        agent_type, config, ego_population, train_partner_population,
        train_flattened_partner_params, env, ego_init_rng, llm_client, static_idx
    )

    # Learning phase
    if mode == 'learn':
        print("Running learning phase...")
        learn(
            config, env, eval_rng, num_eval_episodes, 
            ego_agent, flattened_ego_params, 
            train_partner_population, train_flattened_partner_params
        )
        return None
    
    # elif hasattr(ego_agent, "learn") and mode == 'evaluate':
    #     # Optionally run learning before evaluation
    #     print("Learning from prior experience...")
    #     learn(
    #         config, env, eval_rng, num_eval_episodes, 
    #         ego_agent, flattened_ego_params, 
    #         train_partner_population, train_flattened_partner_params
    #     )

    # Run evaluation (with optional timestep data collection)
    collect_timestep_data = config.get("agent_model", {}).get("collect_timestep_data", False)
    print(f"Evaluating {agent_type} agent... (collect_timestep_data={collect_timestep_data})")
    
    eval_output = evaluate(
        config=config,
        env=env,
        rng=eval_rng,
        num_episodes=num_eval_episodes,
        ego_agent=ego_agent,
        ego_params=flattened_ego_params,
        partner_population=test_partner_population,
        partner_params=test_flattened_partner_params,
        collect_timestep_data=collect_timestep_data
    )
    
    # Unpack results (may include timestep data)
    if collect_timestep_data:
        results, timestep_data = eval_output
    else:
        results = eval_output
        timestep_data = None

    # Compute and display metrics
    all_returns = []
    all_accuracy = []
    all_nlls = []
    all_entropies = []

    for result in results:
        all_returns.append(result["returned_episode_returns"][0])
        all_accuracy.append(result["accuracy"])

        if "nll" in result:
            all_nlls.append(result["nll"])
        if "entropy" in result:
            all_entropies.append(result["entropy"])

    avg_return = np.mean(all_returns)
    avg_accuracy = np.mean(all_accuracy)
    avg_nll = np.mean(all_nlls) if all_nlls else None
    avg_entropy = np.mean(all_entropies) if all_entropies else None

    if print_metrics:
        print(f"\n{'='*60}")
        print(f"EVALUATION RESULTS")
        print(f"{'='*60}")
        print(f"Agent: {agent_type}")
        print(f"Episodes: {num_eval_episodes}")
        print(f"Average return: {avg_return:.3f}")
        print(f"Average accuracy: {avg_accuracy:.3f}")
        if avg_nll is not None:
            print(f"Average NLL: {avg_nll:.3f}")
        if avg_entropy is not None:
            print(f"Average entropy: {avg_entropy:.3f}")
        print(f"{'='*60}\n")

    # Save results
    df = pd.DataFrame(results)

    results_dir = os.path.join(
        str(config["results_path"]),
        str(config["agent_model"]["model_name"]),
        str(config["ENV_NAME"]),
        ego_agent.method
    )

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    results_path = os.path.join(results_dir, "results.csv")
    df.to_csv(results_path, index=False)
    if print_metrics:
        print(f"Results saved to {results_path}")
    
    # Save timestep data if collected
    if timestep_data is not None:
        timestep_results_path = os.path.join(results_dir, "timestep_results.pkl")
        with open(timestep_results_path, 'wb') as f:
            pickle.dump(timestep_data, f)
        if print_metrics:
            print(f"Timestep data saved to {timestep_results_path}")

    return results
