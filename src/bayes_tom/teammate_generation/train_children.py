"""
Training script for IQL on multi-agent offline RL datasets with discrete actions.

Usage:
    python train_multiagent_iql.py task=lbf iql=default
    python train_multiagent_iql.py task=lbf iql=sparse_rewards iql.num_seeds=10

Note: Supports training multiple IQL policies in parallel with different seeds.
"""

import os
from functools import partial
from pathlib import Path
import sys
import shutil
from typing import Any, Callable, Dict, NamedTuple, Optional, Sequence, Tuple

import distrax
import flax
import flax.linen as nn
import h5py
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm
import wandb
from flax.training.train_state import TrainState
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel

from bayes_tom.agents.policies.population_interface import AgentPopulation
from bayes_tom.utils.agent_loader_from_config import initialize_rl_agent_from_config


class IQLConfig(BaseModel):
    # GENERAL
    algo: str = "IQL"
    project: str = "iql-offline-rl-multiagent"
    seed: int = 42
    env_name: str = "lbf"
    env_kwargs: Dict[str, Any] = {}
    max_steps: int = 500000
    eval_interval: int = 10000
    num_eval_episodes: int = 5
    batch_size: int = 256

    # DATA
    normalize_state: bool = True
    reward_scale: float = 1.0
    reward_bias: float = 0.0

    # NETWORK
    hidden_dims: Sequence[int] = (256, 256)
    actor_lr: float = 3e-4
    value_lr: float = 3e-4
    critic_lr: float = 3e-4
    layer_norm: bool = True

    # IQL SPECIFIC
    expectile: float = 0.7
    beta: float = 3.0
    tau: float = 0.005
    discount: float = 0.99
    clip_exp_adv: float = 100.0
    grad_clip_norm: float = 10.0

    # MULTI-AGENT SPECIFIC
    use_multiagent_dataset: bool = True
    dataset_path: str = ""
    agent_type: str = "ego_agent"
    partner_agent_config: Optional[Dict[str, Any]] = None
    action_dim: int = 6
    state_dim: int = 0

    # MULTI-SEED TRAINING
    num_seeds: int = 1
    base_seed: int = 0

    def __hash__(self):
        return hash(self.__repr__())


def default_init(scale: Optional[float] = jnp.sqrt(2)):
    return nn.initializers.orthogonal(scale)


class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: bool = False
    layer_norm: bool = False
    kernel_init: Callable = default_init()

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for i, hidden_dims in enumerate(self.hidden_dims):
            x = nn.Dense(hidden_dims, kernel_init=self.kernel_init)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                if self.layer_norm:
                    x = nn.LayerNorm()(x)
                x = self.activations(x)
        return x


class DiscreteQCritic(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False

    @nn.compact
    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        return MLP(
            (*self.hidden_dims, self.action_dim),
            layer_norm=self.layer_norm,
        )(observations)


class ValueCritic(nn.Module):
    hidden_dims: Sequence[int]
    layer_norm: bool = False

    @nn.compact
    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        value = MLP((*self.hidden_dims, 1), layer_norm=self.layer_norm)(observations)
        return jnp.squeeze(value, -1)


class DiscretePolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False

    @nn.compact
    def __call__(self, observations: jnp.ndarray, temperature: float = 1.0) -> distrax.Distribution:
        logits = MLP(
            (*self.hidden_dims, self.action_dim),
            layer_norm=self.layer_norm,
        )(observations)
        # Temperature scaling: lower temperature makes distribution sharper
        scaled_logits = logits / jnp.maximum(temperature, 1e-8)
        return distrax.Categorical(logits=scaled_logits)


class Transition(NamedTuple):
    observations: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    next_observations: jnp.ndarray
    dones: jnp.ndarray


class IQLTrainState(NamedTuple):
    critic1: TrainState
    critic2: TrainState
    target_critic1: TrainState
    target_critic2: TrainState
    value: TrainState
    actor: TrainState


def update_target_network(main_params: Any, target_params: Any, tau: float) -> Any:
    return jax.tree_util.tree_map(
        lambda p, tp: p * tau + tp * (1.0 - tau), main_params, target_params
    )


def get_multiagent_dataset(config: IQLConfig) -> Tuple[Transition, jnp.ndarray, jnp.ndarray]:
    """Load multi-agent dataset from HDF5 file."""
    assert config.dataset_path, "dataset_path must be specified"

    print(f"Loading {config.agent_type} data from {config.dataset_path}...")

    with h5py.File(config.dataset_path, "r") as f:
        agent_group = f[config.agent_type]

        observations = np.array(agent_group["observations"]) 
        actions = np.array(agent_group["actions"]).astype(np.int32).squeeze()
        rewards = np.array(agent_group["rewards"]).astype(np.float32).squeeze()
        next_observations = np.array(agent_group["next_observations"]).astype(np.float32)
        dones = np.array(agent_group["dones"]).astype(np.float32).squeeze()

    # reward transform
    rewards = rewards * config.reward_scale + config.reward_bias

    # state normalization statistics
    obs_mean = observations.mean(axis=0)
    obs_std = observations.std(axis=0) + 1e-5

    if config.normalize_state:
        observations = (observations - obs_mean) / obs_std
        next_observations = (next_observations - obs_mean) / obs_std

    dataset = Transition(
        observations=jnp.array(observations, dtype=jnp.float32),
        actions=jnp.array(actions, dtype=jnp.int32),
        rewards=jnp.array(rewards, dtype=jnp.float32),
        next_observations=jnp.array(next_observations, dtype=jnp.float32),
        dones=jnp.array(dones, dtype=jnp.float32),
    )

    # shuffle once
    rng = jax.random.PRNGKey(config.seed)
    perm = jax.random.permutation(rng, len(dataset.observations))
    dataset = jax.tree_util.tree_map(lambda x: x[perm], dataset)

    return dataset, obs_mean, obs_std


@partial(jax.jit, static_argnums=(2,))
def sample_batch(rng: jax.random.PRNGKey, dataset: Transition, batch_size: int) -> Transition:
    idx = jax.random.randint(rng, (batch_size,), 0, dataset.observations.shape[0])
    return jax.tree_util.tree_map(lambda x: x[idx], dataset)


class IQL(object):
    @classmethod
    def update(
        cls,
        train_state: IQLTrainState,
        batch: Transition,
        config: IQLConfig,
    ) -> Tuple[IQLTrainState, Dict[str, jnp.ndarray]]:
        observations = batch.observations
        actions = batch.actions
        rewards = batch.rewards
        next_observations = batch.next_observations
        dones = batch.dones

        # ----- value update -----
        q1_t = train_state.target_critic1.apply_fn(train_state.target_critic1.params, observations)
        q2_t = train_state.target_critic2.apply_fn(train_state.target_critic2.params, observations)
        q_t = jnp.minimum(q1_t, q2_t)
        q_a = jnp.take_along_axis(q_t, actions[..., None], axis=-1).squeeze(-1)

        def value_loss_fn(value_params):
            v = train_state.value.apply_fn(value_params, observations)
            diff = q_a - v
            weight = jnp.where(diff > 0, config.expectile, 1.0 - config.expectile)
            return (weight * (diff ** 2)).mean()

        value_loss, value_grad = jax.value_and_grad(value_loss_fn)(train_state.value.params)
        new_value = train_state.value.apply_gradients(grads=value_grad)

        # target for critics
        next_v = new_value.apply_fn(new_value.params, next_observations)
        target_q = rewards + config.discount * (1.0 - dones) * next_v

        # ----- critic1 update -----
        def critic1_loss_fn(critic1_params):
            q1 = train_state.critic1.apply_fn(critic1_params, observations)
            q1_a = jnp.take_along_axis(q1, actions[..., None], axis=-1).squeeze(-1)
            return jnp.mean((q1_a - target_q) ** 2)

        critic1_loss, critic1_grad = jax.value_and_grad(critic1_loss_fn)(train_state.critic1.params)
        new_critic1 = train_state.critic1.apply_gradients(grads=critic1_grad)

        # ----- critic2 update -----
        def critic2_loss_fn(critic2_params):
            q2 = train_state.critic2.apply_fn(critic2_params, observations)
            q2_a = jnp.take_along_axis(q2, actions[..., None], axis=-1).squeeze(-1)
            return jnp.mean((q2_a - target_q) ** 2)

        critic2_loss, critic2_grad = jax.value_and_grad(critic2_loss_fn)(train_state.critic2.params)
        new_critic2 = train_state.critic2.apply_gradients(grads=critic2_grad)

        # ----- actor update -----
        q1_new = new_critic1.apply_fn(new_critic1.params, observations)
        q2_new = new_critic2.apply_fn(new_critic2.params, observations)
        q_new = jnp.minimum(q1_new, q2_new)
        q_new_a = jnp.take_along_axis(q_new, actions[..., None], axis=-1).squeeze(-1)
        v_new = new_value.apply_fn(new_value.params, observations)
        adv = q_new_a - v_new
        exp_adv = jnp.minimum(jnp.exp(adv * config.beta), config.clip_exp_adv)

        def actor_loss_fn(actor_params):
            dist = train_state.actor.apply_fn(actor_params, observations)
            log_probs = dist.log_prob(actions)
            return -(exp_adv * log_probs).mean()

        actor_loss, actor_grad = jax.value_and_grad(actor_loss_fn)(train_state.actor.params)
        new_actor = train_state.actor.apply_gradients(grads=actor_grad)

        # ----- target update -----
        new_target_critic1 = train_state.target_critic1.replace(
            params=update_target_network(new_critic1.params, train_state.target_critic1.params, config.tau)
        )
        new_target_critic2 = train_state.target_critic2.replace(
            params=update_target_network(new_critic2.params, train_state.target_critic2.params, config.tau)
        )

        new_state = train_state._replace(
            critic1=new_critic1,
            critic2=new_critic2,
            target_critic1=new_target_critic1,
            target_critic2=new_target_critic2,
            value=new_value,
            actor=new_actor,
        )

        metrics = {
            "value_loss": value_loss,
            "critic1_loss": critic1_loss,
            "critic2_loss": critic2_loss,
            "actor_loss": actor_loss,
            "adv_mean": adv.mean(),
            "adv_std": adv.std(),
        }
        return new_state, metrics

    @classmethod
    def update_vectorized(
        cls,
        train_states: IQLTrainState,
        batch: Transition,
        config: IQLConfig,
    ) -> Tuple[IQLTrainState, Dict[str, jnp.ndarray]]:
        """Update multiple IQL models in parallel on the same batch."""

        def update_single(single_state):
            return cls.update(single_state, batch, config)

        new_states, metric_dict = jax.vmap(update_single)(train_states)
        return new_states, metric_dict

    @classmethod
    def get_action(
        cls,
        train_state: IQLTrainState,
        observations: jnp.ndarray,
        deterministic: bool = True,
        temperature: float = 1.0,
        seed: Optional[jax.random.PRNGKey] = None,
    ) -> jnp.ndarray:
        obs = observations
        if obs.ndim == 1:
            obs = obs[None, :]
        dist = train_state.actor.apply_fn(train_state.actor.params, obs, temperature=temperature)
        if deterministic:
            # Get logits from Categorical distribution
            logits = dist._logits if hasattr(dist, '_logits') else dist.logits
            action = jnp.argmax(logits, axis=-1)
        else:
            if seed is None:
                seed = jax.random.PRNGKey(0)
            action = dist.sample(seed=seed)
        return action

    @classmethod
    def get_action_from_ensemble(
        cls,
        train_states: IQLTrainState,
        observations: jnp.ndarray,
        model_idx: int = 0,
    ) -> jnp.ndarray:
        single_state = jax.tree_util.tree_map(lambda x: x[model_idx], train_states)
        return cls.get_action(single_state, observations, deterministic=True)


def create_optimizer(lr: float, grad_clip_norm: float):
    return optax.chain(
        optax.clip_by_global_norm(grad_clip_norm),
        optax.adam(lr),
    )


def create_iql_train_state(
    rng: jax.random.PRNGKey,
    observation_dim: int,
    action_dim: int,
    config: IQLConfig,
) -> IQLTrainState:
    rng, actor_rng, c1_rng, c2_rng, v_rng = jax.random.split(rng, 5)

    dummy_obs = jnp.zeros((1, observation_dim), dtype=jnp.float32)

    actor_model = DiscretePolicy(config.hidden_dims, action_dim=action_dim, layer_norm=config.layer_norm)
    critic1_model = DiscreteQCritic(config.hidden_dims, action_dim=action_dim, layer_norm=config.layer_norm)
    critic2_model = DiscreteQCritic(config.hidden_dims, action_dim=action_dim, layer_norm=config.layer_norm)
    value_model = ValueCritic(config.hidden_dims, layer_norm=config.layer_norm)

    actor = TrainState.create(
        apply_fn=actor_model.apply,
        params=actor_model.init(actor_rng, dummy_obs),
        tx=create_optimizer(config.actor_lr, config.grad_clip_norm),
    )
    critic1 = TrainState.create(
        apply_fn=critic1_model.apply,
        params=critic1_model.init(c1_rng, dummy_obs),
        tx=create_optimizer(config.critic_lr, config.grad_clip_norm),
    )
    critic2 = TrainState.create(
        apply_fn=critic2_model.apply,
        params=critic2_model.init(c2_rng, dummy_obs),
        tx=create_optimizer(config.critic_lr, config.grad_clip_norm),
    )
    target_critic1 = TrainState.create(
        apply_fn=critic1_model.apply,
        params=critic1_model.init(c1_rng, dummy_obs),
        tx=optax.set_to_zero(),
    )
    target_critic2 = TrainState.create(
        apply_fn=critic2_model.apply,
        params=critic2_model.init(c2_rng, dummy_obs),
        tx=optax.set_to_zero(),
    )
    value = TrainState.create(
        apply_fn=value_model.apply,
        params=value_model.init(v_rng, dummy_obs),
        tx=create_optimizer(config.value_lr, config.grad_clip_norm),
    )

    return IQLTrainState(
        critic1=critic1,
        critic2=critic2,
        target_critic1=target_critic1,
        target_critic2=target_critic2,
        value=value,
        actor=actor,
    )


def create_iql_train_states_vectorized(
    rng: jax.random.PRNGKey,
    observation_dim: int,
    action_dim: int,
    config: IQLConfig,
    num_models: int,
) -> IQLTrainState:
    rngs = jax.random.split(rng, num_models)
    return jax.vmap(
        lambda r: create_iql_train_state(r, observation_dim, action_dim, config)
    )(rngs)


def create_multiagent_env(env_name: str, config: IQLConfig):
    """Create multi-agent environment for evaluation."""
    sys.path.insert(0, str(Path(__file__).parent.parent / "jax-aht"))
    sys.path.insert(0, str(Path(__file__).parent.parent))

    try:
        from envs import make_env
        from envs.log_wrapper import LogWrapper

        env_kwargs = config.env_kwargs if hasattr(config, "env_kwargs") else {}
        env = make_env(env_name, env_kwargs)
        env = LogWrapper(env)
        return env
    except ImportError as e:
        print(f"Warning: Could not import environment: {e}")
        return None


def evaluate_multiagent_ensemble_vectorized(
    train_states: IQLTrainState,
    env,
    num_episodes: int,
    obs_mean,
    obs_std,
    agent_id: int = 0,
    partner_population=None,
    partner_params=None,
    partner_idx: int = 0,
) -> Dict[int, float]:
    """Evaluate ensemble of models by evaluating each one sequentially.
    
    Each model's action selection is JIT-compiled for fast inference.
    Returns dict mapping model_idx -> average_return
    """
    # Extract number of models from the first leaf of the vmapped params
    first_leaf = jax.tree.leaves(train_states.actor.params)[0]
    num_models = first_leaf.shape[0]
    
    if env is None:
        return {i: 0.0 for i in range(num_models)}
    
    result = {}
    
    # Evaluate each model sequentially
    for model_idx in range(num_models):
        # Extract single model from ensemble
        single_state = jax.tree_util.tree_map(lambda x: x[model_idx], train_states)
        
        # Create JIT-compiled policy function for this model
        @jax.jit
        def get_action_jitted(obs_norm):
            return IQL.get_action(single_state, obs_norm, deterministic=True)
        
        # Create wrapper that calls the JIT function
        def policy_fn(obs):
            return get_action_jitted(obs)
        
        # Evaluate this single model using sequential evaluation
        result[model_idx] = evaluate_multiagent_discrete(
            agent_policy_fn=policy_fn,
            env=env,
            num_episodes=num_episodes,
            obs_mean=obs_mean,
            obs_std=obs_std,
            agent_id=agent_id,
            partner_population=partner_population,
            partner_params=partner_params,
            partner_idx=partner_idx,
        )

    return result


def evaluate_multiagent_discrete(
    agent_policy_fn: Callable[[jnp.ndarray], jnp.ndarray],
    env,
    num_episodes: int,
    obs_mean,
    obs_std,
    agent_id: int = 0,
    partner_population=None,
    partner_params=None,
    partner_idx: int = 0,
    other_policy_fn: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
) -> float:
    """Evaluate discrete policy in multi-agent environment."""
    if env is None:
        return 0.0

    num_episodes = int(num_episodes)

    episode_returns = []
    episode_lengths = []

    for ep in range(num_episodes):
        episode_return = 0
        episode_length = 0
        rng = jax.random.PRNGKey(np.random.randint(0, 1000000))

        try:
            rng, reset_rng = jax.random.split(rng)
            obs, env_state = env.reset(reset_rng)
            done = {k: jnp.zeros((1), dtype=bool) for k in env.agents + ["__all__"]}

            agent_key = f"agent_{agent_id}"
            other_agent_key = f"agent_{1 - agent_id}"

            hstate_partner = None
            if partner_population is not None:
                other_agent_id = 1 - agent_id
                hstate_partner = partner_population.policy_cls.init_hstate(
                    1, aux_info={"agent_id": other_agent_id}
                )
                partner_idx_batched = jnp.array([partner_idx])

            t = 0
            while not done["__all__"] and t < 1000:
                avail_actions = env.get_avail_actions(env_state.env_state)
                avail_actions = jax.lax.stop_gradient(avail_actions)
                avail_actions_other = avail_actions[other_agent_key].astype(jnp.float32).reshape(1, -1)

                rng, action_rng, other_action_rng, step_rng = jax.random.split(rng, 4)

                agent_obs = obs[agent_key]
                other_obs = obs[other_agent_key]
                agent_obs_norm = (agent_obs - obs_mean) / (obs_std + 1e-5)
                print("Getting agent action...")
                print(f"    Agent obs (raw): {agent_obs}")
                print(f"    Agent obs (norm): {agent_obs_norm}")
                print(f"    Agent obs mean: {obs_mean}")
                print(f"    Agent obs std: {obs_std}")
                agent_action = int(jnp.squeeze(agent_policy_fn(obs=agent_obs_norm)))
                if partner_population is not None:
                    print("Getting partner action from population...")
                    other_action, hstate_partner = partner_population.get_actions(
                        pop_params=partner_params,
                        agent_indices=partner_idx_batched,
                        obs=other_obs.reshape(1, 1, -1),
                        done=done[other_agent_key].reshape(1, 1),
                        avail_actions=avail_actions_other,
                        hstate=hstate_partner,
                        rng=other_action_rng,
                        aux_obs=None,
                        env_state=env_state,
                        test_mode=True,
                    )
                    other_action = int(jnp.squeeze(other_action))
                elif other_policy_fn is not None:
                    print("Getting partner action from other_policy_fn...")
                    other_obs_norm = (other_obs - obs_mean) / (obs_std + 1e-5)
                    other_action = int(jnp.squeeze(other_policy_fn(obs=other_obs_norm)))
                else:
                    print("Getting partner action randomly...")
                    valid_actions = jnp.where(avail_actions_other.squeeze() > 0)[0]
                    if len(valid_actions) > 0:
                        other_action = int(jax.random.choice(other_action_rng, valid_actions))
                    else:
                        other_action = 0

                env_act = {agent_key: agent_action, other_agent_key: other_action}
                next_obs, env_state, reward, done_dict, _ = env.step(step_rng, env_state, env_act)

                obs = next_obs
                done = done_dict
                episode_return += float(jnp.squeeze(reward[agent_key]))
                episode_length += 1
                t += 1

            episode_returns.append(episode_return)
            episode_lengths.append(episode_length)

        except Exception as e:
            print(f"Warning: Episode {ep} failed: {e}")
            continue

    if len(episode_returns) == 0:
        return 0.0

    avg_return = np.mean(episode_returns)
    avg_length = np.mean(episode_lengths)
    print(f"    Eval: {num_episodes} episodes, avg return: {avg_return:.2f}, avg length: {avg_length:.1f}")
    return avg_return


@hydra.main(version_base=None, config_path="configs", config_name="config")
def train_children(config: dict):
    # Build config
    iql_params = {
        "env_name": config["task"]["ENV_NAME"],
        "env_kwargs": config["task"].get("ENV_KWARGS", {}),
        "dataset_path": config["task"]["dataset_path"],
        "agent_type": config.get("agent_type", "ego_agent"),
        "partner_agent_config": config["heldout_set"]["lbf"]["ippo"],
        "seed": config.get("seed", 42),
        "max_steps": config.get("max_steps", 500000),
        "eval_interval": config.get("eval_interval", 10000),
        "num_eval_episodes": config.get("eval_episodes", 5),
        "batch_size": config.get("batch_size", 256),
        "project": config.get("project", "iql-offline-rl-multiagent"),
        "normalize_state": config.get("normalize_state", True),
        "reward_scale": config.get("reward_scale", 1.0),
        "reward_bias": config.get("reward_bias", 0.0),
        "hidden_dims": config.get("hidden_dims", [256, 256]),
        "actor_lr": config.get("actor_lr", 3e-4),
        "value_lr": config.get("value_lr", 3e-4),
        "critic_lr": config.get("critic_lr", 3e-4),
        "layer_norm": config.get("layer_norm", True),
        "expectile": config.get("expectile", 0.7),
        "beta": config.get("beta", 3.0),
        "tau": config.get("tau", 0.005),
        "discount": config.get("discount", 0.99),
        "clip_exp_adv": config.get("clip_exp_adv", 100.0),
        "grad_clip_norm": config.get("grad_clip_norm", 10.0),
        "num_seeds": config.get("num_seeds", 1),
        "base_seed": config.get("base_seed", config.get("seed", 42)),
    }

    config = IQLConfig(**iql_params)
    num_models = config.num_seeds
    base_seed = config.base_seed

    print("=" * 60)
    print("Training IQL on Multi-Agent Discrete Action Dataset")
    print("=" * 60)
    print(f"Dataset path: {config.dataset_path}")
    print(f"Agent type: {config.agent_type}")
    print(f"Environment: {config.env_name}")
    print(f"Multi-seed training: {num_models} model(s)")
    if num_models > 1:
        print(f"Seeds: {base_seed} to {base_seed + num_models - 1}")
    print("=" * 60)

    wandb.init(project=config.project, config=config)

    rng = jax.random.PRNGKey(config.seed)
    dataset, obs_mean, obs_std = get_multiagent_dataset(config)

    config.state_dim = dataset.observations.shape[-1]
    config.action_dim = int(dataset.actions.max()) + 1

    print("\nDataset loaded successfully!")
    print(f"Number of transitions: {len(dataset.observations)}")
    print(f"State dim: {config.state_dim}")
    print(f"Action dim: {config.action_dim}")
    print(f"Reward mean/std: {dataset.rewards.mean():.4f}/{dataset.rewards.std():.4f}")

    # Load parent and partner indices from dataset metadata
    parent_idx = None
    partner_idx = None
    with h5py.File(config.dataset_path, "r") as f:
        if "specific_partner_idx" in f.attrs:
            parent_idx = int(f.attrs["specific_partner_idx"])
            partner_idx = parent_idx
            print(f"Dataset parent index: {parent_idx}")

    # Load partner population for evaluation
    partner_population = None
    partner_params = None
    if partner_idx is not None:
        try:
            print("\nLoading partner population for evaluation...")

            partner_agent_name = "agent_0"
            partner_agent_config = config.partner_agent_config

            print(f"Partner agent config: {type(partner_agent_config)}")

            partner_init_rng = jax.random.PRNGKey(config.seed + 100)

            temp_env = create_multiagent_env(config.env_name, config)
            if temp_env is not None:
                partner_policy, partner_params, init_partner_params, _ = initialize_rl_agent_from_config(
                    agent_config=partner_agent_config,
                    agent_name=partner_agent_name,
                    env=temp_env,
                    rng=partner_init_rng,
                )

                flattened_partner_params = jax.tree.map(
                    lambda x, y: x.reshape((-1,) + y.shape),
                    partner_params,
                    init_partner_params,
                )
                pop_size = jax.tree.leaves(flattened_partner_params)[0].shape[0]

                partner_population = AgentPopulation(pop_size=pop_size, policy_cls=partner_policy)
                partner_params = flattened_partner_params

                print(f"✓ Partner population loaded (size: {pop_size})")
                if partner_idx >= pop_size:
                    partner_idx = 0
                    print("  Using partner_idx 0 instead")
        except Exception as e:
            print(f"✗ Could not load partner population: {e}")

    # Create evaluation environment
    eval_env = None
    try:
        eval_env = create_multiagent_env(config.env_name, config)
        if eval_env is not None:
            print("✓ Evaluation environment created successfully")
    except Exception as e:
        print(f"✗ Could not create evaluation environment: {e}")

    # Create train state(s)
    rng, init_rng = jax.random.split(rng)
    if num_models > 1:
        train_states = create_iql_train_states_vectorized(
            init_rng,
            config.state_dim,
            config.action_dim,
            config,
            num_models,
        )
        print(f"\n{num_models} models initialized!")
    else:
        train_states = create_iql_train_state(
            init_rng,
            config.state_dim,
            config.action_dim,
            config,
        )
        print("\nModel initialized!")

    print(f"Hidden dims: {config.hidden_dims}")
    print(f"Batch size: {config.batch_size}")
    print(f"Total steps: {config.max_steps}")

    algo = IQL()
    if num_models > 1:
        update_fn = jax.jit(lambda ts, b: algo.update_vectorized(ts, b, config))
    else:
        update_fn = jax.jit(lambda ts, b: algo.update(ts, b, config))

    print("\nStarting training...")

    # Training loop
    for i in tqdm.tqdm(range(1, config.max_steps + 1), smoothing=0.1, dynamic_ncols=True):
        metrics = {"step": i}

        rng, batch_rng = jax.random.split(rng)
        batch = sample_batch(batch_rng, dataset, config.batch_size)

        if num_models > 1:
            train_states, info = update_fn(train_states, batch)

            # aggregate metrics
            for k, v in info.items():
                metrics[f"{k}_mean"] = float(v.mean())
                metrics[f"{k}_std"] = float(v.std())
            metrics["actor_loss_min"] = float(info["actor_loss"].min())
            metrics["actor_loss_max"] = float(info["actor_loss"].max())
        else:
            train_states, info = update_fn(train_states, batch)
            metrics.update({k: float(v) for k, v in info.items()})

        # Evaluate
        if i % config.eval_interval == 0 and eval_env is not None:
            agent_id = 0 if config.agent_type == "ego_agent" else 1

            if num_models > 1:
                print(f"\n{'='*60}")
                print(f"Evaluation at step {i} (vectorized)")
                print(f"{'='*60}")

                # Use vectorized evaluation for faster parallel evaluation
                eval_scores = evaluate_multiagent_ensemble_vectorized(
                    train_states=train_states,
                    env=eval_env,
                    num_episodes=config.num_eval_episodes,
                    obs_mean=obs_mean,
                    obs_std=obs_std,
                    agent_id=agent_id,
                    partner_population=partner_population,
                    partner_params=partner_params,
                    partner_idx=partner_idx if partner_idx is not None else 0,
                )
                
                if eval_scores:
                    metrics[f"{config.env_name}/avg_return_mean"] = float(np.mean(list(eval_scores.values())))
                    metrics[f"{config.env_name}/avg_return_std"] = float(np.std(list(eval_scores.values())))
                    metrics[f"{config.env_name}/avg_return_max"] = float(np.max(list(eval_scores.values())))
                    metrics[f"{config.env_name}/avg_return_min"] = float(np.min(list(eval_scores.values())))
                    
                    for model_idx, score in eval_scores.items():
                        print(f"  Model {model_idx} (seed {base_seed + model_idx}): {score:.2f}")
                    
                    print(
                        f"\nAggregate: mean={np.mean(list(eval_scores.values())):.2f}, "
                        f"std={np.std(list(eval_scores.values())):.2f}, "
                        f"max={np.max(list(eval_scores.values())):.2f}"
                    )
                    print(f"{'='*60}")
            else:
                try:
                    policy_fn = lambda obs: algo.get_action(train_states, obs, deterministic=True)
                    avg_return = evaluate_multiagent_discrete(
                        policy_fn=policy_fn,
                        env=eval_env,
                        num_episodes=config.num_eval_episodes,
                        obs_mean=obs_mean,
                        obs_std=obs_std,
                        agent_id=agent_id,
                        partner_population=partner_population,
                        partner_params=partner_params,
                        partner_idx=partner_idx if partner_idx is not None else 0,
                    )
                    metrics[f"{config.env_name}/avg_return"] = float(avg_return)
                    print(f"\nStep {i}, Avg Return: {avg_return:.2f}")
                except Exception as e:
                    print(f"Evaluation failed: {e}")

        wandb.log(metrics)

    print("\nTraining completed!")
    print(f"num_models = {num_models}, base_seed = {base_seed}")

    # Save final model(s)
    if num_models > 1:
        print(f"✓ Saving multi-seed models (num_models={num_models} > 1)")
        if parent_idx is not None:
            save_dir = Path(f"checkpoints/{config.env_name}/{config.agent_type}_iql_multiseed_parent_{parent_idx}")
            print(f"  Parent ID: {parent_idx}")
        else:
            save_dir = Path(f"checkpoints/{config.env_name}/{config.agent_type}_iql_multiseed")
        save_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nSaving {num_models} models...")
        for model_idx in range(num_models):
            model_save_dir = save_dir / f"seed_{base_seed + model_idx}"
            model_save_dir.mkdir(exist_ok=True)

            single_actor_params = jax.tree_util.tree_map(
                lambda x: x[model_idx],
                train_states.actor.params,
            )
            np.save(model_save_dir / "actor_params.npy", single_actor_params)
            print(f"  Model {model_idx} (seed {base_seed + model_idx}) saved to: {model_save_dir}")

        norm_stats_path = save_dir / "norm_stats.npz"
        np.savez(norm_stats_path, obs_mean=obs_mean, obs_std=obs_std)
        print(f"\nNormalization stats saved to: {norm_stats_path}")

        config_path = save_dir / "config.txt"
        with open(config_path, "w") as f:
            f.write(str(config))
        print(f"Config saved to: {config_path}")
    else:
        print(f"✓ Saving single-seed model (num_models={num_models} <= 1)")
        if parent_idx is not None:
            save_dir = Path(f"checkpoints/{config.env_name}/{config.agent_type}_iql_parent_{parent_idx}")
            print(f"  Parent ID: {parent_idx}")
        else:
            save_dir = Path(f"checkpoints/{config.env_name}/{config.agent_type}_iql")
        save_dir.mkdir(parents=True, exist_ok=True)

        actor_params_path = save_dir / "actor_params.npy"
        np.save(actor_params_path, train_states.actor.params)
        print(f"\nActor parameters saved to: {actor_params_path}")

        norm_stats_path = save_dir / "norm_stats.npz"
        np.savez(norm_stats_path, obs_mean=obs_mean, obs_std=obs_std)
        print(f"Normalization stats saved to: {norm_stats_path}")

        config_path = save_dir / "config.txt"
        with open(config_path, "w") as f:
            f.write(str(config))
        print(f"Config saved to: {config_path}")

    wandb.finish()
    print("\nDone!")


if __name__ == "__main__":
    main()
