import jax

from bayes_tom.agents.policies.mlp_actor_critic_policy import MLPActorCriticPolicy, ActorWithDoubleCriticPolicy, \
    ActorWithConditionalCriticPolicy, PseudoActorWithDoubleCriticPolicy, \
    PseudoActorWithConditionalCriticPolicy


def initialize_mlp_agent(config, env, rng):
    """
    Initialize an MLP agent with the given config.
    """
    policy = MLPActorCriticPolicy(
        action_dim=env.action_space(env.agents[0]).n,
        obs_dim=config.get("POLICY_INPUT_DIM", env.observation_space(env.agents[0]).shape[0]),
        activation=config.get("ACTIVATION", "tanh"),
    )
    rng, init_rng = jax.random.split(rng)
    init_params = policy.init_params(init_rng)

    return policy, init_params


def initialize_actor_with_double_critic(config, env, rng):
    """Initialize an actor with double critic with the given config."""
    policy = ActorWithDoubleCriticPolicy(
        action_dim=env.action_space(env.agents[0]).n,
        obs_dim=config.get("POLICY_INPUT_DIM", env.observation_space(env.agents[0]).shape[0]),
        activation=config.get("ACTIVATION", "tanh"),
    )
    rng, init_rng = jax.random.split(rng)
    init_params = policy.init_params(init_rng)

    return policy, init_params


def initialize_pseudo_actor_with_double_critic(config, env, rng):
    """Initialize a pseudo actor with double critic with the given config."""
    policy = PseudoActorWithDoubleCriticPolicy(
        action_dim=env.action_space(env.agents[0]).n,
        obs_dim=config.get("POLICY_INPUT_DIM", env.observation_space(env.agents[0]).shape[0]),
        activation=config.get("ACTIVATION", "tanh"),
    )
    rng, init_rng = jax.random.split(rng)
    init_params = policy.init_params(init_rng)

    return policy, init_params


def initialize_actor_with_conditional_critic(config, env, rng):
    """Initialize an actor with conditional critic with the given config."""
    policy = ActorWithConditionalCriticPolicy(
        action_dim=env.action_space(env.agents[0]).n,
        obs_dim=config.get("POLICY_INPUT_DIM", env.observation_space(env.agents[0]).shape[0]),
        pop_size=config["POP_SIZE"],
        activation=config.get("ACTIVATION", "tanh"),
    )
    rng, init_rng = jax.random.split(rng)
    init_params = policy.init_params(init_rng)

    return policy, init_params


def initialize_pseudo_actor_with_conditional_critic(config, env, rng):
    """Initialize a pseudo actor with conditional critic with the given config."""
    policy = PseudoActorWithConditionalCriticPolicy(
        action_dim=env.action_space(env.agents[0]).n,
        obs_dim=config.get("POLICY_INPUT_DIM", env.observation_space(env.agents[0]).shape[0]),
        pop_size=config["POP_SIZE"],
        activation=config.get("ACTIVATION", "tanh"),
    )
    rng, init_rng = jax.random.split(rng)
    init_params = policy.init_params(init_rng)

    return policy, init_params
