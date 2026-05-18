def get_metric_names(env_name: str) -> tuple:
    """Return the episode return metric names for a given environment."""
    metric_map = {
        "lbf": ("returned_episode_returns",),
        "lbf-reward-shaping": ("returned_episode_returns",),
        "overcooked-v1": ("returned_episode_returns",),
        "hanabi": ("returned_episode_returns",),
    }
    return metric_map.get(env_name, ("returned_episode_returns",))
