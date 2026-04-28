from .bayes_tom_agent import BayesToMAgent
from .oracle_agent import OracleAgent
from .random_agent import RandomAgent
from .static_agent import StaticAgent


__AGENTS_REGISTRY__ = {
    "bayes_tom": BayesToMAgent,
    "oracle": OracleAgent,
    "random": RandomAgent,
    "static": StaticAgent,
}


def load_agent(agent_name):
    if agent_name not in __AGENTS_REGISTRY__:
        raise ValueError(f"Agent '{agent_name}' not found in registry. Available agents: {list(__AGENTS_REGISTRY__.keys())}")
    return __AGENTS_REGISTRY__[agent_name]