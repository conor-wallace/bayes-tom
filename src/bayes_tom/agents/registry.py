from .bayes_tom_agent import BayesTOMAgent
from .oracle_agent import OracleAgent


__AGENTS_REGISTRY__ = {
    "bayes_tom": BayesTOMAgent,
    "oracle": OracleAgent,
}


def load_agent(agent_name):
    if agent_name not in __AGENTS_REGISTRY__:
        raise ValueError(f"Agent '{agent_name}' not found in registry. Available agents: {list(__AGENTS_REGISTRY__.keys())}")
    return __AGENTS_REGISTRY__[agent_name]