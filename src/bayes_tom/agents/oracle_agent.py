from bayes_tom.agents.policies.population_interface import AgentPopulation


class OracleAgent:
    population: AgentPopulation

    def __init__(self, population):
        self.method = "oracle"
        self.population = population

    def init_hstate(self, batch_size=1, aux_info=None):
        return self.population.init_hstate(batch_size, aux_info=aux_info)

    def reset(self):
        pass

    def get_action(self, partner_indices, obs, done, avail_actions, hstate, rng,
                   aux_obs=None, env_state=None, test_mode=False, trace=None):

        self.pred_partner_idx = partner_indices

        return self.population.get_actions(partner_indices, obs, done, avail_actions,
                                           hstate, rng, env_state, aux_obs, test_mode=True)