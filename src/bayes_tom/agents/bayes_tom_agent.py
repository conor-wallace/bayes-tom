import glob
import os
import re
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import jax
import jax.numpy as jnp

from bayes_tom.agents.llm import LLMClient
from bayes_tom.agents.behavior_model import KNNBehaviorModel


def parse_teammate_scores(text: str, num_partners: int = 4) -> Dict[str, float]:
    """
    Parse LLM-ToM inverse-planning output.

    Expected JSON format:
    {
      "log_scores": {
        "Partner 1": <number>,
        "Partner 2": <number>,
        ...
      }
    }

    Returns:
        Dict[str, float] mapping "Partner k" -> score
    """

    # 1) Try strict JSON parse
    try:
        obj = json.loads(text.strip())
    except json.JSONDecodeError:
        # Fallback: extract first JSON object from text
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            raise ValueError(f"Could not find JSON object in LLM output: {text!r}")
        obj = json.loads(m.group(0))

    # 2) Validate structure
    if "log_scores" in obj:
        if not isinstance(obj["log_scores"], dict):
            raise ValueError(f"Missing or invalid 'log_scores' field in LLM output: {obj!r}")
        score_field = "log_scores"
    elif "log_likelihoods" in obj:
        if not isinstance(obj["log_likelihoods"], dict):
            raise ValueError(f"Missing or invalid 'log_likelihoods' field in LLM output: {obj!r}")
        score_field = "log_likelihoods"
    else:
        raise ValueError(f"Response is not in the expected format")

    scores = {}
    for k in range(1, num_partners + 1):
        key = f"Partner {k}"
        if key not in obj[score_field]:
            # LLM didn't return score for this partner - use low default
            print(f"Warning: Missing score for {key} in LLM output, using default value -10.0")
            scores[key] = -10.0
        else:
            try:
                scores[key] = float(obj[score_field][key])
            except (TypeError, ValueError):
                raise ValueError(
                    f"Non-numeric score for {key}: {obj[score_field][key]!r}"
                )

    return scores


class BayesToMAgent:
    """
    Wraps your ego AgentPopulation and selects a policy index by querying an LLM
    once after probing.

    Assumptions:
      - ego_population.get_actions(pop_params, agent_indices, ...) works like your partner_population
      - pop_params is shaped [pop_size, ...] pytree
      - agent_indices is shape (batch,) (your code uses jnp.array([idx]))
    """
    def __init__(
        self,
        *,
        config,
        ego_population,
        partner_population,
        partner_params,
        llm: LLMClient,
        eta: float = 0.5,
        alpha: float = 0.9
    ):
        self.method = "hybrid"
        self.population = ego_population
        self.partner_population = partner_population
        self.partner_params = partner_params
        self.llm = llm
        self.teammate_types = [f"teammate_{p}" for p in range(ego_population.pop_size)]
        self.temperature = float(config["agent_model"]["temperature"])
        self.default_partner_idx = jnp.array([int(config["agent_model"]["default_idx"])])
        self.pred_partner_idx = self.default_partner_idx
        self.is_learning = False
        self.probe_length = int(config["agent_model"]["probe_steps"])
        self.eta = config["agent_model"]["eta"]
        self.alpha = config["agent_model"]["alpha"]
        self.delta = 5
        self.use_oracle_likelihood = config["agent_model"]["oracle_likelihood"]
        self.posterior = None

        self.summary_dir = os.path.join(
            str(config["agent_model"]["summary_path"]),
            str(config["agent_model"]["model_name"]),
            str(config["ENV_NAME"]),
            "probe_" + str(self.probe_length),
        )

        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)

        if str(config["ENV_NAME"]) == "lbf":
            from .features.lbf import LBFBehaviorExtractor as BehaviorExtractor
            from .prompts.lbf import lbf_ip_prompt as ip_prompt
            from .prompts.lbf import lbf_system_prompt as system_prompt
        elif str(config["ENV_NAME"]) == "hanabi":
            from .features.hanabi import HanabiBehaviorExtractor as BehaviorExtractor
            from .prompts.hanabi import hanabi_ip_prompt as ip_prompt
            from .prompts.hanabi import hanabi_system_prompt as system_prompt
        elif "overcooked-v1" in str(config["ENV_NAME"]):
            from .features.overcooked import OvercookedBehaviorExtractor as BehaviorExtractor
            from .prompts.overcooked import overcooked_ip_prompt as ip_prompt
            from .prompts.overcooked import overcooked_system_prompt as system_prompt

        self.extractor = BehaviorExtractor()
        self.ip_prompt = ip_prompt
        self.system_prompt = system_prompt

        self.behavior_model_dir = os.path.join(
            str(config["agent_model"]["behavior_model_path"]),
            str(config["ENV_NAME"]),
        )

        self.behavior_models = {}
        for partner_idx in range(self.partner_population.pop_size):
            behavior_model_path = os.path.join(self.behavior_model_dir, f"partner_{partner_idx}_behavior_model.pkl")
            if os.path.exists(behavior_model_path):
                self.behavior_models[partner_idx] = KNNBehaviorModel.load(behavior_model_path)
            else:
                print(f"Warning: no behavior model found for partner {partner_idx} at {behavior_model_path}. Posterior updates will be uniform.")
                self.behavior_models[partner_idx] = None

    def reset(self):
        self.pred_partner_idx = self.default_partner_idx
        self.posterior = None

        self.behavior_models = {}
        for partner_idx in range(self.partner_population.pop_size):
            behavior_model_path = os.path.join(self.behavior_model_dir, f"partner_{partner_idx}_behavior_model.pkl")
            if os.path.exists(behavior_model_path):
                self.behavior_models[partner_idx] = KNNBehaviorModel.load(behavior_model_path)
            else:
                print(f"Warning: no behavior model found for partner {partner_idx} at {behavior_model_path}. Posterior updates will be uniform.")
                self.behavior_models[partner_idx] = None

    def init_hstate(self, batch_size=1, aux_info=None):
        return self.population.policy_cls.init_hstate(
            batch_size, aux_info={"agent_id": 0}
        )

    def learn(
        self,
        trace,
        partner_idx: int = 0,
        episode_idx: int = 0,
    ):
        if self.is_learning:
            timestep = len(trace.partner_obs)
            if timestep == 0:
                return

            summary_path = os.path.join(
                self.summary_dir,
                f"parter_{partner_idx}_eps_{episode_idx}_time_{timestep}.txt"
            )

            probe_text = self.generate_history(trace)

            with open(summary_path, "w") as f:
                f.write(probe_text)

    def generate_history(self, trace):
        fingerprint = self.extractor.extract_fingerprint(
            observations=trace.partner_obs,
            actions=[jnp.array([ego_act, ptr_act]) for ego_act, ptr_act in zip(trace.ego_actions, trace.partner_actions)],
            rewards=[jnp.array([ego_rew, ptr_rew]) for ego_rew, ptr_rew in zip(trace.ego_rewards, trace.partner_rewards)],
            probe_length=self.probe_length
        )

        fp_text = self.extractor.describe_fingerprint(fingerprint)

        return fp_text

    def load_past_examples(self, trace):
        curr_timestep = len(trace.partner_obs)

        summary_files = glob.glob(os.path.join(self.summary_dir, "*.txt"))

        # partner_idx -> list of (timestep, filepath)
        candidates: Dict[int, List[Tuple[int, str]]] = {}

        for summary_file in summary_files:
            summary_name = Path(summary_file).stem
            parts = summary_name.split("_")

            partner_idx = int(parts[1]) + 1
            episode_idx = int(parts[3])
            timestep = int(parts[5])

            candidates.setdefault(partner_idx, []).append((timestep, summary_file))

        # Pick the best match per partner
        chosen_files: List[Tuple[int, str]] = []  # (partner_idx, filepath)
        for partner_idx, items in sorted(candidates.items(), key=lambda x: x[0]):
            # sort by timestep ascending
            items = sorted(items, key=lambda t: t[0])

            # 1) best timestep <= curr_timestep
            leq = [it for it in items if it[0] <= curr_timestep]
            if leq:
                best = leq[-1]  # max timestep <= curr_timestep
            else:
                best = items[-1]  # fallback: max timestep overall

            chosen_files.append((partner_idx, best[1]))

        # Load summaries in partner order
        past_examples: List[str] = []
        for partner_idx, filepath in chosen_files:
            # print(f"Using file {filepath} for partner {partner_idx}")
            with open(filepath, "r") as f:
                partner_summary = f.read()

            past_examples.append(f"Partner {partner_idx}")
            past_examples.append(partner_summary)
            past_examples.append("")

        return "\n".join(past_examples)

    def linear_opinion_pool(self, llm_scores, bayes_scores):
        return self.alpha * llm_scores + (1 - self.alpha) * bayes_scores

    def bayes_update(self, scores):
        M = self.partner_population.pop_size
        eps = 1e-12

        if self.posterior is None:
            self.posterior = jnp.ones((M,)) / M

        posterior = []
        for partner_idx in range(M):
            p = jnp.clip(jnp.asarray(scores[partner_idx]), eps, 1.0)

            loss = 1.0 - p
            weight = 1.0 - self.eta * loss
            posterior.append(self.posterior[partner_idx] * weight)

        posterior = jnp.array(posterior)
        self.posterior = posterior / (jnp.sum(posterior) + eps)
        return self.posterior.copy()

    def knn_update(self, trace):
        t = len(trace.partner_actions)
        M = self.partner_population.pop_size
        eps = 1e-12

        # if self.posterior is None:
        #     self.posterior = jnp.ones((M,)) / M

        # if t == 0:
        #     return self.posterior

        partner_action = jnp.array(trace.partner_actions[t - 1])
        partner_obs = jnp.array(trace.partner_obs[t - 1])

        knn_scores = []
        for partner_idx in range(M):
            score = self.behavior_models[partner_idx].likelihood(partner_obs, partner_action)
            score = jnp.clip(jnp.asarray(score), eps, 1.0)
            knn_scores.append(score)

        knn_scores = jnp.array(knn_scores)
        knn_scores = jax.nn.softmax(knn_scores)

        return knn_scores

    def llm_update(self, trace, true_idx):
        probe_text = self.generate_history(trace)

        past_examples = self.load_past_examples(trace)

        user_prompt = self.ip_prompt.format(EXAMPLES=past_examples, PROBE_TEXT=probe_text)

        response = self.llm.chat(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
        )

        try:
            scores = parse_teammate_scores(response, num_partners=self.partner_population.pop_size)
            # print(f"Parsed LLM scores: {scores}")
        except ValueError as e:
            # If parsing fails completely, fall back to uniform distribution
            print(f"Warning: Failed to parse LLM response, using uniform distribution. Error: {e}")
            scores = {f"Partner {i}": 0.0 for i in range(1, self.partner_population.pop_size + 1)}

        llm_scores = jnp.array([s for s in scores.values()])
        llm_scores = jax.nn.softmax(llm_scores)

        return llm_scores

    def classify_behavior(self, trace, true_idx):
        curr_timestep = len(trace.partner_obs)

        if curr_timestep == 0:
            return

        if curr_timestep % self.delta != 0:
            return

        if self.alpha == 0:
            llm_scores = jnp.zeros((self.partner_population.pop_size,))
        else:
            llm_scores = self.llm_update(trace, true_idx)

        if self.alpha == 1:
            knn_scores = jnp.zeros((self.partner_population.pop_size,))
        else:
            knn_scores = self.knn_update(trace)

        scores = self.linear_opinion_pool(llm_scores, knn_scores)
        posterior = self.bayes_update(scores)

        # print(f"LLM scores: {llm_scores}")
        # print(f"KNN scores: {knn_scores}")
        # print(f"Posterior: {posterior}")

        eps = 1e-8
        self.nll = -jnp.log(jnp.clip(posterior[true_idx], eps, 1.0)).squeeze()
        self.entropy = -jnp.sum(posterior * jnp.log(jnp.clip(posterior, eps, 1.0)))
        self.pred_partner_idx = jnp.array([jnp.argmax(posterior)], dtype=jnp.int32)

        # print(f"NLL: {self.nll:.4f}")


    def get_action(self, params, partner_indices, obs, done, avail_actions, hstate, rng,
                   aux_obs=None, env_state=None, test_mode=False, trace=None):

        # New method
        if self.is_learning:
            partner_idx = int(partner_indices[0])
            self.learn(trace, partner_idx, episode_idx=trace.episode_idx)

        if not self.is_learning:
            self.classify_behavior(trace, partner_indices)

        # print(f"Selected partner idx: {self.pred_partner_idx[0]} (true idx: {partner_indices[0]})")

        return self.population.get_actions(params, self.pred_partner_idx, obs, done, avail_actions,
                                           hstate, rng, env_state, aux_obs, test_mode=True)
