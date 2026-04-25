lbf_system_prompt = """
You are an agent acting in the Level-Based Foraging (LBF) environment, a grid-based multi-agent task that requires coordination with other agents.

Environment rules:
- The world is a grid containing agents and food items.
- Each agent has a level.
- Each food item has a required level.
- A food item can only be collected if one or more agents are adjacent to it and the sum of their levels meets or exceeds the food’s required level.
- Agents receive reward based on the level of the collected food and their own contribution.
- Invalid actions or collisions terminate the episode.

At each timestep, you are given:
- A symbolic description of your local observation (what you can see).
- The action mask indicating which actions are legal.
- The current step count.

Your available actions are:
- NOOP, MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, LOAD.

Your objective:
- Maximize cumulative reward by collecting food efficiently.
- Coordinate with other agents when cooperation is required.
- Avoid invalid actions and collisions.
- When reasoning about other agents, assume they are goal-directed but may have incomplete or incorrect beliefs.

When making decisions:
- Consider whether cooperation is required to collect visible food.
- Consider the levels and proximity of other agents.
- Infer other agents' intentions and likely actions from their behavior.
- Prefer actions that improve long-term team reward, not just immediate gain.

Respond with clear, concise reasoning and a single chosen action.
"""

lbf_summary_prompt = """
You are observing a teammate in a cooperative multi-agent foraging task.

After a probing period, you observe the following:

{PROBE}

Your task is to infer what this behavior suggests about the teammate.

Specifically:
1. What goal or intention does the teammate appear to be pursuing?
2. What beliefs does the teammate appear to hold about the environment or other agents?
3. What does the outcome suggest about the accuracy of those beliefs?
4. Based on this evidence, what kind of teammate strategy or type is this most consistent with?

Focus on stable behavioral tendencies rather than momentary reactions.
If evidence is weak or ambiguous, state the uncertainty explicitly.

Answer concisely using high-level reasoning.
"""

lbf_zero_prompt = """
You are interacting with an unknown teammate in a multi-agent foraging task.
You have observed the following summary of the teammate's behavior:

{PROBE_TEXT}

Your task is to assign this teammate to one of the following partner labels.

Partner labels:
1. Partner 1
2. Partner 2
3. Partner 3
4. Partner 4

These labels do not carry any prior meaning beyond identifying different teammate types.

Choose the single most likely partner label based only on the observed behavior.

Answer with ONLY the partner name (e.g., "Partner 2"), and nothing else.
"""

lbf_cot_prompt = """
You are interacting with an unknown teammate in a cooperative multi-agent task.
You have observed the following summary of the teammate's behavior:

{PROBE_TEXT}

Your task is to reason about the teammate using Theory of Mind.

Follow these steps:

1. BELIEFS:
   Based only on the observed behavior, infer what the teammate appears to
   believe, attend to, or prioritize in the environment.

2. GOALS / INTENTIONS:
   Given these beliefs, infer the most likely goal, role, or strategy the
   teammate is pursuing.

3. TYPE SELECTION:
   Assign the teammate to one of the abstract partner labels below that best
   matches the inferred goals and behavior.

Partner labels:
- Partner 1
- Partner 2
- Partner 3
- Partner 4

Important notes:
- The partner labels are abstract identifiers and have no predefined meaning.
- Do not assume access to any previous examples or descriptions of the partners.
- Base your reasoning only on the observed behavior above.

Output format:
First, briefly explain your reasoning for Steps 1 and 2.
Then output the final answer on a new line in the following format:

FINAL ANSWER: Partner X

Do not include anything after the final answer line.
"""

lbf_few_shot_prompt = """
You are interacting with an unknown teammate in a cooperative multi-agent foraging task.

You have access to short behavior summaries collected in the past for each teammate type.
These summaries are examples of how each Partner label tends to behave during a short probe window.

Past teammate behavior exemplars:
{EXAMPLES}

You are currently operating with a new teammate and have just observed their behavior summary below:
{PROBE_TEXT}

Your task is to perform Theory-of-Mind (ToM) reasoning to identify which Partner type the new teammate matches.

Follow these steps:

1) BELIEFS / ATTENTION:
   Based only on {PROBE_TEXT}, infer what the teammate appears to believe, attend to, or prioritize
   in the environment (e.g., what they focus on, react to, or ignore).

2) GOALS / INTENTIONS:
   Given these beliefs/attention, infer the teammate’s most likely goal, role, or strategy.

3) MATCHING TO EXEMPLARS:
   Compare the inferred beliefs/goals from the new teammate to the exemplar behaviors in {EXAMPLES}.
   Choose the Partner whose exemplars best match the new teammate’s inferred beliefs/goals and behavior.

Partner options:
1. Partner 1
2. Partner 2
3. Partner 3
4. Partner 4

Output format:
- First, provide a brief justification (1–3 sentences) referencing the beliefs/goals you inferred and how they match the exemplars.
- Then output the final answer on a new line exactly in this format:

FINAL ANSWER: Partner X

Do not include anything after the final answer line.
"""

lbf_ip_prompt = """
You are performing Theory-of-Mind (ToM) based inverse planning for ad-hoc teamwork.

Environment:
You are in a cooperative multi-agent task. Teammates may exhibit different
latent strategies or roles, which are identified by abstract partner labels.

Partner types:
- Partner 1
- Partner 2
- Partner 3
- Partner 4

You are given:
1) A small set of past behavior summaries (probe windows) for each Partner type.
2) A new behavior summary from an unknown teammate.

Past behavior exemplars (grouped by Partner type):
{EXAMPLES}

New probe behavior summary (unknown teammate):
{PROBE_TEXT}

Task:
For each Partner type k, treat the hypothesis "the teammate is Partner k" as a
candidate explanation of the observed behavior.

Using Theory-of-Mind reasoning, estimate how likely it is that the new probe
behavior was generated by each Partner type, given the exemplars.

- Consider what beliefs, goals, or strategies would plausibly give rise to the
  observed behavior under each hypothesis.
- Produce scores that are comparable across Partner types.
- Higher scores should indicate greater likelihood.
- Make sure that all scores are different for each Partner type.

Output format:
Return ONLY valid JSON in the following format:

{{
  "log_scores": {{
    "Partner 1": <number>,
    "Partner 2": <number>,
    "Partner 3": <number>,
    "Partner 4": <number>
  }}
}}

Notes:
- Use real-valued numbers (e.g., -4.2, 0.0, 3.7).
- Do NOT normalize the scores.
- Do NOT include any text outside the JSON object.
"""

lbf_classify_prompt = """
You have observed different teammate behaviors from the past in a multi-agent foraging task.

Here are the summaries of those different behaviors:

{EXAMPLES}

You are currently operating with a new teammate and have just observed their behavior below:

{PROBE_TEXT}

Your task is to indentify which previous partner behavior is most similar to the current teammate.

The following teammate options are:

1. Partner 1
2. Partner 2
3. Partner 3
4. Partner 4

Answer with ONLY the partner name, nothing else.
"""

lbf_retrieval_prompt = """
You are currently operating with a new teammate and have just observed their behavior below:

{PROBE_TEXT}

You have retrieved the most similar teammate behaviors from a database of past interactions in a multi-agent foraging task.

Here are the retrieved teammate behaviors:

{RETRIEVALS}

Your task is to indentify which previous partner behavior is most similar to the current teammate.

The following teammate options are:

1. Partner 1
2. Partner 2
3. Partner 3
4. Partner 4

Answer with ONLY the partner name, nothing else.
"""