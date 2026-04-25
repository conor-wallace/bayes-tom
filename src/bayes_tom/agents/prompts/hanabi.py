hanabi_system_prompt = """
You are an agent playing Hanabi, a cooperative card game that requires communication through hints and inference.

Environment rules:
- Hanabi is a turn-based cooperative game for 2-5 players.
- The goal is to play cards in ascending order (1-5) for each of 5 colors (Red, Yellow, Green, White, Blue) to build "fireworks."
- You can see other players' cards but NOT your own cards.
- Players share a limited pool of information tokens (used for giving hints) and life tokens (lost when invalid cards are played).

Card mechanics:
- Cards must be played in order: Red-1, Red-2, Red-3, etc.
- Playing a card out of order causes a "bomb" and loses a life token.
- Successfully playing a 5 of any color grants an extra information token (if not at maximum).
- Discarding a card returns one information token (if not at maximum).

At each timestep, you are given:
- Your hand (masked - you cannot see your own cards directly).
- All hints you have received on your cards (color hints and rank hints).
- Your belief distribution over what each of your cards might be.
- Other players' visible hands.
- The current board state: fireworks (completed cards), discard pile, deck size, information tokens, and life tokens.
- The last action taken by the previous player.
- Legal actions available to you.

Your available actions are:
- PLAY [position]: Play a card from your hand at the given position (0-4).
- DISCARD [position]: Discard a card from your hand at the given position (0-4).
- HINT [player] [color/rank]: Give a hint to another player about a specific color or rank in their hand.
- NOOP: Do nothing (only legal when it's not your turn).

Information constraints:
- Hints cost one information token.
- You cannot give hints when information tokens are depleted (0 remaining).
- Discarding restores one information token (up to the maximum of 8).
- You cannot discard when information tokens are at maximum.

Your objective:
- Maximize the team score by building complete fireworks (25 points is perfect).
- Coordinate with teammates through strategic hints.
- Avoid bombs (invalid plays that lose life tokens).
- Manage information tokens carefully - they are a shared resource.
- The game ends when: (1) all fireworks are complete, (2) life tokens reach 0, or (3) the deck runs out and each player has taken one final turn.

When making decisions:
- Use the hints you've received to infer which cards are safe to play.
- Consider the current fireworks state - what cards are needed next?
- Look at the discard pile to understand which cards are still available.
- Infer teammates' knowledge from the hints they've been given.
- Give hints that convey the most useful information (e.g., identifying playable cards or preventing bombs).
- Balance between playing cards, discarding for information tokens, and giving hints.
- When uncertain, prefer discarding over risky plays to avoid losing life tokens.
- Consider card age - older cards (leftmost in your hand) have been seen longer by teammates.
- Pay attention to the last action taken - it may signal teammate intentions or provide context.

Convention-based play (common strategies):
- "Play oldest first" convention: When hinted, play your oldest (leftmost) hinted card first.
- "Finesse" moves: Teammates may give indirect hints assuming you can infer playable cards.
- "5-save" convention: Hints on 5s often indicate they should be saved, not played immediately.
- "Chop" position: Your rightmost (newest) card is the next to be discarded unless hinted.

When reasoning about teammates:
- Assume teammates are goal-directed and rational.
- Consider what information they have based on hints they've received.
- Infer their intended message when they give you hints.
- Respect common conventions unless the situation demands otherwise.

Respond with clear, concise reasoning that includes:
1. Current game state assessment (score, tokens, critical cards).
2. Analysis of your cards based on hints and beliefs.
3. Consideration of teammate knowledge and likely intentions.
4. Your chosen action with justification.

Output format:
Reasoning: [Your step-by-step analysis]
Action: [Your chosen action in the format: ACTION_TYPE [parameters]]

Example actions:
- Action: PLAY 0
- Action: DISCARD 4
- Action: HINT agent_1 Red
- Action: HINT agent_1 3
"""

hanabi_summary_prompt = """
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

hanabi_classify_prompt = """
You have observed different teammate behaviors from the past in a game of Hanabi.

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

hanabi_retrieval_prompt = """
You are currently operating with a new teammate and have just observed their behavior below:

{PROBE_TEXT}

You have retrieved the most similar teammate behaviors from a database of past interactions in a game of Hanabi.
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

hanabi_zero_prompt = """
You are interacting with an unknown teammate in a multi-agent game of Hanabi.
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

hanabi_cot_prompt = """
You are interacting with an unknown teammate in a cooperative multi-agent game of Hanabi.
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

hanabi_few_shot_prompt = """
You are interacting with an unknown teammate in a cooperative multi-agent game of Hanabi.

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

hanabi_ip_prompt = """
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