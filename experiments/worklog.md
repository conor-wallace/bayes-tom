# Worklog: LBRDiv 6-Team Population Autoresearch

**Goal:** 6-team LBRDiv population for LBF where all teams achieve non-zero return AND have high behavioral diversity.

**Metric:** composite = mean_jsd × min_sp_return (higher is better; 0 if any team collapses)

**Screening budget:** 15M timesteps per experiment (vs 45M for production)

**Baseline context (3-team @ 45M steps):**
- 1/3 teams collapsed (br_0: SP=0.0), others at 0.5
- Mean JSD=0.385, agreement=0.272 (diverse but collapse kills composite)
- composite = 0.385 × 0.0 = 0.0 (failure)

---

### Run 1: Baseline — 6 teams default params 15M steps — composite=0.0 (KEEP/baseline)
- Timestamp: 2026-05-17 22:34
- What changed: Default config, PARTNER_POP_SIZE=6, 15M steps (screening budget)
- Result: composite=0.0, mean_jsd=0.314, min_sp=0.0, mean_sp=0.188, n_collapsed=3, agreement=0.221
- Insight: 3/6 teams (br_0, br_1, br_2) completely collapsed to near-random policies (entropy ~1.33).
  Survivors (br_3, br_4, br_5) perform 0.225–0.458. Signal dilution with 6 teams: each pair gets
  ~10 SP rollout samples/step vs ~21 with 3 teams. Collapsed teams have very high entropy, survivors have lower.
  LM dynamics likely converged to 3 winners/3 losers early in training.
- Next: Try ENT_COEF=0.05 to prevent premature collapse, or NUM_ENVS=128 for more SP signal

### Run 2: ENT_COEF=0.05 — composite=0.0 (DISCARD)
- Timestamp: 2026-05-17 22:55
- What changed: ENT_COEF 0.01 → 0.05
- Result: composite=0.0, mean_jsd=0.093 (way down!), min_sp=0.0, mean_sp=0.019, n_collapsed=6, agree=0.268
- Insight: ALL 6 teams collapsed. High entropy coef prevents learning entirely. All policies output
  near-uniform distributions (entropy ~1.41), creating similar noise policies with very low JSD.
  ENT_COEF fights convergence, not collapse.
- Next: ENT_COEF is bad. Try NUM_ENVS=128 (more SP signal) or lower LAGRANGE_LR.

## Key Insights

- **Signal dilution is a major problem**: with 6 teams, each team's SP pair gets ~10 envs/rollout
  vs ~21 with 3 teams. The Lagrange dynamics may converge to a 3-winner/3-loser regime early.
- **Collapsed teams have high entropy** (~1.33) indicating near-uniform random policies.
- **ENT_COEF is HARMFUL here**: High entropy bonus causes ALL teams to collapse. The collapsed
  policies already have high entropy - incentivizing entropy makes things worse, not better.
  Do NOT try ENT_COEF > 0.01 for this problem.

## Next Ideas

1. NUM_ENVS=128 — double envs to restore SP signal per team
2. LAGRANGE_LR=0.005 — slower LM updates to prevent early lock-in
3. CLIP_EPS=0.2 — standard PPO clip (default 0.05 is very conservative)
4. TOTAL_TIMESTEPS=25M — more training time
5. ENT_COEF=0.001 or 0.0 — maybe less entropy helps teams specialize faster

---
