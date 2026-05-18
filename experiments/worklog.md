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

### Run 3: NUM_ENVS=128 — composite=0.072 (DISCARD under new metric)
- Improved secondary metrics but still 3 collapsed. Under new metric would have been keep.

### Run 4: LAGRANGE_LR=0.001 — composite=0.079 (KEEP — new best)
- Timestamp: 2026-05-17 23:06
- n_collapsed=1 (down from 3!), mean_sp=0.257. Slow LMs prevent early winner lock-in.

### Run 5: LAGRANGE_LR=0.001 + NUM_ENVS=128 — composite=0.194 (KEEP — goal achieved!)
- Timestamp: 2026-05-17 23:27
- n_collapsed=0! ALL 6 TEAMS SURVIVED. mean_jsd=0.472, mean_sp=0.411, min_sp=0.167.
- Combination of slow LM (0.001) + more envs (128) eliminated all collapses.

### Run 6: +25M timesteps — composite=0.155 (DISCARD)
- Timestamp: 2026-05-18 ~00:00
- More training hurts diversity (JSD falls to 0.396). 15M is the sweet spot.

### Run 7: +CLIP_EPS=0.2 (standard PPO) — composite=0.260 (KEEP — best!)
- Timestamp: 2026-05-18 07:28
- What changed: CLIP_EPS 0.05 → 0.2 (on top of LAGRANGE_LR=0.001, NUM_ENVS=128, 15M)
- Result: composite=0.260, mean_jsd=0.525, min_sp=0.483, mean_sp=0.496, n_collapsed=0
- All 6 teams essentially at max SP return (0.48–0.50)! Highest diversity ever.
- Two "right-go" teams (br_0, br_1) are the least diverse pair (JSD=0.247) — still meaningful.
- CLIP_EPS=0.2 allows larger policy updates → faster convergence to good SP + preserved diversity.
- Next: try even slower LM (LAGRANGE_LR=0.0005), different seed, or LR tuning.

### Run 8: LAGRANGE_LR=0.0005 — composite=0.250 (DISCARD)
- Timestamp: 2026-05-18 ~08:00
- What changed: LAGRANGE_LR 0.001 → 0.0005 (on best config)
- Result: composite=0.250, mean_jsd=0.518, min_jsd=0.188, min_sp=0.450, mean_sp=0.482, n_collapsed=0
- Insight: Too slow LM — teams cluster into tight right/up groups, min_jsd=0.188 (below br_0/br_1 gap of 0.247). Even slower doesn't help.
- Next: LAGRANGE_LR=0.001 is optimal. Try seeds 0/1/2 for robustness.

### Run 9: TRAIN_SEED=0 — composite=0.232 (DISCARD)
- Timestamp: 2026-05-18 ~08:20
- What changed: TRAIN_SEED 42 → 0 (same otherwise)
- Result: composite=0.232, mean_jsd=0.541, min_jsd=0.150, min_sp=0.233, mean_sp=0.429, n_collapsed=0
- Insight: All survived but br_1/br_3 cluster to near-identical right-go (JSD=0.15). Win on seed 42 is partially seed-dependent.
- Next: Try TOLERANCE_FACTOR variants to improve min_jsd.

### Run 10: TOLERANCE_FACTOR=0.2 — composite=0.216 (DISCARD)
- Timestamp: 2026-05-18 ~08:40
- What changed: TOLERANCE_FACTOR 0.1 → 0.2
- Result: composite=0.216, mean_jsd=0.545, min_jsd=0.365, min_sp=0.0, mean_sp=0.396, n_collapsed=1
- Insight: Best diversity floor ever (min_jsd=0.365!) but br_1 collapsed. Higher tolerance gives more room for behavioral divergence but one team got squeezed out.
- Next: Try TOLERANCE_FACTOR=0.15 (middle ground).

### Run 11: TOLERANCE_FACTOR=0.15 — composite=0.189 (DISCARD)
- Timestamp: 2026-05-18 ~09:00
- What changed: TOLERANCE_FACTOR 0.1 → 0.15
- Result: composite=0.189, mean_jsd=0.421, min_jsd=0.186, min_sp=0.325, mean_sp=0.450, n_collapsed=0
- Insight: 5/6 teams all cluster to up-go, very low diversity. Non-monotonic: 0.15 is worse than both 0.1 and 0.2.
- Next: TOLERANCE_FACTOR=0.1 is the sweet spot. Try LAGRANGE_LR=0.002.

### Run 12: LAGRANGE_LR=0.002 — composite=0.241 (DISCARD)
- Timestamp: 2026-05-18 ~09:20
- What changed: LAGRANGE_LR 0.001 → 0.002
- Result: composite=0.241, mean_jsd=0.506, min_jsd=0.334, min_sp=0.458, mean_sp=0.475, n_collapsed=0
- Insight: Better diversity floor (min_jsd=0.334 > 0.247) but overall composite slightly lower. Good JSD floor suggests less clustering.
- Next: LR variations.

### Run 13: LR=1e-3 — composite=0.158 (DISCARD)
- Timestamp: 2026-05-18 ~09:40
- What changed: LR 5e-4 → 1e-3
- Result: composite=0.158, mean_jsd=0.503, min_jsd=0.321, min_sp=0.0, mean_sp=0.314, n_collapsed=1
- Insight: br_0 collapsed. Faster LR destabilizes training with 6 teams. LR=5e-4 is optimal.
- Next: Try ROLLOUT_LENGTH=256 for more stable gradients.

### Run 14: ROLLOUT_LENGTH=256 — composite=0.228 (DISCARD)
- Timestamp: 2026-05-18 ~10:00
- What changed: ROLLOUT_LENGTH 128 → 256
- Result: composite=0.228, mean_jsd=0.457, min_jsd=0.263, min_sp=0.500, mean_sp=0.500, n_collapsed=0
- Insight: Remarkable SP performance — ALL 6 teams at max return (0.50)! But less diversity (JSD drops).
  Longer rollouts → more stable gradients → better convergence, but at cost of diversity pressure.
  SP-XP gap narrower (0.069 vs 0.144), meaning diversity signal is weaker.
- Next: ANNEAL_LR=true or NUM_ENVS=192/256.

## Key Insights

- **Signal dilution**: 6 teams × random SP sampling → each team's SP pair gets only ~1-2 envs/rollout
  with 64 envs. Fixing with NUM_ENVS=128 is necessary but not sufficient.
- **Lagrange LR is critical**: Default 0.01 causes 3/6 teams to collapse (early winner lock-in).
  Reducing to 0.001 gives LM dynamics time to stabilize before teams diverge irrecoverably.
- **ENT_COEF is HARMFUL**: High entropy bonus causes ALL 6 to collapse. Never raise it above 0.01.
- **CLIP_EPS=0.2 is the biggest win**: Standard PPO clip allows teams to converge quickly to
  good SP policies. Tight 0.05 clip slows learning and leaves weak teams unable to catch up.
  With 0.2, ALL 6 teams reached near-optimal SP (0.48-0.50) with high diversity (JSD=0.525).
- **More timesteps hurt diversity**: At 25M steps teams converge to similar behaviors. 15M is sweet spot.
- **TOLERANCE_FACTOR=0.1 optimal**: 0.15 causes clustering, 0.2 gives best min_jsd but causes 1 collapse.
- **LR=5e-4 optimal**: Higher 1e-3 destabilizes with 6 teams.
- **ROLLOUT_LENGTH tradeoff**: 256 → all 6 at max SP (0.50!) but lower JSD (0.457 vs 0.525). 128 is sweet spot.
- **Best config (Run 7)**: LAGRANGE_LR=0.001 + NUM_ENVS=128 + CLIP_EPS=0.2 + ROLLOUT_LENGTH=128 + 15M steps.

## Next Ideas

1. ANNEAL_LR=true — annealing often helps final convergence quality; not tried
2. NUM_ENVS=192 or 256 — further reduces SP sampling variance; not tried
3. NUM_MINIBATCHES=8 — more gradient updates per rollout with same data
4. UPDATE_EPOCHS=20 — more epochs per rollout (currently 15)
5. LAGRANGE_LR=0.002 + CLIP_EPS=0.3 — combining better diversity floor with more aggressive clipping

---
