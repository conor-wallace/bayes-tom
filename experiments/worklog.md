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

### Run 15: ANNEAL_LR=true — composite=0.229 (DISCARD)
- Timestamp: 2026-05-18 ~10:20
- What changed: ANNEAL_LR false → true
- Result: composite=0.229, mean_jsd=0.458, min_jsd=0.196, min_sp=0.500, mean_sp=0.500, n_collapsed=0
- Insight: All 6 at max SP (0.50) but br_1/br_2 cluster tightly (JSD=0.196, agree=73%). LR annealing
  hurts diversity — as LR decays teams converge to same local optima.
- Next: NUM_ENVS=192 (more SP signal).

### Run 16: NUM_ENVS=192 — composite=0.264 (KEEP — small-env best!)
- Timestamp: 2026-05-18 ~10:35
- What changed: NUM_ENVS 128 → 192 (on LAGRANGE_LR=0.001, CLIP_EPS=0.2, 15M steps)
- Result: composite=0.264, mean_jsd=0.529, min_jsd=0.181, min_sp=0.500, mean_sp=0.500, n_collapsed=0
- Insight: All 6 at max SP. Highest composite on small env. More envs → less SP variance per update.
  Min_jsd=0.181 is still limited by small env's strategy ceiling (two "go-right" teams are similar).
- Next: NUM_ENVS=256 or switch to large env.

### Run 17: NUM_ENVS=256 — composite=0.219 (DISCARD)
- Timestamp: 2026-05-18 ~10:50
- What changed: NUM_ENVS 192 → 256
- Result: composite=0.219, mean_jsd=0.438, min_jsd=0.120, min_sp=0.500, mean_sp=0.500, n_collapsed=0
- Insight: Severe over-convergence — 4/6 teams in "go-up" cluster (min_jsd=0.120!). Too many envs
  → over-averaging diversity signal, teams converge to same policy.
- Next: 192 is the sweet spot. Try large env with different_levels.

---
## PHASE 2: Large Environment (12×12, 6 food, different_levels=true)
**Motivation**: Small env (7×7, 3 food) has only ~4 distinct navigation strategies. Teams hit a
behavioral diversity ceiling — two "go-right" teams always cluster (min_jsd≈0.18). Larger env has
richer strategy space with both solo-collectible and cooperation-required foods.

**Key difference**: DifferentLevelsGenerator produces [1,1,2,2,2,2] food levels (shuffled) with 2 agents
both at level 1. Solo foods (level=1) → any team can collect alone. Coop foods (level=2=combined) →
require both agents. This creates strategic variety without artificially forcing cooperation.

**Environment properties**: time_limit=100 steps (Jumanji default), 12×12 grid, obs_dim=24, fov=12.
ROLLOUT_LENGTH=128 still used (spans 1.28 episodes per rollout chunk — fine for PPO).

**Reward scale**: Solo food ≈ +1/(2×6)=0.083 per collect; coop food ≈ 2×0.083=0.167. Max if all 6
collected ≈ 0.833. Current best teams reach mean_sp≈0.25 (≈30% of max, ~1.5 food items per episode).

**Note**: different_levels HURTS the small env (Run 19: SP-XP gap=0.007 — solo foods make XP trivially
easy, eliminating LBRDiv's diversity pressure). Only works well on large env.

### Run 18: Large env crash + 12x12 force_coop 25M — composite=0.020 (CRASH + DISCARD)
- Timestamp: 2026-05-18 ~11:20
- Crashed: obs_dim mismatch (env rebuilt with default 15-dim obs vs trained 24-dim). Fixed --env-kwargs passthrough.
- Second attempt: 4/6 teams collapsed. force_coop is incompatible with DifferentLevelsGenerator (no effect).
- Next: Use different_levels properly.

### Run 19: different_levels on small env — composite=0.209 (DISCARD)
- Timestamp: 2026-05-18 ~11:45
- What changed: switched to small env (7×7, 3 food) with different_levels=true
- Result: composite=0.209, mean_sp=0.500, SP-XP gap=0.007, min_jsd=0.307
- Insight: Solo foods make ANY team easy to work with → XP returns nearly as high as SP → diversity
  pressure near-zero. Confirmation that different_levels only works as intended in large env.
- Next: Large env 45M with LAGRANGE_LR=0.001.

### Run 20: Large env 45M LAGRANGE_LR=0.001 — composite=0.088 (DISCARD)
- Timestamp: 2026-05-18 ~12:10
- What changed: 12×12 grid, 6 food, different_levels=true, 45M steps, LAGRANGE_LR=0.001, NUM_ENVS=192
- Result: composite=0.088, mean_jsd=0.518, min_jsd=0.376, min_sp=0.073, mean_sp=0.170, n_collapsed=0
- Insight: Best diversity ever (min_jsd=0.376 >> small env best 0.181). ALL 6 teams survived!
  But SP learning hasn't converged — teams still learning basics on harder 12×12 env.
  LAGRANGE_LR=0.001 (fine for small env) might be too fast for large env.
- Next: More timesteps (75M) or slower LAGRANGE_LR (0.0001).

### Run 21: Large env 75M LAGRANGE_LR=0.001 — composite=0.076 (DISCARD)
- Timestamp: 2026-05-18 ~12:40
- What changed: TOTAL_TIMESTEPS 45M → 75M (same large env, LAGRANGE_LR=0.001)
- Result: composite=0.076, mean_jsd=0.516, min_jsd=0.338, min_sp=0.063, mean_sp=0.147, n_collapsed=0
- Insight: SP regressed (0.170→0.147) with more training! Lagrange constraint fighting SP learning.
  75M with LAGRANGE_LR=0.001 is WORSE than 45M. Lagrange dynamics too fast for large env.
  LAGRANGE_LR scaling: optimal LR inversely proportional to task difficulty.
- Next: LAGRANGE_LR=0.0001 (10× slower), still 45M.

### Run 22: Large env 45M LAGRANGE_LR=0.0001 — composite=0.118 (DISCARD)
- Timestamp: 2026-05-18 ~13:20
- What changed: LAGRANGE_LR 0.001 → 0.0001
- Result: composite=0.118, mean_jsd=0.521, min_jsd=0.353, min_sp=0.182, mean_sp=0.226, n_collapsed=0
- Insight: Huge SP improvement (0.170→0.226). Slower LM lets SP converge before diversity pressure
  kicks in. SP-XP gap=0.040 (teams are specializing). Could still be training, try 60M.
- Next: 60M timesteps with LAGRANGE_LR=0.0001.

### Run 23: Large env 45M LAGRANGE_LR=0.0005 — composite=0.094 (DISCARD)
- Timestamp: 2026-05-18 ~14:00
- What changed: LAGRANGE_LR 0.0001 → 0.0005 (intermediate, 45M)
- Result: composite=0.094, mean_jsd=0.505, min_jsd=0.326, min_sp=0.100, mean_sp=0.186, n_collapsed=0
- Insight: Non-monotonic — 0.0001 beats 0.0005 on both composite and min_jsd. LAGRANGE_LR=0.0001
  is optimal for large env (10× slower than small env optimum of 0.001).
- Next: Stick to 0.0001, try 60M.

### Run 24: Large env 60M LAGRANGE_LR=0.0001 — composite=0.127 (DISCARD — large-env best)
- Timestamp: 2026-05-18 ~14:50
- What changed: TOTAL_TIMESTEPS 45M → 60M, LAGRANGE_LR=0.0001
- Result: composite=0.127, mean_jsd=0.508, min_jsd=0.386, min_sp=0.180, mean_sp=0.250, n_collapsed=0
- Insight: Both SP and diversity still improving vs 45M (mean_sp 0.226→0.250, min_jsd 0.353→0.386).
  Trend was positive — 60M is better. Best individual team: br_0 at SP=0.318 (38% of max).
  SP-XP gap=0.072 (larger than 45M's 0.040) — teams are specializing more.
- Next: 75M timesteps — check if improvement continues or reversal like small env.

### Run 25: Large env 75M LAGRANGE_LR=0.0001 — composite=0.101 (DISCARD)
- Timestamp: 2026-05-18 ~15:30
- What changed: TOTAL_TIMESTEPS 60M → 75M, LAGRANGE_LR=0.0001
- Result: composite=0.101, mean_jsd=0.496, min_jsd=0.393, min_sp=0.187, mean_sp=0.204, n_collapsed=0
- Insight: SP regressed (0.250→0.204) while min_jsd improved slightly (0.386→0.393). The Lagrange
  constraint is fighting SP convergence after 60M. 60M is the sweet spot for LAGRANGE_LR=0.0001.
  Min_jsd still at 0.393 — excellent diversity floor, but SP too weak.
- Next: Even slower LAGRANGE_LR=0.00005 with 75M — maybe the sweet spot shifts to 75M+ with slower LM.

### Run 26: Large env 75M LAGRANGE_LR=0.00005 — IN PROGRESS
- Timestamp: 2026-05-18 ~15:50
- What changed: LAGRANGE_LR 0.0001 → 0.00005 (2× slower), TOTAL_TIMESTEPS=75M
- Hypothesis: With even slower LM dynamics, the SP→diversity sweet spot shifts to 75M+.
  Teams can develop better SP policies before diversity pressure kicks in.
- Expected: mean_sp > 0.250 (current large-env best), min_jsd ≥ 0.35

---

## Key Insights

### Small Env (7×7, 3 food)
- **Signal dilution**: 6 teams × random SP sampling → each team's SP pair gets only ~1-2 envs/rollout
  with 64 envs. Fixing with NUM_ENVS=128 is necessary but not sufficient.
- **Lagrange LR is critical**: Default 0.01 causes 3/6 teams to collapse (early winner lock-in).
  Reducing to 0.001 gives LM dynamics time to stabilize before teams diverge irrecoverably.
- **ENT_COEF is HARMFUL**: High entropy bonus causes ALL 6 to collapse. Never raise it above 0.01.
- **CLIP_EPS=0.2 is the biggest win**: Standard PPO clip allows teams to converge quickly to
  good SP policies. Tight 0.05 clip slows learning and leaves weak teams unable to catch up.
- **More timesteps hurt diversity**: At 25M steps teams converge to similar behaviors. 15M is sweet spot.
- **TOLERANCE_FACTOR=0.1 optimal**: 0.15 causes clustering, 0.2 gives best min_jsd but causes 1 collapse.
- **NUM_ENVS=192 optimal**: 128 gives best results, 256 causes over-convergence and clustering.
- **Diversity ceiling**: Small env has only ~4 distinct navigation strategies. min_jsd≈0.18 is the ceiling.
  Two "go-right" teams always cluster regardless of hyperparameters.
- **Best small-env config (Run 16)**: LAGRANGE_LR=0.001 + NUM_ENVS=192 + CLIP_EPS=0.2 + 15M steps → composite=0.264

### Large Env (12×12, 6 food, different_levels=true)
- **LAGRANGE_LR scales inversely with task difficulty**: Small env optimum=0.001; large env optimum=0.0001.
  The harder env needs more time for SP learning before diversity pressure kicks in.
- **60M is the sweet spot for LAGRANGE_LR=0.0001**: Improvement from 45M→60M, regression at 75M.
  If we use LAGRANGE_LR=0.00005, the sweet spot may shift to 75M+.
- **min_jsd is MUCH better in large env**: Consistently 0.35–0.40 vs 0.18 ceiling in small env.
  More strategy space → teams find genuinely different behavioral niches.
- **SP performance is the bottleneck**: Mean SP ≈ 0.25 (30% of theoretical max). No team exceeds 0.32.
  Cooperation on a 12×12 grid requires longer navigation paths. Need more training time or slower LM.
- **n_collapsed=0 consistently**: Large env doesn't cause collapses once LAGRANGE_LR ≤ 0.0001.
- **SP-XP gap grows with LAGRANGE_LR slowing**: 0.040 at 45M vs 0.072 at 60M — teams specializing more.

## Next Ideas

1. **LAGRANGE_LR=0.00005, 90M steps** — if 75M is still learning, push further
2. **LAGRANGE_LR=0.00005, 100M steps** — max training, check if SP keeps improving
3. **LAGRANGE_LR=0.00001, 75M steps** — even slower LM, last resort
4. **NUM_ENVS=256 in large env** — more signal per update (didn't work for small env but different regime)
5. **LR=3e-4 (slower actor LR)** — may help SP convergence on harder task
6. **TOLERANCE_FACTOR=0.15 or 0.2 in large env** — larger tolerance may help teams develop SP on harder task

---
