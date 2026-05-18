# Autoresearch: LBRDiv 6-Team Population Training

## Objective

Train a population of **6 teams** in LBF using LBRDiv (Rahman et al., AAAI 2024) such that:
1. **ALL 6 teams achieve non-zero return** (no policy collapse)
2. **Teams are behaviorally diverse** (high pairwise JSD, low action agreement)

The default 3-team config works at 45M timesteps but with 6 teams, collapse is a risk. We screen
experiments at 15M timesteps to identify promising hyperparameter settings, then can run full
45M+ runs for the best configs.

## Metrics

- **Primary**: `composite` (higher is better) = `mean_jsd × min_sp_return`
  - Naturally 0 if any team collapses (min_sp_return=0), higher with both diversity and performance
- **Secondary**: `mean_jsd`, `min_jsd`, `min_sp_return`, `mean_sp_return`, `n_collapsed`, `mean_agreement`

## How to Run

```bash
./autoresearch.sh
```

Outputs `METRIC name=number` lines at the end. Training runs for ~15M timesteps (screening budget)
then the diagnostic script evaluates the checkpoint.

## Files in Scope

- `autoresearch_experiment.yaml` — the config file modified per experiment; **the main lever**
- `scripts/diagnose_diversity.py` — diagnostic script (modified to add `--machine-readable`)

## Off Limits

- `src/bayes_tom/teammate_generation/train_lbrdiv.py` — algorithm implementation (don't change)
- `src/bayes_tom/envs/` — environment code
- `src/bayes_tom/agents/` — policy network definitions
- Any existing checkpoints in `src/bayes_tom/outputs/`

## Constraints

- Must use `PARTNER_POP_SIZE: 6` (this is the goal)
- Training must complete without OOM or crash
- Screening budget: 15M timesteps (raise if results are inconclusive)

## Key Parameter Space

| Parameter | Default | Notes |
|-----------|---------|-------|
| `LAGRANGE_LR` | 0.01 | Lagrange multiplier update rate; too high → instability |
| `TOLERANCE_FACTOR` | 0.1 | SP−XP gap requirement; higher → more diverse but harder |
| `ENT_COEF` | 0.01 | Entropy bonus; higher → prevents early collapse |
| `CLIP_EPS` | 0.05 | PPO clip; standard PPO uses 0.2 — current is very tight |
| `LR` | 5e-4 | Policy learning rate |
| `ANNEAL_LR` | false | LR annealing |
| `UPDATE_EPOCHS` | 15 | PPO epochs per rollout |
| `TOTAL_TIMESTEPS` | 15M | Screening budget (45M for final runs) |
| `TRAIN_SEED` | 42 | Change to test robustness |

## What's Been Tried

### Phase 1: Small Env (7×7, 3 food, 15M steps)
**Best config (Run 16)**: `LAGRANGE_LR=0.001, NUM_ENVS=192, CLIP_EPS=0.2` → composite=0.264
- All 6 teams at max SP=0.500, mean_jsd=0.529, min_jsd=0.181
- **Diversity ceiling**: Small env has only ~4 distinct navigation strategies (go-right, go-up, go-down-right, etc.)
  Two "go-right" teams always cluster at min_jsd≈0.18 regardless of hyperparameters.
- Critical findings:
  - `ENT_COEF > 0.01` → catastrophic: ALL 6 collapse
  - `LAGRANGE_LR=0.001` optimal for small env (default 0.01 causes 3/6 collapse)
  - `CLIP_EPS=0.2` is the biggest win (0.05 = severely tight clip, slow convergence)
  - `NUM_ENVS=192` optimal; 256 causes over-convergence and clustering
  - `TOLERANCE_FACTOR=0.1` optimal; 0.15 causes clustering, 0.2 causes 1 collapse
  - `ANNEAL_LR=true` hurts diversity (teams converge to same local optima)
  - More training hurts: 25M → diversity falls, 15M is the small env sweet spot

### Phase 2: Large Env (12×12, 6 food, different_levels=true) — ACTIVE
**Context**: User requested larger env for more behavioral diversity headroom.
`DifferentLevelsGenerator` produces [1,1,2,2,2,2] food levels — 2 solo-collectible + 4 coop-required.
Time limit: 100 steps (Jumanji default). Obs dim: 24. Max theoretical SP ≈ 0.833 (all 6 collected).

**Current large-env best (Run 24)**: `LAGRANGE_LR=0.0001, 60M steps` → composite=0.127
- mean_sp=0.250 (30% of max), min_jsd=0.386, n_collapsed=0

**Key large-env findings**:
- LAGRANGE_LR scales inversely with task difficulty: small env=0.001, large env=0.0001
- 60M steps is the sweet spot for LAGRANGE_LR=0.0001 (45M→60M improves, 75M regresses)
- min_jsd consistently 0.35–0.40 — MUCH better than small env ceiling of 0.18
- SP performance (mean_sp≈0.25) is the bottleneck — harder env needs longer to learn cooperation
- n_collapsed=0 consistently with LAGRANGE_LR≤0.0001
- `different_levels` HURTS small env (SP-XP gap→0.007, no diversity pressure); large env only
- `LAGRANGE_LR=0.0005` worse than 0.0001 at all timesteps (non-monotonic)
- More training with fast LM: 75M @ LR=0.001 → SP REGRESSES vs 45M

**Current experiment (Run 26)**: `LAGRANGE_LR=0.00005, 75M steps`
- Hypothesis: 2× slower LM shifts the SP sweet spot to 75M+, achieving mean_sp > 0.250

**Ideas to try next**:
1. LAGRANGE_LR=0.00005 + 90M steps (if 75M still trending up)
2. LAGRANGE_LR=0.00001 + 75M steps (extreme slowdown)
3. NUM_ENVS=256 in large env (more SP signal per update)
4. LR=3e-4 (slower actor learning, more stable SP convergence)
5. TOLERANCE_FACTOR=0.15–0.2 (larger tolerance → more SP headroom in hard env)
