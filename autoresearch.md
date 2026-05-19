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

### Phase 2: Large Env (12×12, 6 food, different_levels=true) — COMPLETE ✓

**Context**: `DifferentLevelsGenerator` produces [1,1,2,2,2,2] food levels (2 solo-collectible + 4 coop-required).
Time limit: 100 steps (Jumanji default). Obs dim: 24. Max theoretical SP ≈ 0.833.

**FINAL BEST (Run 43)**: `LR=1e-4, TOLERANCE=0.2, LAGRANGE_LR=0.0001, NUM_ENVS=192, 350M steps`
- composite=0.223, mean_jsd=0.510, min_jsd=0.315, mean_sp=0.438, min_sp=0.398
- n_collapsed=0, mean_agreement=0.355, max_agreement=0.648
- All 6 teams at 48–56% of theoretical max SP — remarkably balanced

**Key large-env findings**:
- **LR is the primary SP lever**: 5e-4→3e-4→2e-4→1e-4 each ~20-24% SP improvement; slower actor LR lets SP mature before diversity pressure
- **LAGRANGE_LR scales with task difficulty**: small env=0.001, large env=0.0001
- **NUM_ENVS=192 optimal**: 256 causes over-convergence and behavioral clustering
- **TOLERANCE=0.2 breaks two-cluster trap**: at TOL=0.1+LR=2e-4 teams form "go-down" vs "go-up" groups (min_jsd=0.215); TOL=0.2 prevents this
- **TOLERANCE=0.3 causes re-clustering**: non-monotonic — 0.2 is the sweet spot
- **350M is the sweet spot for final config**: 300M still improving (+8%), 400M regresses (Lagrange overconstrained SP)
- **JSD and SP can improve together** up to the sweet spot: 250M→350M, composite 0.200→0.223 (+11.5%)

**Settled config** (use for Overcooked/Hanabi as starting point):
```yaml
LR: 1.0e-4
TOLERANCE_FACTOR: 0.2
LAGRANGE_LR: 0.0001
NUM_ENVS: 192
TOTAL_TIMESTEPS: 350_000_000  # adjust per env difficulty
```
