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

### Baseline Context (3-team default config @ 45M steps)
- br_0 collapsed (SP return = 0.0), br_1 and br_2 achieved 0.5
- Mean JSD = 0.385, mean agreement = 0.272 (good diversity for surviving teams)
- Key problem: collapse of one team in a 3-team population; will be worse with 6 teams

### Experiment Ideas Queue
1. **Baseline 6-team** (default params, 15M steps) — establish floor
2. **Higher ENT_COEF** (0.05) — entropy bonus should combat early collapse
3. **Lower LAGRANGE_LR** (0.005) — more stable multiplier dynamics
4. **Higher CLIP_EPS** (0.2) — standard PPO; current 0.05 may be too tight
5. **Lower TOLERANCE_FACTOR** (0.05) — less strict diversity constraint, less collapse risk
6. **ANNEAL_LR=true** — often helps convergence
7. Combinations of winners
