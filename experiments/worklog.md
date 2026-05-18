# Worklog: LBRDiv 6-Team Population Autoresearch

**Goal:** 6-team LBRDiv population for LBF where all teams achieve non-zero return AND have high behavioral diversity.

**Metric:** composite = mean_jsd × min_sp_return (higher is better; 0 if any team collapses)

**Screening budget:** 15M timesteps per experiment (vs 45M for production)

**Baseline context (3-team @ 45M steps):**
- 1/3 teams collapsed (br_0: SP=0.0), others at 0.5
- Mean JSD=0.385, agreement=0.272 (diverse but collapse kills composite)
- composite = 0.385 × 0.0 = 0.0 (failure)

---

## Key Insights

*(Updated as experiments accumulate)*

## Next Ideas

*(Updated after each experiment)*

---
