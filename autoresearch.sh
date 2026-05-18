#!/usr/bin/env bash
# Autoresearch experiment runner for LBRDiv 6-team population training.
# Outputs METRIC lines for autoresearch loop parsing.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
CONFIG="$REPO_ROOT/autoresearch_experiment.yaml"

# REPO_PATH used by save_load_utils.py is src/bayes_tom/ (2 dirs up from utils/)
INTERNAL_REPO="$REPO_ROOT/src/bayes_tom"
INTERNAL_OUTDIR="$INTERNAL_REPO/autoresearch_outputs/lbrdiv_lbf_exp"

# Remove old outputs so find always gets the right (freshly trained) checkpoint
rm -rf "$INTERNAL_OUTDIR"

cd "$REPO_ROOT"

echo "=== Starting LBRDiv training ==="
uv run bayes-tom train-lbrdiv "$CONFIG"
echo "=== Training complete ==="

# Find the checkpoint produced by this run
LATEST=$(find "$INTERNAL_OUTDIR" -name "saved_train_run" -type d 2>/dev/null | sort | tail -1)
if [ -z "$LATEST" ]; then
    echo "ERROR: No checkpoint found in $INTERNAL_OUTDIR" >&2
    exit 1
fi
echo "Evaluating checkpoint: $LATEST"

# Run diagnostics with machine-readable metric output
uv run python scripts/diagnose_diversity.py "$LATEST" \
    --env lbf --alg lbrdiv --n-eps 20 --n-probe 256 --machine-readable
