#!/usr/bin/env bash
# =============================================================================
# colab_orchestrate.sh — Full AutoGluon Colab session lifecycle
#
# Usage (from project root):
#   bash src/colab_orchestrate.sh
#
# Prerequisites:
#   - Colab CLI installed: pip install colab-cli
#   - ADC authentication: gcloud auth application-default login
#   - Run from project root (house-prices/)
# =============================================================================
set -euo pipefail

SESSION="autogluon"
TRAIN_CSV="data/train.csv"
TEST_CSV="data/test.csv"
SCRIPT="src/run_colab_autogluon.py"
REMOTE_SUBMISSION="submissions/submission_autogluon.csv"
LOCAL_SUBMISSIONS="./submissions/"

# ── Helper: is the named session alive AND on T4 GPU? ─────────────────────────
session_is_ready() {
  local out
  out=$(colab status -s "$SESSION" 2>&1) || true
  # Output looks like: "[name] ... | Hardware: T4 | ... | Status: IDLE" (or READY)
  echo "$out" | grep -q "Hardware: T4" && echo "$out" | grep -qE "Status: (IDLE|READY)"
}

# ── 1. Pre-flight: verify local data files exist ──────────────────────────────
echo "=== [1/5] Pre-flight checks ==="
for f in "$TRAIN_CSV" "$TEST_CSV" "$SCRIPT"; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: Required file not found: $f"
    exit 1
  fi
done
echo "  ✓ $TRAIN_CSV"
echo "  ✓ $TEST_CSV"
echo "  ✓ $SCRIPT"

# ── 2. Check Session Status (Warm Session Reuse) ──────────────────────────────
echo ""
echo "=== [2/5] Checking Colab session '$SESSION' ==="
if session_is_ready; then
  echo "  ⚡ Warm session found: '$SESSION' is ACTIVE & READY."
  echo "  ⚡ Skipping session creation, data uploads, and library installation."
  SKIP_PROVISION=true
else
  echo "  ℹ  No active session found (cold start) — creating new T4 GPU session..."
  colab stop -s "$SESSION" 2>/dev/null || true   # clean up any zombie session
  colab new -s "$SESSION" --gpu T4
  echo "  ✓ Session '$SESSION' created."
  SKIP_PROVISION=false
fi

# ── 3. Upload data files (Cold start only) ────────────────────────────────────
echo ""
echo "=== [3/5] Data provisioning ==="
if [[ "$SKIP_PROVISION" == "true" ]]; then
  echo "  ⚡ Reusing uploaded data on warm VM."
else
  echo "  Uploading training and test datasets..."
  colab upload -s "$SESSION" "$TRAIN_CSV" train.csv
  echo "  ✓ Uploaded train.csv"
  colab upload -s "$SESSION" "$TEST_CSV" test.csv
  echo "  ✓ Uploaded test.csv"
fi

# ── 4. Execute training script ────────────────────────────────────────────────
echo ""
echo "=== [4/5] Running AutoGluon training on Colab ==="
colab exec -s "$SESSION" --timeout 1800 -f "$SCRIPT"
echo "  ✓ Training execution complete."

# ── 5. Download submission ────────────────────────────────────────────────────
echo ""
echo "=== [5/5] Downloading submission ==="
mkdir -p "$LOCAL_SUBMISSIONS"
colab download -s "$SESSION" "$REMOTE_SUBMISSION" "${LOCAL_SUBMISSIONS}submission_autogluon.csv"
echo "  ✓ Saved to ${LOCAL_SUBMISSIONS}submission_autogluon.csv"

# ── Keep VM Warm Notice ───────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  ⚡ VM kept warm for instant subsequent runs."
echo "  To manually stop when finished:"
echo "    colab stop -s $SESSION"
echo ""
echo "  Run QA validation with:"
echo "    python src/check_predictions.py"
echo "============================================================"
