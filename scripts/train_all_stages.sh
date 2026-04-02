#!/bin/bash
# Run all curriculum stages sequentially, stopping on gate failure.
#
# Usage:
#   ./scripts/train_all_stages.sh           # all 8 stages
#   ./scripts/train_all_stages.sh 3 6       # stages 3 through 6
#   ./scripts/train_all_stages.sh 4         # stage 4 only

set -e

PYTHON="${PYTHON:-.venv/bin/python3}"
START=${1:-1}
END=${2:-12}

echo "Training stages ${START}-${END}"
echo "Python: ${PYTHON}"
echo ""

for stage in $(seq "$START" "$END"); do
    echo "============================================================"
    echo "  Starting stage ${stage} of ${END}"
    echo "============================================================"

    ${PYTHON} scripts/train_stage.py --stage "$stage" --eval-episodes 3
    exit_code=$?

    if [ $exit_code -ne 0 ]; then
        echo ""
        echo "Stage ${stage} FAILED gate check. Stopping."
        echo "Fix issues and re-run: ${PYTHON} scripts/train_stage.py --stage ${stage} --timesteps <more>"
        exit 1
    fi

    echo ""
done

echo "============================================================"
echo "  All stages ${START}-${END} completed successfully!"
echo "============================================================"
