#!/bin/bash
set -euo pipefail

RFAB="$HOME/RFantibody"
RF2DIR="/home/schrogpu/Documents/RFantibody_outputs/12bn_campaign/06_rf2"

cd "$RFAB"

shopt -s nullglob
RF2_QUIVERS=("$RF2DIR"/*_rf2.qv)

if [ "${#RF2_QUIVERS[@]}" -eq 0 ]; then
    echo "ERROR: No RF2 Quiver files found in:"
    echo "$RF2DIR"
    exit 1
fi

for QUIVER in "${RF2_QUIVERS[@]}"; do
    echo "Scoring: $QUIVER"
    uv run qvscorefile "$QUIVER"

    SCORE="${QUIVER%.qv}.sc"
    if [ ! -s "$SCORE" ]; then
        echo "ERROR: RF2 scorefile was not created:"
        echo "$SCORE"
        exit 1
    fi
done