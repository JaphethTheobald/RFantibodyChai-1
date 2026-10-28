#!/bin/bash
set -euo pipefail

# Review the final 6i58 templated Chai scores without changing campaign outputs.
# Run from the RFantibody repository root:
#   ./score_6i58_final_pdbs.sh

# ============================================================
# EDITABLE CONFIGURATION
# ============================================================

RFAB="$HOME/RFantibody"
BASE="/home/schrogpu/Documents/RFantibody_outputs/8ol9_long_goodpae_chai"

SCORE_DIR="$BASE/06_chai_target_template"
RAW_DIR="$SCORE_DIR/long_raw"
SCORE_FILE="$SCORE_DIR/8ol9_chai_prelim_rf2.sc"
PDB_DIR="$BASE/07_final_pdbs_target_template/long"
OUTPUT_DIR="$BASE/08_filtered_PDBs_score_review"

CDR_RMSD_MAX=3
ANTIBODY_RMSD_MAX=4
INTERACTION_PAE_MAX=10
DUPLICATE_MIN_COUNT=4
DUPLICATE_KEEP_COUNT=2
DUPLICATE_PAE_DELTA=3
DUPLICATE_COMPOSITE_DELTA=0.3

# ============================================================

cd "$RFAB"

if [ ! -d "$SCORE_DIR" ]; then
    echo "ERROR: Score directory does not exist: $SCORE_DIR"
    exit 1
fi

if [ ! -d "$RAW_DIR" ]; then
    echo "ERROR: Chai raw-results directory does not exist: $RAW_DIR"
    exit 1
fi

if [ ! -d "$PDB_DIR" ]; then
    echo "ERROR: Final PDB directory does not exist: $PDB_DIR"
    exit 1
fi

PDB_COUNT="$(find "$PDB_DIR" -type f -name '*.pdb' -print | wc -l)"

if [ "$PDB_COUNT" -eq 0 ]; then
    echo "ERROR: No final PDBs found in: $PDB_DIR"
    exit 1
fi

RAW_DIR="$RAW_DIR" SCORE_FILE="$SCORE_FILE" uv run python - <<'PY'
import csv
import json
import os
from pathlib import Path

raw_dir = Path(os.environ["RAW_DIR"])
scorefile = Path(os.environ["SCORE_FILE"])
result_files = sorted(raw_dir.glob("*/chai_result.json"))
if not result_files:
    raise SystemExit(f"ERROR: No chai_result.json files found in {raw_dir}")

columns = [
    "interaction_pae", "pae", "pred_lddt",
    "target_aligned_antibody_rmsd", "target_aligned_cdr_rmsd",
    "framework_aligned_antibody_rmsd", "framework_aligned_cdr_rmsd",
    "framework_aligned_H1_rmsd", "framework_aligned_H2_rmsd",
    "framework_aligned_H3_rmsd", "framework_aligned_L1_rmsd",
    "framework_aligned_L2_rmsd", "framework_aligned_L3_rmsd", "tag",
]
rows = []
for result_file in result_files:
    result = json.loads(result_file.read_text())
    legacy = result.get("legacy")
    if not isinstance(legacy, dict):
        raise SystemExit(f"ERROR: Missing legacy scores in {result_file}")
    rows.append({column: legacy.get(column, "nan") for column in columns})

scorefile.parent.mkdir(parents=True, exist_ok=True)
with scorefile.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)
print(f"Wrote scorefile: {scorefile} ({len(rows)} designs)")
PY

SCORE_COUNT=1

echo "Scorefiles: $SCORE_COUNT"
echo "Final PDBs: $PDB_COUNT"
echo "Review output: $OUTPUT_DIR"

rm -rf "$OUTPUT_DIR"

uv run python scripts/filter_final_pdbs.py \
    "$SCORE_DIR" \
    "$PDB_DIR" \
    "$OUTPUT_DIR" \
    --cdr-rmsd-max "$CDR_RMSD_MAX" \
    --antibody-rmsd-max "$ANTIBODY_RMSD_MAX" \
    --interaction-pae-max "$INTERACTION_PAE_MAX" \
    --duplicate-min-count "$DUPLICATE_MIN_COUNT" \
    --duplicate-keep-count "$DUPLICATE_KEEP_COUNT" \
    --pae-delta "$DUPLICATE_PAE_DELTA" \
    --composite-delta "$DUPLICATE_COMPOSITE_DELTA"

echo
echo "Score report: $OUTPUT_DIR/filter_report.tsv"
if command -v column >/dev/null 2>&1; then
    column -t -s $'\t' "$OUTPUT_DIR/filter_report.tsv"
else
    cat "$OUTPUT_DIR/filter_report.tsv"
fi
