#!/bin/bash
set -euo pipefail

# ============================================================
# 1JBU noX / h-NbBCII10 H3-LENGTH CAMPAIGN
#
# RFdiffusion
#   -> geometry filter
#   -> ProteinMPNN
#   -> RF2
#   -> scorefiles
#   -> final PDB extraction
# ============================================================

RFAB="$HOME/RFantibody"

TARGET="$HOME/RFantibody/1jbu_good_noLC.pdb"
RESIDUE_MAP="$HOME/RFantibody/1jbu_residue_mapping.tsv"
FRAMEWORK="$RFAB/scripts/examples/example_inputs/h-NbBCII10.pdb"

BASE="$HOME/Documents/RFantibody_outputs/1jbu_A183_mimic"

RFDIR="$BASE/04_rfdiffusion"
FILTERDIR="$BASE/04b_filtered"
MPNNDIR="$BASE/05_mpnn"
RF2DIR="$BASE/06_rf2"
FINALDIR="$BASE/07_final_pdbs"
FINAL_FILTERDIR="$BASE/08_filtered_PDBs"

# Final filtering configuration.
FINAL_CDR_RMSD_MAX=2.05
FINAL_ANTIBODY_RMSD_MAX=4
FINAL_INTERACTION_PAE_MAX=10
FINAL_DUPLICATE_MIN_COUNT=4
FINAL_DUPLICATE_KEEP_COUNT=2
FINAL_DUPLICATE_PAE_DELTA=3
FINAL_DUPLICATE_COMPOSITE_DELTA=0.3
FINAL_AFSERVER_BATCH_SIZE=30

SCRIPT_START="$(date +%s)"

format_duration () {
    local seconds="$1"
    printf '%02dh:%02dm:%02ds' \
        "$((seconds / 3600))" \
        "$(((seconds % 3600) / 60))" \
        "$((seconds % 60))"
}

report_duration () {
    local label="$1"
    local start="$2"
    local elapsed=$(($(date +%s) - start))
    printf '%s took %s\n' "$label" "$(format_duration "$elapsed")"
}

mkdir -p \
    "$RFDIR" \
    "$FILTERDIR" \
    "$MPNNDIR" \
    "$RF2DIR" \
    "$FINALDIR"

cd "$RFAB"

# ============================================================
# ENVIRONMENT
# ============================================================

export PATH="$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/.conda/envs/rfantibody-runtime/lib:${LD_LIBRARY_PATH:-}"
export DGLBACKEND=pytorch

# Known-working A5000
export CUDA_VISIBLE_DEVICES=1

# ============================================================
# CAMPAIGN SETTINGS
# ============================================================

NUM_BACKBONES=1000
SEQS_PER_BACKBONE=10
MPNN_TEMP=0.2

MINDIST_MAX=6
AVERAGEMIN_MAX=10

# HOTSPOTS="H59,H61,H90,H94,H95,H96,H251,H253," #Glycol Binding
# HOTSPOTS+="H20,H134,H135,H137,H184,H207" #Japhybinding1 interferes with TF...
# HOTSPOTS_ORIGINAL="H170,H170A,H184,H222,H225" #japhybinding3.1
HOTSPOTS_ORIGINAL="H89,H241,H250,H251,H252" #A183 mimic
# ============================================================
# TRANSLATE ORIGINAL PDB HOTSPOTS -> RFANTIBODY-SAFE NUMBERING
# ============================================================

if [ ! -s "$RESIDUE_MAP" ]; then
    echo "ERROR: Residue mapping file not found:"
    echo "$RESIDUE_MAP"
    exit 1
fi

HOTSPOTS="$(
    HOTSPOTS_ORIGINAL="$HOTSPOTS_ORIGINAL" \
    RESIDUE_MAP="$RESIDUE_MAP" \
    python3 - <<'PY'
import csv
import os
import re
import sys

mapping_file = os.environ["RESIDUE_MAP"]

requested = [
    x.strip()
    for x in os.environ["HOTSPOTS_ORIGINAL"].split(",")
    if x.strip()
]

mapping = {}

with open(mapping_file) as f:
    reader = csv.DictReader(f, delimiter="\t")

    for row in reader:
        mapping[row["old_id"]] = row["new_id"]

translated = []

for hotspot in requested:

    # Examples accepted:
    # H170
    # H170A
    # H170G
    # L95
    if not re.fullmatch(r"[A-Za-z0-9]-?\d+[A-Za-z]?", hotspot):
        print(
            f"ERROR: Invalid hotspot notation: {hotspot}",
            file=sys.stderr,
        )
        sys.exit(1)

    if hotspot not in mapping:
        print(
            f"ERROR: Hotspot {hotspot} was not found "
            f"in the original PDB mapping.",
            file=sys.stderr,
        )
        sys.exit(1)

    translated.append(mapping[hotspot])

print(",".join(translated))
PY
)"

echo
echo "Original hotspot numbering:"
echo "  $HOTSPOTS_ORIGINAL"

echo
echo "RFantibody hotspot numbering:"
echo "  $HOTSPOTS"
echo

REGIMES=(
    "long|H1:7,H2:6,H3:11-15"
)


# ============================================================
# BASIC CHECKS
# ============================================================

PREFLIGHT_START="$(date +%s)"

echo
echo "============================================================"
echo "PREFLIGHT"
echo "============================================================"

if [ ! -s "$TARGET" ]; then
    echo "ERROR: Target not found:"
    echo "$TARGET"
    exit 1
fi

if [ ! -s "$FRAMEWORK" ]; then
    echo "ERROR: Framework not found:"
    echo "$FRAMEWORK"
    exit 1
fi

uv run python - <<'PY'
import torch

print("CUDA available:", torch.cuda.is_available())
print("Visible GPU count:", torch.cuda.device_count())

if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA is unavailable")

print("GPU:", torch.cuda.get_device_name(0))
PY

report_duration "Preflight" "$PREFLIGHT_START"


# ============================================================
# HELPER: COUNT QUIVER ENTRIES
# ============================================================

qv_count () {
    local qv="$1"

    if [ ! -s "$qv" ]; then
        echo 0
        return
    fi

    uv run qvls "$qv" 2>/dev/null | wc -l
}


# ============================================================
# STEP 4 — RFDIFFUSION
# ============================================================

echo
echo "============================================================"
echo "STEP 4: RFdiffusion"
echo "============================================================"

RFDIFFUSION_START="$(date +%s)"

for ITEM in "${REGIMES[@]}"; do

    NAME="${ITEM%%|*}"
    LOOPS="${ITEM#*|}"

    OUTQV="$RFDIR/${NAME}_rfdiffusion.qv"
    OUTSC="$RFDIR/${NAME}_rfdiffusion.sc"

    echo
    echo "------------------------------------------------------------"
    echo "REGIME: $NAME"
    echo "LOOPS:  $LOOPS"
    echo "------------------------------------------------------------"

    REGIME_START="$(date +%s)"

    EXISTING="$(qv_count "$OUTQV")"

    if [ "$EXISTING" -eq "$NUM_BACKBONES" ]; then
        echo "RFdiffusion already complete: $EXISTING designs"
    else

        if [ -e "$OUTQV" ]; then
            echo "Removing incomplete RFdiffusion output:"
            echo "$OUTQV"
            rm -f "$OUTQV"
        fi

        rm -f "$OUTSC"

        uv run rfdiffusion \
            --target "$TARGET" \
            --framework "$FRAMEWORK" \
            --output-quiver "$OUTQV" \
            --num-designs "$NUM_BACKBONES" \
            --design-loops "$LOOPS" \
            --hotspots "$HOTSPOTS"

        COUNT="$(qv_count "$OUTQV")"

        if [ "$COUNT" -ne "$NUM_BACKBONES" ]; then
            echo "ERROR: Expected $NUM_BACKBONES RFdiffusion designs."
            echo "Found: $COUNT"
            exit 1
        fi
    fi

    # qvscorefile creates sibling .sc itself.
    rm -f "$OUTSC"
    uv run qvscorefile "$OUTQV"

    if [ ! -s "$OUTSC" ]; then
        echo "ERROR: RFdiffusion scorefile was not created:"
        echo "$OUTSC"
        exit 1
    fi

    report_duration "RFdiffusion ($NAME)" "$REGIME_START"

done


# ============================================================
# STEP 4B — FILTER RFDIFFUSION BACKBONES
# ============================================================

echo
echo "============================================================"
echo "STEP 4B: RFdiffusion geometry filtering"
echo "mindist <= $MINDIST_MAX"
echo "averagemin <= $AVERAGEMIN_MAX"
echo "============================================================"

FILTER_START="$(date +%s)"

for ITEM in "${REGIMES[@]}"; do

    NAME="${ITEM%%|*}"

    INPUT_QV="$RFDIR/${NAME}_rfdiffusion.qv"
    INPUT_SC="$RFDIR/${NAME}_rfdiffusion.sc"

    FILTERED_QV="$FILTERDIR/${NAME}_filtered.qv"
    SUMMARY="$FILTERDIR/${NAME}_filter_summary.tsv"
    TEMPDIR="$FILTERDIR/${NAME}_temp_pdbs"

    echo
    echo "------------------------------------------------------------"
    echo "Filtering: $NAME"
    echo "------------------------------------------------------------"

    REGIME_START="$(date +%s)"

    rm -rf "$TEMPDIR"
    mkdir -p "$TEMPDIR"

    # Extract all RFdiffusion designs.
    uv run qvextract \
        "$INPUT_QV" \
        -o "$TEMPDIR"

    # Parse scorefile and delete failing structures.
    INPUT_SC="$INPUT_SC" \
    TEMPDIR="$TEMPDIR" \
    SUMMARY="$SUMMARY" \
    MINDIST_MAX="$MINDIST_MAX" \
    AVERAGEMIN_MAX="$AVERAGEMIN_MAX" \
    uv run python - <<'PY'
import os
from pathlib import Path

sc = Path(os.environ["INPUT_SC"])
tempdir = Path(os.environ["TEMPDIR"])
summary = Path(os.environ["SUMMARY"])

mindist_max = float(os.environ["MINDIST_MAX"])
averagemin_max = float(os.environ["AVERAGEMIN_MAX"])

with sc.open() as f:
    header = f.readline().split()

    mindist_i = header.index("mindist")
    averagemin_i = header.index("averagemin")
    tag_i = header.index("tag")

    rows = []
    passing = set()

    for line in f:
        if not line.strip():
            continue

        cols = line.split()

        try:
            mindist = float(cols[mindist_i])
            averagemin = float(cols[averagemin_i])
        except (ValueError, IndexError):
            continue

        tag = cols[tag_i]

        passed = (
            mindist <= mindist_max
            and averagemin <= averagemin_max
        )

        rows.append((tag, mindist, averagemin, passed))

        if passed:
            passing.add(tag)

with summary.open("w") as out:
    out.write("tag\tmindist\taveragemin\tpass\n")

    for tag, mindist, averagemin, passed in rows:
        out.write(
            f"{tag}\t{mindist:.3f}\t"
            f"{averagemin:.3f}\t{passed}\n"
        )

for pdb in tempdir.glob("*.pdb"):
    if pdb.stem not in passing:
        pdb.unlink()

print(f"Passed {len(passing)}/{len(rows)} designs")

for tag, mindist, averagemin, passed in rows:
    if passed:
        print(
            f" PASS {tag:25s} "
            f"mindist={mindist:.2f} "
            f"averagemin={averagemin:.2f}"
        )
PY

    PASS_COUNT="$(find "$TEMPDIR" -maxdepth 1 -name '*.pdb' | wc -l)"

    echo
    echo "Passing backbones: $PASS_COUNT"

    rm -f "$FILTERED_QV"

    if [ "$PASS_COUNT" -eq 0 ]; then
        echo "WARNING: No passing backbones for $NAME."
        echo "Skipping ProteinMPNN/RF2 for this regime."
        report_duration "Filtering ($NAME)" "$REGIME_START"
        continue
    fi

    # qvfrompdbs writes Quiver binary to stdout.
    uv run qvfrompdbs \
        "$TEMPDIR"/*.pdb \
        > "$FILTERED_QV"

    CREATED="$(qv_count "$FILTERED_QV")"

    if [ "$CREATED" -ne "$PASS_COUNT" ]; then
        echo "ERROR: Filtered Quiver count mismatch."
        echo "Expected: $PASS_COUNT"
        echo "Created:  $CREATED"
        exit 1
    fi

    echo "Filtered Quiver contains $CREATED backbones."

    report_duration "Filtering ($NAME)" "$REGIME_START"

done

report_duration "Step 4B geometry filtering" "$FILTER_START"


# ============================================================
# STEP 5 — PROTEINMPNN
# ============================================================

echo
echo "============================================================"
echo "STEP 5: ProteinMPNN"
echo "============================================================"

MPNN_START="$(date +%s)"

for ITEM in "${REGIMES[@]}"; do

    NAME="${ITEM%%|*}"

    INPUT="$FILTERDIR/${NAME}_filtered.qv"
    OUTPUT="$MPNNDIR/${NAME}_mpnn.qv"

    echo
    echo "------------------------------------------------------------"
    echo "ProteinMPNN: $NAME"
    echo "------------------------------------------------------------"

    REGIME_START="$(date +%s)"

    if [ ! -s "$INPUT" ]; then
        echo "No filtered backbones. Skipping."
        report_duration "ProteinMPNN ($NAME)" "$REGIME_START"
        continue
    fi

    BACKBONES="$(qv_count "$INPUT")"
    EXPECTED=$((BACKBONES * SEQS_PER_BACKBONE))

    if [ -s "$OUTPUT" ]; then
        EXISTING="$(qv_count "$OUTPUT")"
    else
        EXISTING=0
    fi

    if [ "$EXISTING" -eq "$EXPECTED" ]; then
        echo "ProteinMPNN already complete:"
        echo "$EXISTING sequences"
        report_duration "ProteinMPNN ($NAME)" "$REGIME_START"
        continue
    fi

    if [ -e "$OUTPUT" ]; then
        echo "Removing incomplete ProteinMPNN output."
        rm -f "$OUTPUT"
    fi

    echo "Backbones: $BACKBONES"
    echo "Sequences/backbone: $SEQS_PER_BACKBONE"
    echo "Expected sequences: $EXPECTED"

    uv run proteinmpnn \
        -q "$INPUT" \
        --output-quiver "$OUTPUT" \
        -n "$SEQS_PER_BACKBONE" \
        -t "$MPNN_TEMP"

    CREATED="$(qv_count "$OUTPUT")"

    echo "ProteinMPNN generated: $CREATED"

    if [ "$CREATED" -ne "$EXPECTED" ]; then
        echo "WARNING: Expected $EXPECTED sequence designs."
        echo "ProteinMPNN produced $CREATED."
    fi

    report_duration "ProteinMPNN ($NAME)" "$REGIME_START"

done

report_duration "Step 5 ProteinMPNN" "$MPNN_START"


# ============================================================
# STEP 6 — RF2
# ============================================================

echo
echo "============================================================"
echo "STEP 6: RF2 validation"
echo "============================================================"

RF2_START="$(date +%s)"

for ITEM in "${REGIMES[@]}"; do

    NAME="${ITEM%%|*}"

    INPUT="$MPNNDIR/${NAME}_mpnn.qv"
    OUTPUT="$RF2DIR/${NAME}_rf2.qv"
    SCORE="$RF2DIR/${NAME}_rf2.sc"

    echo
    echo "------------------------------------------------------------"
    echo "RF2: $NAME"
    echo "------------------------------------------------------------"

    REGIME_START="$(date +%s)"

    if [ ! -s "$INPUT" ]; then
        echo "No ProteinMPNN output. Skipping."
        report_duration "RF2 ($NAME)" "$REGIME_START"
        continue
    fi

    EXPECTED="$(qv_count "$INPUT")"

    if [ -s "$OUTPUT" ]; then
        EXISTING="$(qv_count "$OUTPUT")"
    else
        EXISTING=0
    fi

    if [ "$EXISTING" -eq "$EXPECTED" ]; then
        echo "RF2 already complete:"
        echo "$EXISTING predictions"
    else

        if [ -e "$OUTPUT" ]; then
            echo "Removing incomplete RF2 output."
            rm -f "$OUTPUT"
        fi

        uv run rf2 \
            -q "$INPUT" \
            --output-quiver "$OUTPUT"

        CREATED="$(qv_count "$OUTPUT")"

        echo "RF2 predictions generated: $CREATED"

        if [ "$CREATED" -ne "$EXPECTED" ]; then
            echo "WARNING: RF2 input/output count mismatch."
            echo "Expected: $EXPECTED"
            echo "Created:  $CREATED"
        fi
    fi

    rm -f "$SCORE"
    uv run qvscorefile "$OUTPUT"

    if [ ! -s "$SCORE" ]; then
        echo "ERROR: RF2 scorefile was not created:"
        echo "$SCORE"
        exit 1
    fi

    report_duration "RF2 ($NAME)" "$REGIME_START"

done

report_duration "Step 6 RF2 validation" "$RF2_START"


# ============================================================
# STEP 7 — EXTRACT RF2 PDBS
# ============================================================

echo
echo "============================================================"
echo "STEP 7: Extract RF2 structures"
echo "============================================================"

EXTRACT_START="$(date +%s)"

for ITEM in "${REGIMES[@]}"; do

    NAME="${ITEM%%|*}"

    INPUT="$RF2DIR/${NAME}_rf2.qv"
    OUTDIR="$FINALDIR/$NAME"

    if [ ! -s "$INPUT" ]; then
        echo "No RF2 output for $NAME. Skipping."
        continue
    fi

    echo
    echo "------------------------------------------------------------"
    echo "Extracting: $NAME"
    echo "------------------------------------------------------------"

    REGIME_START="$(date +%s)"

    rm -rf "$OUTDIR"
    mkdir -p "$OUTDIR"

    uv run qvextract \
        "$INPUT" \
        -o "$OUTDIR"

    COUNT="$(find "$OUTDIR" -maxdepth 1 -name '*.pdb' | wc -l)"

    echo "Extracted $COUNT RF2 PDBs."

    report_duration "Extraction ($NAME)" "$REGIME_START"

done

report_duration "Step 7 RF2 extraction" "$EXTRACT_START"


# ============================================================
# STEP 8 — FILTER FINAL PDBS AND MAKE ALPHAFOLD SERVER JSON
# ============================================================

rm -rf "$FINAL_FILTERDIR"
uv run python "$RFAB/scripts/filter_final_pdbs.py" \
    "$RF2DIR" "$FINALDIR" "$FINAL_FILTERDIR" \
    --cdr-rmsd-max "$FINAL_CDR_RMSD_MAX" \
    --antibody-rmsd-max "$FINAL_ANTIBODY_RMSD_MAX" \
    --interaction-pae-max "$FINAL_INTERACTION_PAE_MAX" \
    --duplicate-min-count "$FINAL_DUPLICATE_MIN_COUNT" \
    --duplicate-keep-count "$FINAL_DUPLICATE_KEEP_COUNT" \
    --pae-delta "$FINAL_DUPLICATE_PAE_DELTA" \
    --composite-delta "$FINAL_DUPLICATE_COMPOSITE_DELTA"
uv run python "$RFAB/pdbs_to_afserver_json.py" \
    "$FINAL_FILTERDIR" --output "$FINAL_FILTERDIR/alphafold_jobs.json" \
    --batch-size "$FINAL_AFSERVER_BATCH_SIZE"


# ============================================================
# CAMPAIGN SUMMARY
# ============================================================

echo
echo "============================================================"
echo "CAMPAIGN COMPLETE"
echo "============================================================"
echo

printf "%-10s %-12s %-12s %-12s\n" \
    "REGIME" \
    "RFDIFF" \
    "FILTERED" \
    "RF2"

for ITEM in "${REGIMES[@]}"; do

    NAME="${ITEM%%|*}"

    RFQ="$RFDIR/${NAME}_rfdiffusion.qv"
    FQ="$FILTERDIR/${NAME}_filtered.qv"
    RQ="$RF2DIR/${NAME}_rf2.qv"

    RFN="$(qv_count "$RFQ")"
    FN="$(qv_count "$FQ")"
    RN="$(qv_count "$RQ")"

    printf "%-10s %-12s %-12s %-12s\n" \
        "$NAME" "$RFN" "$FN" "$RN"

done

echo
echo "RF2 scorefiles:"
find "$RF2DIR" -maxdepth 1 -name '*_rf2.sc' -print | sort

echo
echo "Final RF2 PDBs:"
echo "$FINALDIR"

report_duration "Total campaign" "$SCRIPT_START"
