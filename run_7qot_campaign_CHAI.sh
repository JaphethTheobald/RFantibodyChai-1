#!/bin/bash
set -euo pipefail

# ============================================================
# 1JBU noX / h-NbBCII10 H3-LENGTH CAMPAIGN
#
# RFdiffusion
#   -> geometry filter
#   -> ProteinMPNN
#   -> Chai-1
#   -> scorefiles
#   -> final PDB extraction
# ============================================================

RFAB="$HOME/RFantibody"
BASE="$HOME/Documents/RFantibody_outputs/7qot_chaiversion"

RFDIR="$BASE/04_rfdiffusion"
FILTERDIR="$BASE/04b_filtered"
MPNNDIR="$BASE/05_mpnn"
CHAIDIR="$BASE/06_chai"
FINALDIR="$BASE/07_final_pdbs"
FINAL_FILTERDIR="$BASE/08_filtered_PDBs"

# Chai-1 is deliberately isolated from RFantibody's uv environment.  This
# avoids replacing the tested RFantibody torch/DGL stack.
CHAI_PYTHON="$RFAB/.chai-venv/bin/python"
CHAI_VALIDATOR="$RFAB/scripts/chai_validate_campaign.py"
CHAI_DOWNLOADS_DIR="$RFAB/.chai-downloads"

CHAI_NUM_SAMPLES=5
CHAI_RECYCLES=3
CHAI_TIMESTEPS=200
CHAI_SEED=20261007

# Final filtering configuration.
# Start with the legacy gate for the calibration batch; reset this only after
# inspecting the Chai score distribution and recovered A-183-groove poses.
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
    "$CHAIDIR" \
    "$FINALDIR"

cd "$RFAB"

# ============================================================
# ENVIRONMENT
# ============================================================

export PATH="$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/.conda/envs/rfantibody-runtime/lib:${LD_LIBRARY_PATH:-}"
export DGLBACKEND=pytorch

export CUDA_VISIBLE_DEVICES=1

# ============================================================
# CAMPAIGN SETTINGS
# ============================================================

NUM_BACKBONES=2
SEQS_PER_BACKBONE=1
MPNN_TEMP=0.2

MINDIST_MAX=6
AVERAGEMIN_MAX=10

# HOTSPOTS="A224,A228,A229,A233,A253" #Japhybinding4 interactions with H196 H238 not well resolved on HK peptide
HOTSPOTS="A105,A107,A127,A151,A243" #Japhybinding5
FRAMEWORK="$RFAB/scripts/examples/example_inputs/h-NbBCII10.pdb"
TARGET="$HOME/RFantibody/7QOTA2A3trunc.pdb"
FULL_TARGET_FASTA="$HOME/RFantibody/7qota2a3trunc.fasta"

# Regime format:
#   name|loops|base/framework PDB|target PDB|hotspots
# The legacy name|loops format uses the global values above.
REGIMES=(
    #"long|H1:7,H2:6,H3:10-16|$FRAMEWORK|$TARGET|$HOTSPOTS"
    "a2a3|H1:7,H2:6,H3:10-16|$FRAMEWORK|$TARGET|$HOTSPOTS"
)
# ============================================================
# TRANSLATE ORIGINAL PDB HOTSPOTS -> RFANTIBODY-SAFE NUMBERING
# ============================================================


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

if [ ! -s "$FULL_TARGET_FASTA" ]; then
    echo "ERROR: Full mature FVII-zymogen FASTA not found:"
    echo "$FULL_TARGET_FASTA"
    echo "Create this once before the campaign; do not use the 1JBU fragment."
    exit 1
fi

if [ ! -x "$CHAI_PYTHON" ]; then
    echo "ERROR: Chai Python is not installed:"
    echo "$CHAI_PYTHON"
    echo "Install Chai in $RFAB/.chai-venv before running this campaign."
    exit 1
fi

if [ ! -s "$CHAI_VALIDATOR" ]; then
    echo "ERROR: Chai campaign validator not found:"
    echo "$CHAI_VALIDATOR"
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

CHAI_DOWNLOADS_DIR="$CHAI_DOWNLOADS_DIR" \
"$CHAI_PYTHON" - <<'PY'
import chai_lab
import torch

print("Chai:", getattr(chai_lab, "__version__", "installed"))
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("bfloat16 supported:", torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)

if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
    raise SystemExit("ERROR: Chai requires a CUDA GPU with bfloat16 support")
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

    IFS='|' read -r NAME LOOPS REGIME_FRAMEWORK REGIME_TARGET REGIME_HOTSPOTS <<< "$ITEM"

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
            --target "$REGIME_TARGET" \
            --framework "$REGIME_FRAMEWORK" \
            --output-quiver "$OUTQV" \
            --num-designs "$NUM_BACKBONES" \
            --design-loops "$LOOPS" \
            --hotspots "$REGIME_HOTSPOTS"

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
        echo "Skipping ProteinMPNN/Chai-1 for this regime."
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
# STEP 6 — CHAI-1 VALIDATION
# ============================================================

echo
echo "============================================================"
echo "STEP 6: Chai-1 validation"
echo "============================================================"

CHAI_START="$(date +%s)"

for ITEM in "${REGIMES[@]}"; do

    NAME="${ITEM%%|*}"

    INPUT="$MPNNDIR/${NAME}_mpnn.qv"
    INPUT_PDBDIR="$CHAIDIR/${NAME}_input_pdbs"
    RAW_OUTPUT="$CHAIDIR/${NAME}_raw"
    SCORE="$CHAIDIR/${NAME}_rf2.sc" # Legacy basename required by the sorter.
    DETAILS="$CHAIDIR/${NAME}_chai_details.tsv"
    OUTDIR="$FINALDIR/$NAME"

    echo
    echo "------------------------------------------------------------"
    echo "Chai-1: $NAME"
    echo "------------------------------------------------------------"

    REGIME_START="$(date +%s)"

    if [ ! -s "$INPUT" ]; then
        echo "No ProteinMPNN output. Skipping."
        report_duration "Chai-1 ($NAME)" "$REGIME_START"
        continue
    fi

    EXPECTED="$(qv_count "$INPUT")"

    # Create the regime-specific destination before checking for prior output.
    mkdir -p "$OUTDIR"
    if [ -d "$OUTDIR" ]; then
        EXISTING="$(find "$OUTDIR" -maxdepth 1 -name '*.pdb' -print | wc -l)"
    else
        EXISTING=0
    fi

    echo "debugggg2"

    if [ "$EXISTING" -eq "$EXPECTED" ]; then
        echo "Chai-1 already complete:"
        echo "$EXISTING predictions"
    else
        echo "debugggg"
        if [ ! -d "$INPUT_PDBDIR" ] || [ "$(find "$INPUT_PDBDIR" -maxdepth 1 -name '*.pdb' | wc -l)" -ne "$EXPECTED" ]; then
            rm -rf "$INPUT_PDBDIR"
            mkdir -p "$INPUT_PDBDIR"
            uv run qvextract \
                "$INPUT" \
                -o "$INPUT_PDBDIR"
        fi

        mkdir -p "$RAW_OUTPUT"

        CHAI_DOWNLOADS_DIR="$CHAI_DOWNLOADS_DIR" \
        "$CHAI_PYTHON" "$CHAI_VALIDATOR" \
            --input-pdb-dir "$INPUT_PDBDIR" \
            --full-target-fasta "$FULL_TARGET_FASTA" \
            --raw-output-dir "$RAW_OUTPUT" \
            --pdb-output-dir "$OUTDIR" \
            --scorefile "$SCORE" \
            --details "$DETAILS" \
            --num-samples "$CHAI_NUM_SAMPLES" \
            --recycles "$CHAI_RECYCLES" \
            --timesteps "$CHAI_TIMESTEPS" \
            --seed "$CHAI_SEED" \
            --resume

        CREATED="$(find "$OUTDIR" -maxdepth 1 -name '*.pdb' | wc -l)"

        echo "Chai-1 predictions generated: $CREATED"

        if [ "$CREATED" -ne "$EXPECTED" ]; then
            echo "WARNING: Chai-1 input/output count mismatch."
            echo "Expected: $EXPECTED"
            echo "Created:  $CREATED"
        fi
    fi

    if [ ! -s "$SCORE" ]; then
        echo "ERROR: Chai-compatible scorefile was not created:"
        echo "$SCORE"
        exit 1
    fi

    report_duration "Chai-1 ($NAME)" "$REGIME_START"

done

report_duration "Step 6 Chai-1 validation" "$CHAI_START"


# ============================================================
# STEP 7 — VERIFY CHAI PDBS
# ============================================================

echo
echo "============================================================"
echo "STEP 7: Verify Chai structures"
echo "============================================================"

EXTRACT_START="$(date +%s)"

for ITEM in "${REGIMES[@]}"; do

    NAME="${ITEM%%|*}"

    OUTDIR="$FINALDIR/$NAME"

    if [ ! -d "$OUTDIR" ]; then
        echo "No Chai output for $NAME. Skipping."
        continue
    fi

    echo
    echo "------------------------------------------------------------"
    echo "Verifying: $NAME"
    echo "------------------------------------------------------------"

    REGIME_START="$(date +%s)"

    COUNT="$(find "$OUTDIR" -maxdepth 1 -name '*.pdb' | wc -l)"

    echo "Found $COUNT Chai PDBs."

    report_duration "Chai verification ($NAME)" "$REGIME_START"

done

report_duration "Step 7 Chai verification" "$EXTRACT_START"


# ============================================================
# STEP 8 — FILTER FINAL PDBS AND MAKE ALPHAFOLD SERVER JSON
# ============================================================

rm -rf "$FINAL_FILTERDIR"
uv run python "$RFAB/scripts/filter_final_pdbs.py" \
    "$CHAIDIR" "$FINALDIR" "$FINAL_FILTERDIR" \
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
    "CHAI"

for ITEM in "${REGIMES[@]}"; do

    NAME="${ITEM%%|*}"

    RFQ="$RFDIR/${NAME}_rfdiffusion.qv"
    FQ="$FILTERDIR/${NAME}_filtered.qv"

    RFN="$(qv_count "$RFQ")"
    FN="$(qv_count "$FQ")"
    RN="$(find "$FINALDIR/$NAME" -maxdepth 1 -name '*.pdb' 2>/dev/null | wc -l)"

    printf "%-10s %-12s %-12s %-12s\n" \
        "$NAME" "$RFN" "$FN" "$RN"

done

echo
echo "Chai-compatible scorefiles:"
find "$CHAIDIR" -maxdepth 1 -name '*_rf2.sc' -print | sort

echo
echo "Final Chai PDBs:"
echo "$FINALDIR"

report_duration "Total campaign" "$SCRIPT_START"
