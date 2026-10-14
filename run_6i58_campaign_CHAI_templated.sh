#!/bin/bash
set -euo pipefail

# ============================================================
# 6i58 / h-NbBCII10 / Chai-1 campaign
#
# Pipeline: RFdiffusion -> geometry filter -> ProteinMPNN -> Chai-1
#           -> final filtering -> AlphaFold Server JSON
#
# Chai validation uses a target-only structural template made from
# 6i58_a3.pdb.  The target is conditioned on that experimental
# conformation; the VHH remains untemplated and there are no complex/contact
# restraints.
#
# Edit this block for a new campaign. Everything below it is derived from
# these settings or implements the pipeline.
# ============================================================

# Paths
RFAB="$HOME/RFantibody"
BASE="$HOME/Documents/RFantibody_outputs/6i58_campaign_1500"
FRAMEWORK="$RFAB/scripts/examples/example_inputs/h-NbBCII10.pdb"
TARGET="$RFAB/12bn_chainA.pdb"
FULL_TARGET_FASTA="$RFAB/12bn_chainA.fasta"
TARGET_TEMPLATE_PDB="$RFAB/12bn_chainA.pdb"
TARGET_TEMPLATE_CIF_GZ="$RFAB/chai_templates/12bn_clean_chain_A_template.cif.gz"

# RFdiffusion and ProteinMPNN
NUM_BACKBONES=2000
SEQS_PER_BACKBONE=5
MPNN_TEMP=0.2
MINDIST_MAX=6
AVERAGEMIN_MAX=7.5 # this harsh value screens for ones that clash with LC

# Regime format: name|loops|framework PDB|target PDB|hotspots
HOTSPOTS="A187,A260"
REGIMES=(
    "short|H1:7,H2:6,H3:5-9|$FRAMEWORK|$TARGET|$HOTSPOTS"
    "medium|H1:7,H2:6,H3:7-11|$FRAMEWORK|$TARGET|$HOTSPOTS"
)

# Chai-1 uses a separate environment from RFantibody's uv environment.
CHAI_PYTHON="$RFAB/.chai-venv/bin/python"
CHAI_VALIDATOR="$RFAB/scripts/chai_validate_campaign_templated.py"
CHAI_DOWNLOADS_DIR="$RFAB/.chai-downloads"
KALIGN_ENV="$RFAB/.kalign-env"
CHAI_TEMPLATE_WORKDIR="$BASE/chai_target_template_workdir"
CHAI_TEMPLATE_IDENTIFIER="12bn"
CHAI_TEMPLATE_TARGET_NAME="T"
CHAI_NUM_SAMPLES=3
CHAI_RECYCLES=3
CHAI_TIMESTEPS=200
CHAI_SEED=20261007

# Final filtering and AlphaFold Server output
FINAL_CDR_RMSD_MAX=3
FINAL_ANTIBODY_RMSD_MAX=4
FINAL_INTERACTION_PAE_MAX=10
FINAL_DUPLICATE_MIN_COUNT=4
FINAL_DUPLICATE_KEEP_COUNT=2
FINAL_DUPLICATE_PAE_DELTA=3
FINAL_DUPLICATE_COMPOSITE_DELTA=0.3
FINAL_AFSERVER_BATCH_SIZE=30
KEEP_GEOMETRY_FILTER_PDBS=false
KEEP_FINAL_PDBS=false

# Environment
export CUDA_VISIBLE_DEVICES=0

# Derived output paths
RFDIR="$BASE/04_rfdiffusion"
FILTERDIR="$BASE/04b_filtered"
MPNNDIR="$BASE/05_mpnn"
# Keep template-conditioned predictions separate from the prior blind Chai
# results.  This prevents --resume from treating an old blind result as a
# templated prediction.
CHAIDIR="$BASE/06_chai_target_template"
FINALDIR="$BASE/07_final_pdbs_target_template"
FINAL_FILTERDIR="$BASE/08_filtered_PDBs_target_template"
COMPLETE_DIR="$BASE/.complete"

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
    "$FINALDIR" \
    "$COMPLETE_DIR"

shopt -s nullglob
EXISTING_RFDIFFUSION_QVS=("$RFDIR"/*.qv)
# if [ "${#EXISTING_RFDIFFUSION_QVS[@]}" -gt 0 ]; then
#     echo "ERROR: RFdiffusion .qv file(s) already exist in $RFDIR."
#     printf 'Refusing to run without deleting: %s\n' "${EXISTING_RFDIFFUSION_QVS[@]}"
#     exit 1
# fi
shopt -u nullglob

cd "$RFAB"

export PATH="$KALIGN_ENV/bin:$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/.conda/envs/rfantibody-runtime/lib:${LD_LIBRARY_PATH:-}"
export DGLBACKEND=pytorch

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
    echo "ERROR: Full mature 7QOT target FASTA not found:"
    echo "$FULL_TARGET_FASTA"
    exit 1
fi

if [ ! -s "$TARGET_TEMPLATE_PDB" ]; then
    echo "ERROR: Target-only Chai template PDB not found:"
    echo "$TARGET_TEMPLATE_PDB"
    exit 1
fi

if [ ! -s "$TARGET_TEMPLATE_CIF_GZ" ]; then
    echo "ERROR: Prepared Chai template CIF not found:"
    echo "$TARGET_TEMPLATE_CIF_GZ"
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

if ! command -v kalign >/dev/null 2>&1; then
    echo "ERROR: Kalign executable not found in PATH."
    echo "Expected: $KALIGN_ENV/bin/kalign"
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
    RF_MARKER="$COMPLETE_DIR/${NAME}_rfdiffusion.complete"

    echo
    echo "------------------------------------------------------------"
    echo "REGIME: $NAME"
    echo "LOOPS:  $LOOPS"
    echo "------------------------------------------------------------"

    REGIME_START="$(date +%s)"

    EXISTING="$(qv_count "$OUTQV")"

    # if [ "$EXISTING" -eq "$NUM_BACKBONES" ] && [ -s "$OUTSC" ]; then
    #     echo "RFdiffusion already complete: $EXISTING designs"
    # else

    #     if [ -e "$OUTQV" ]; then
    #         echo "Removing incomplete RFdiffusion output:"
    #         echo "$OUTQV"
    #         rm -f "$OUTQV"
    #     fi

    #     rm -f "$OUTSC"

    #     uv run rfdiffusion \
    #         --target "$REGIME_TARGET" \
    #         --framework "$REGIME_FRAMEWORK" \
    #         --output-quiver "$OUTQV" \
    #         --num-designs "$NUM_BACKBONES" \
    #         --design-loops "$LOOPS" \
    #         --hotspots "$REGIME_HOTSPOTS"

    #     COUNT="$(qv_count "$OUTQV")"

    #     if [ "$COUNT" -ne "$NUM_BACKBONES" ]; then
    #         echo "ERROR: Expected $NUM_BACKBONES RFdiffusion designs."
    #         echo "Found: $COUNT"
    #         exit 1
    #     fi
    #     rm -f "$OUTSC"
    #     uv run qvscorefile "$OUTQV"
    # fi

    # if [ ! -s "$OUTSC" ]; then
    #     echo "ERROR: RFdiffusion scorefile was not created:"
    #     echo "$OUTSC"
    #     exit 1
    # fi
    # touch "$RF_MARKER"

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
    GEOMETRY_MARKER="$COMPLETE_DIR/${NAME}_geometry_filter.complete"

    if [ -s "$SUMMARY" ] \
        && { [ -s "$FILTERED_QV" ] || [ -f "$GEOMETRY_MARKER" ]; }; then
        echo "Geometry filtering already complete: $NAME"
        continue
    fi

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
PY

    PASS_COUNT="$(find "$TEMPDIR" -maxdepth 1 -name '*.pdb' | wc -l)"

    echo
    echo "Passing backbones: $PASS_COUNT"

    rm -f "$FILTERED_QV"

    if [ "$PASS_COUNT" -eq 0 ]; then
        echo "WARNING: No passing backbones for $NAME."
        echo "Skipping ProteinMPNN/Chai-1 for this regime."
        touch "$GEOMETRY_MARKER"
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

    if [ "$KEEP_GEOMETRY_FILTER_PDBS" != true ]; then
        rm -rf "$TEMPDIR"
    fi

    echo "Filtered Quiver contains $CREATED backbones."
    touch "$GEOMETRY_MARKER"

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
# CLEAN RFdiffusion FILES TO SAVE STORAGE
# ============================================================

echo "Removing RFdiffusion artifacts after ProteinMPNN."
rm -rf "$RFDIR"
mkdir -p "$RFDIR"


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


    if [ "$EXISTING" -eq "$EXPECTED" ] \
        && [ -s "$SCORE" ] \
        && [ -s "$DETAILS" ]; then
        echo "Chai-1 already complete:"
        echo "$EXISTING predictions"
    else
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
            --template-pdb "$TARGET_TEMPLATE_PDB" \
            --template-cif-gz "$TARGET_TEMPLATE_CIF_GZ" \
            --template-target-fasta "$FULL_TARGET_FASTA" \
            --template-work-dir "$CHAI_TEMPLATE_WORKDIR" \
            --template-identifier "$CHAI_TEMPLATE_IDENTIFIER" \
            --template-target-name "$CHAI_TEMPLATE_TARGET_NAME" \
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
    touch "$COMPLETE_DIR/${NAME}_chai.complete"

    report_duration "Chai-1 ($NAME)" "$REGIME_START"

done

report_duration "Step 6 Chai-1 validation" "$CHAI_START"


# ============================================================
# STEP 7 — FILTER FINAL PDBS AND MAKE ALPHAFOLD SERVER JSON
# ============================================================

FINAL_MARKER="$COMPLETE_DIR/final_filter_and_afserver.complete"
if [ -f "$FINAL_MARKER" ] \
    && [ -s "$FINAL_FILTERDIR/filter_report.tsv" ] \
    && find "$FINAL_FILTERDIR" -maxdepth 1 -name 'alphafold_jobs*.json' -print -quit | grep -q .; then
    echo "Final filtering and AlphaFold Server JSON already complete."
else
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
    touch "$FINAL_MARKER"
fi


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

if [ "$KEEP_FINAL_PDBS" != true ]; then
    rm -rf "$FINALDIR"
    echo "Removed final PDBs to save storage."
fi

report_duration "Total campaign" "$SCRIPT_START"
