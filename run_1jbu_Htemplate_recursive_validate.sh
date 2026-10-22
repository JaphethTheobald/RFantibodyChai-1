#!/usr/bin/env bash
set -euo pipefail

# Recursively Chai-validate existing 1JBU VHH PDBs.
#
# Usage:
#   bash run_1jbu_Htemplate_recursive_validate.sh \
#     /home/schrogpu/Documents/RFantibody_outputs/Best/f7_1jbu \
#     ~/Documents/RFantibody_outputs/1jbu_recursive_chai
#
# Input PDBs must use chain T for 1JBU H and chain H for the VHH, as in the
# RFantibody final PDBs. The experimental 1JBU H structure is templated; L is
# included in every Chai prediction as untemplated target context. No design,
# ProteinMPNN, RFdiffusion, or final filtering steps are run here.

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 INPUT_PDB_DIRECTORY OUTPUT_DIRECTORY" >&2
    exit 2
fi

INPUT_ROOT="$(realpath -e "$1")"
OUTPUT_ROOT="$(realpath -m "$2")"
case "$OUTPUT_ROOT" in
    "$INPUT_ROOT"|"$INPUT_ROOT"/*)
        echo "ERROR: OUTPUT_DIRECTORY must not be inside INPUT_PDB_DIRECTORY." >&2
        echo "Choose a separate output location so generated PDBs cannot be revalidated." >&2
        exit 2
        ;;
esac

# ---------------------------------------------------------------------------
# 1JBU / Chai configuration
# ---------------------------------------------------------------------------
RFAB="$HOME/RFantibody"
FULL_TARGET_FASTA="$RFAB/1jbu_H.fasta"
ADDITIONAL_TARGET_FASTA="$RFAB/1jbu_L.fasta"
TARGET_TEMPLATE_PDB="$RFAB/1jbu_HL.pdb"
TARGET_TEMPLATE_CIF_GZ="$RFAB/chai_templates/1jbu_chain_H_template.cif.gz"

CHAI_PYTHON="$RFAB/.chai-venv/bin/python"
CHAI_VALIDATOR="$RFAB/scripts/chai_validate_campaign_templated_H_Lcontext.py"
CHAI_DOWNLOADS_DIR="$RFAB/.chai-downloads"
KALIGN_ENV="$RFAB/.kalign-env"

CHAI_TEMPLATE_WORKDIR="$OUTPUT_ROOT/chai_target_template_1jbu_chain_H"
CHAI_TEMPLATE_IDENTIFIER="1JBUHTEMPLATE"
CHAI_TEMPLATE_TARGET_NAME="T"
CHAI_NUM_SAMPLES=1
CHAI_RECYCLES=5
CHAI_TIMESTEPS=200
CHAI_SEED=20261014
TARGET_CHAIN=T
VHH_CHAIN=H

export CUDA_VISIBLE_DEVICES=0
export PATH="$KALIGN_ENV/bin:$HOME/.local/bin:$PATH"

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
for path in "$FULL_TARGET_FASTA" "$ADDITIONAL_TARGET_FASTA" "$TARGET_TEMPLATE_PDB" "$TARGET_TEMPLATE_CIF_GZ" "$CHAI_VALIDATOR"; do
    if [ ! -s "$path" ]; then
        echo "ERROR: Required file is missing or empty: $path" >&2
        exit 1
    fi
done
if [ ! -x "$CHAI_PYTHON" ]; then
    echo "ERROR: Chai Python is not executable: $CHAI_PYTHON" >&2
    exit 1
fi
if ! command -v kalign >/dev/null 2>&1; then
    echo "ERROR: kalign was not found. Expected it in $KALIGN_ENV/bin." >&2
    exit 1
fi

mkdir -p "$OUTPUT_ROOT"
STAGED_INPUT="$OUTPUT_ROOT/input_snapshot"
MANIFEST="$OUTPUT_ROOT/input_manifest.tsv"
RAW_OUTPUT="$OUTPUT_ROOT/raw"
PDB_OUTPUT="$OUTPUT_ROOT/selected_pdbs"
SCOREFILE="$OUTPUT_ROOT/chai_scorefile.tsv"
DETAILS="$OUTPUT_ROOT/chai_details.tsv"

# Snapshot the inputs once under flat, stable names. The validator only reads
# this snapshot; it never scans OUTPUT_ROOT, so its selected PDBs cannot enter
# a subsequent validation pass. Do not add/remove source PDBs after creating a
# snapshot—use a new OUTPUT_DIRECTORY for a different source set.
if [ ! -f "$MANIFEST" ]; then
    mkdir -p "$STAGED_INPUT"
    mapfile -d '' PDBS < <(find "$INPUT_ROOT" -type f -iname '*.pdb' -print0 | sort -z)
    if [ "${#PDBS[@]}" -eq 0 ]; then
        echo "ERROR: No .pdb files were found below $INPUT_ROOT" >&2
        exit 1
    fi
    printf 'staged_name\toriginal_path\n' > "$MANIFEST"
    index=0
    for pdb in "${PDBS[@]}"; do
        index=$((index + 1))
        staged_name="$(printf '%06d' "$index")_$(basename "$pdb")"
        ln -s "$pdb" "$STAGED_INPUT/$staged_name"
        printf '%s\t%s\n' "$staged_name" "$pdb" >> "$MANIFEST"
    done
    echo "Snapshotted $index input PDBs in $STAGED_INPUT"
else
    echo "Reusing existing input snapshot: $STAGED_INPUT"
fi

expected="$(find "$STAGED_INPUT" -maxdepth 1 -type l -name '*.pdb' | wc -l)"
if [ "$expected" -eq 0 ]; then
    echo "ERROR: Input snapshot is empty: $STAGED_INPUT" >&2
    exit 1
fi

mkdir -p "$RAW_OUTPUT" "$PDB_OUTPUT"

CHAI_TEMPLATE_CIF_FOLDER="$CHAI_TEMPLATE_WORKDIR/cif" \
CHAI_DOWNLOADS_DIR="$CHAI_DOWNLOADS_DIR" \
"$CHAI_PYTHON" "$CHAI_VALIDATOR" \
    --input-pdb-dir "$STAGED_INPUT" \
    --full-target-fasta "$FULL_TARGET_FASTA" \
    --additional-target-fasta "$ADDITIONAL_TARGET_FASTA" \
    --template-pdb "$TARGET_TEMPLATE_PDB" \
    --template-cif-gz "$TARGET_TEMPLATE_CIF_GZ" \
    --template-target-fasta "$FULL_TARGET_FASTA" \
    --template-work-dir "$CHAI_TEMPLATE_WORKDIR" \
    --template-identifier "$CHAI_TEMPLATE_IDENTIFIER" \
    --template-target-name "$CHAI_TEMPLATE_TARGET_NAME" \
    --raw-output-dir "$RAW_OUTPUT" \
    --pdb-output-dir "$PDB_OUTPUT" \
    --scorefile "$SCOREFILE" \
    --details "$DETAILS" \
    --target-chain "$TARGET_CHAIN" \
    --vhh-chain "$VHH_CHAIN" \
    --num-samples "$CHAI_NUM_SAMPLES" \
    --recycles "$CHAI_RECYCLES" \
    --timesteps "$CHAI_TIMESTEPS" \
    --seed "$CHAI_SEED" \
    --resume

created="$(find "$PDB_OUTPUT" -maxdepth 1 -type f -name '*.pdb' | wc -l)"
echo "Completed recursive 1JBU Chai validation: $created / $expected selected PDBs"
echo "Scores:   $SCOREFILE"
echo "Details:  $DETAILS"
echo "Inputs:   $MANIFEST"
