#!/bin/bash
set -euo pipefail

# ============================================================
# 12BN / h-NbBCII10 / Chai-1 campaign
#
# Pipeline: RFdiffusion -> geometry filter -> ProteinMPNN -> Chai-1
#           -> final filtering -> AlphaFold Server JSON
#
# Edit the configuration block below for a new run.
# ============================================================

# Paths
RFAB="$HOME/RFantibody"
BASE="$HOME/Documents/RFantibody_outputs/12bn_campaign2"
TARGET="$RFAB/12bn_clean_trunc.pdb"
FULL_TARGET_PDB="$RFAB/12bn_clean.pdb"
FRAMEWORK="$RFAB/scripts/examples/example_inputs/h-NbBCII10.pdb"
# RFantibody design PDBs use T/H; the source 12bn PDB uses A/B.
TARGET_CHAIN="T"
VHH_CHAIN="H"

# RFdiffusion and ProteinMPNN
NUM_BACKBONES=500
SEQS_PER_BACKBONE=5
MPNN_TEMP=0.2
MINDIST_MAX=4
AVERAGEMIN_MAX=10
HOTSPOTS="A187,A192"
REGIMES=(
    "long|H1:7,H2:6,H3:10-16"
)

# Chai-1 uses a separate environment from RFantibody's uv environment.
CHAI_PYTHON="$RFAB/.chai-venv/bin/python"
CHAI_VALIDATOR="$RFAB/scripts/chai_validate_campaign.py"
CHAI_DOWNLOADS_DIR="$RFAB/.chai-downloads"
CHAI_NUM_SAMPLES=5
CHAI_RECYCLES=3
CHAI_TIMESTEPS=200
CHAI_SEED=20261007

# Final filtering and AlphaFold Server output
FINAL_CDR_RMSD_MAX=3
FINAL_ANTIBODY_RMSD_MAX=4
FINAL_INTERACTION_PAE_MAX=10
FINAL_DUPLICATE_MIN_COUNT=5
FINAL_DUPLICATE_KEEP_COUNT=2
FINAL_DUPLICATE_PAE_DELTA=3
FINAL_DUPLICATE_COMPOSITE_DELTA=0.3
FINAL_AFSERVER_BATCH_SIZE=30
KEEP_GEOMETRY_FILTER_PDBS=false
KEEP_FINAL_PDBS=false

# Environment
export CUDA_VISIBLE_DEVICES=0

# Derived paths
PREPDIR="$BASE/00_target"
RFDIR="$BASE/04_rfdiffusion"
FILTERDIR="$BASE/04b_filtered"
MPNNDIR="$BASE/05_mpnn"
CHAIDIR="$BASE/06_chai"
FINALDIR="$BASE/07_final_pdbs"
FINAL_FILTERDIR="$BASE/08_filtered_PDBs"
FULL_TARGET_FASTA="$PREPDIR/12bn_target_chain_A.fasta"

SCRIPT_START="$(date +%s)"

format_duration() {
    local seconds="$1"
    printf '%02dh:%02dm:%02ds' \
        "$((seconds / 3600))" \
        "$(((seconds % 3600) / 60))" \
        "$((seconds % 60))"
}

report_duration() {
    local label="$1"
    local start="$2"
    printf '%s took %s\n' "$label" "$(format_duration "$(($(date +%s) - start))")"
}

qv_count() {
    local qv="$1"
    if [ ! -s "$qv" ]; then
        echo 0
        return
    fi
    uv run qvls "$qv" 2>/dev/null | wc -l
}

mkdir -p "$PREPDIR" "$RFDIR" "$FILTERDIR" "$MPNNDIR" "$CHAIDIR" "$FINALDIR"
shopt -s nullglob
EXISTING_RFDIFFUSION_QVS=("$RFDIR"/*.qv)
if [ "${#EXISTING_RFDIFFUSION_QVS[@]}" -gt 0 ]; then
    echo "ERROR: RFdiffusion .qv file(s) already exist in $RFDIR."
    printf 'Refusing to run without deleting: %s\n' "${EXISTING_RFDIFFUSION_QVS[@]}"
    exit 1
fi
shopt -u nullglob

cd "$RFAB"

export PATH="$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/.conda/envs/rfantibody-runtime/lib:${LD_LIBRARY_PATH:-}"
export DGLBACKEND=pytorch

# ============================================================
# PREFLIGHT
# ============================================================

echo "== PREFLIGHT =="
for required_file in "$TARGET" "$FULL_TARGET_PDB" "$FRAMEWORK"; do
    if [ ! -s "$required_file" ]; then
        echo "ERROR: Required file not found: $required_file"
        exit 1
    fi
done
if [ ! -x "$CHAI_PYTHON" ]; then
    echo "ERROR: Chai Python is not executable: $CHAI_PYTHON"
    exit 1
fi
if [ ! -s "$CHAI_VALIDATOR" ]; then
    echo "ERROR: Chai validator not found: $CHAI_VALIDATOR"
    exit 1
fi

uv run python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA is unavailable")
print(f"RFantibody GPU: {torch.cuda.get_device_name(0)}")
PY

CHAI_DOWNLOADS_DIR="$CHAI_DOWNLOADS_DIR" "$CHAI_PYTHON" - <<'PY'
import torch
import chai_lab
if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
    raise SystemExit("ERROR: Chai requires CUDA with bfloat16 support")
print(f"Chai {getattr(chai_lab, '__version__', 'installed')} ready")
PY

# Build the full target FASTA required by Chai from the untruncated target PDB.
FULL_TARGET_PDB="$FULL_TARGET_PDB" FULL_TARGET_FASTA="$FULL_TARGET_FASTA" uv run python - <<'PY'
import os
from pathlib import Path

three_to_one = {
    "ALA":"A", "ARG":"R", "ASN":"N", "ASP":"D", "CYS":"C",
    "GLN":"Q", "GLU":"E", "GLY":"G", "HIS":"H", "ILE":"I",
    "LEU":"L", "LYS":"K", "MET":"M", "PHE":"F", "PRO":"P",
    "SER":"S", "THR":"T", "TRP":"W", "TYR":"Y", "VAL":"V",
    "MSE":"M",
}
source = Path(os.environ["FULL_TARGET_PDB"])
output = Path(os.environ["FULL_TARGET_FASTA"])
seen = set()
sequence = []
for line in source.read_text().splitlines():
    if not line.startswith(("ATOM  ", "HETATM")) or line[21] != "A":
        continue
    key = (line[22:26].strip(), line[26].strip())
    if key in seen:
        continue
    residue = three_to_one.get(line[17:20].strip().upper())
    if residue is not None:
        seen.add(key)
        sequence.append(residue)
if not sequence:
    raise SystemExit(f"No protein residues found in {source} chain A")
output.write_text(">protein|name=T\n" + "".join(sequence) + "\n")
print(f"Wrote full target FASTA: {output} ({len(sequence)} residues)")
PY

# Verify the hotspot residues exist in the RF target.
TARGET="$TARGET" HOTSPOTS="$HOTSPOTS" uv run python - <<'PY'
import os
from pathlib import Path

pdb = Path(os.environ["TARGET"])
requested = {int(item.strip()[1:]) for item in os.environ["HOTSPOTS"].split(",")}
found = set()
for line in pdb.read_text().splitlines():
    if line.startswith("ATOM") and line[21] == "A":
        try:
            number = int(line[22:26])
        except ValueError:
            continue
        if number in requested:
            found.add(number)
missing = requested - found
if missing:
    raise SystemExit(f"Missing hotspot residues in {pdb}: {sorted(missing)}")
print(f"Hotspots verified: {os.environ['HOTSPOTS']}")
PY

# ============================================================
# STEP 1: RFDIFFUSION
# ============================================================

echo "== STEP 1: RFdiffusion =="
for ITEM in "${REGIMES[@]}"; do
    IFS='|' read -r NAME LOOPS <<< "$ITEM"
    OUTQV="$RFDIR/${NAME}_rfdiffusion.qv"
    OUTSC="$RFDIR/${NAME}_rfdiffusion.sc"
    existing="$(qv_count "$OUTQV")"
    if [ "$existing" -ne "$NUM_BACKBONES" ]; then
        rm -f "$OUTQV" "$OUTSC"
        uv run rfdiffusion \
            --target "$TARGET" \
            --framework "$FRAMEWORK" \
            --output-quiver "$OUTQV" \
            --num-designs "$NUM_BACKBONES" \
            --design-loops "$LOOPS" \
            --hotspots "$HOTSPOTS"
        created="$(qv_count "$OUTQV")"
        if [ "$created" -ne "$NUM_BACKBONES" ]; then
            echo "ERROR: $NAME produced $created/$NUM_BACKBONES RFdiffusion designs"
            exit 1
        fi
    fi
    rm -f "$OUTSC"
    uv run qvscorefile "$OUTQV"
    [ -s "$OUTSC" ] || { echo "ERROR: Missing $OUTSC"; exit 1; }
    echo "$NAME: $(qv_count "$OUTQV") RFdiffusion designs"
done

# ============================================================
# STEP 2: GEOMETRY FILTER
# ============================================================

echo "== STEP 2: Geometry filter =="
for ITEM in "${REGIMES[@]}"; do
    NAME="${ITEM%%|*}"
    INPUT_QV="$RFDIR/${NAME}_rfdiffusion.qv"
    INPUT_SC="$RFDIR/${NAME}_rfdiffusion.sc"
    FILTERED_QV="$FILTERDIR/${NAME}_filtered.qv"
    TEMPDIR="$FILTERDIR/${NAME}_temp_pdbs"
    SUMMARY="$FILTERDIR/${NAME}_filter_summary.tsv"
    rm -rf "$TEMPDIR"
    mkdir -p "$TEMPDIR"
    uv run qvextract "$INPUT_QV" -o "$TEMPDIR"
    INPUT_SC="$INPUT_SC" TEMPDIR="$TEMPDIR" SUMMARY="$SUMMARY" \
    MINDIST_MAX="$MINDIST_MAX" AVERAGEMIN_MAX="$AVERAGEMIN_MAX" \
    uv run python - <<'PY'
import os
from pathlib import Path
sc = Path(os.environ["INPUT_SC"])
tempdir = Path(os.environ["TEMPDIR"])
summary = Path(os.environ["SUMMARY"])
mindist_max = float(os.environ["MINDIST_MAX"])
averagemin_max = float(os.environ["AVERAGEMIN_MAX"])
with sc.open() as handle:
    header = handle.readline().split()
    mindist_i = header.index("mindist")
    average_i = header.index("averagemin")
    tag_i = header.index("tag")
    rows = []
    passing = set()
    for line in handle:
        columns = line.split()
        if not columns:
            continue
        try:
            mindist = float(columns[mindist_i])
            average = float(columns[average_i])
            tag = columns[tag_i]
        except (ValueError, IndexError):
            continue
        passed = mindist <= mindist_max and average <= averagemin_max
        rows.append((tag, mindist, average, passed))
        if passed:
            passing.add(tag)
with summary.open("w") as handle:
    handle.write("tag\tmindist\taveragemin\tpass\n")
    for tag, mindist, average, passed in rows:
        handle.write(f"{tag}\t{mindist:.3f}\t{average:.3f}\t{passed}\n")
for pdb in tempdir.glob("*.pdb"):
    if pdb.stem not in passing:
        pdb.unlink()
print(f"Passed {len(passing)}/{len(rows)} designs")
PY
    pass_count="$(find "$TEMPDIR" -maxdepth 1 -name '*.pdb' -print | wc -l)"
    rm -f "$FILTERED_QV"
    if [ "$pass_count" -eq 0 ]; then
        echo "$NAME: no geometry-passing backbones"
        continue
    fi
    uv run qvfrompdbs "$TEMPDIR"/*.pdb > "$FILTERED_QV"
    created="$(qv_count "$FILTERED_QV")"
    [ "$created" -eq "$pass_count" ] || { echo "ERROR: Geometry count mismatch"; exit 1; }
    if [ "$KEEP_GEOMETRY_FILTER_PDBS" != true ]; then
        rm -rf "$TEMPDIR"
    fi
    echo "$NAME: $pass_count geometry-passing backbones"
done

# ============================================================
# STEP 3: PROTEINMPNN
# ============================================================

# echo "== STEP 3: ProteinMPNN =="
# for ITEM in "${REGIMES[@]}"; do
#     NAME="${ITEM%%|*}"
#     INPUT="$FILTERDIR/${NAME}_filtered.qv"
#     OUTPUT="$MPNNDIR/${NAME}_mpnn.qv"
#     [ -s "$INPUT" ] || { echo "$NAME: no filtered backbones, skipping"; continue; }
#     expected=$(( $(qv_count "$INPUT") * SEQS_PER_BACKBONE ))
#     existing="$(qv_count "$OUTPUT")"
#     if [ "$existing" -eq "$expected" ]; then
#         echo "$NAME ProteinMPNN already complete: $existing sequences"
#     else
#         rm -f "$OUTPUT"
#         uv run proteinmpnn -q "$INPUT" --output-quiver "$OUTPUT" -n "$SEQS_PER_BACKBONE" -t "$MPNN_TEMP"
#         echo "$NAME ProteinMPNN generated: $(qv_count "$OUTPUT") sequences"
#     fi
# done

# ============================================================
# STEP 4: CHAI-1
# ============================================================

echo "Removing RFdiffusion artifacts after ProteinMPNN."
rm -rf "$RFDIR"
mkdir -p "$RFDIR"

echo "== STEP 4: Chai-1 =="
for ITEM in "${REGIMES[@]}"; do
    NAME="${ITEM%%|*}"
    INPUT="$MPNNDIR/${NAME}_mpnn.qv"
    INPUT_PDBDIR="$CHAIDIR/${NAME}_input_pdbs"
    RAW_OUTPUT="$CHAIDIR/${NAME}_raw"
    SCORE="$CHAIDIR/${NAME}_rf2.sc"
    DETAILS="$CHAIDIR/${NAME}_chai_details.tsv"
    OUTDIR="$FINALDIR/$NAME"
    [ -s "$INPUT" ] || { echo "$NAME: no ProteinMPNN output, skipping"; continue; }
    expected="$(qv_count "$INPUT")"
    mkdir -p "$OUTDIR"
    existing="$(find "$OUTDIR" -maxdepth 1 -name '*.pdb' -print | wc -l)"
    if [ "$existing" -ne "$expected" ]; then
        rm -rf "$INPUT_PDBDIR"
        mkdir -p "$INPUT_PDBDIR" "$RAW_OUTPUT"
        uv run qvextract "$INPUT" -o "$INPUT_PDBDIR"
        CHAI_DOWNLOADS_DIR="$CHAI_DOWNLOADS_DIR" "$CHAI_PYTHON" "$CHAI_VALIDATOR" \
            --input-pdb-dir "$INPUT_PDBDIR" \
            --full-target-fasta "$FULL_TARGET_FASTA" \
            --raw-output-dir "$RAW_OUTPUT" \
            --pdb-output-dir "$OUTDIR" \
            --scorefile "$SCORE" \
            --details "$DETAILS" \
            --target-chain "$TARGET_CHAIN" \
            --vhh-chain "$VHH_CHAIN" \
            --num-samples "$CHAI_NUM_SAMPLES" \
            --recycles "$CHAI_RECYCLES" \
            --timesteps "$CHAI_TIMESTEPS" \
            --seed "$CHAI_SEED" \
            --resume
    fi
    [ -s "$SCORE" ] || { echo "ERROR: Missing Chai scorefile $SCORE"; exit 1; }
    echo "$NAME: $(find "$OUTDIR" -maxdepth 1 -name '*.pdb' -print | wc -l) Chai predictions"
done

# ============================================================
# STEP 5: FINAL FILTERING AND ALPHAFOLD SERVER JSON
# ============================================================

echo "== STEP 5: Final filtering and AlphaFold Server JSON =="
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
    "$FINAL_FILTERDIR" \
    --output "$FINAL_FILTERDIR/alphafold_jobs.json" \
    --batch-size "$FINAL_AFSERVER_BATCH_SIZE"

printf '12bn Chai campaign complete. Final PDBs: %s\n' "$FINAL_FILTERDIR"
if [ "$KEEP_FINAL_PDBS" != true ]; then
    rm -rf "$FINALDIR"
    echo "Removed final PDBs to save storage."
fi
report_duration "Total campaign" "$SCRIPT_START"
