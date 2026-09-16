#!/bin/bash
set -euo pipefail

# ============================================================
# 6i58 / h-NbBCII10 H3-LENGTH CAMPAIGN
#
# targets/6i58_exported.pdb
#   -> clean chain A
#   -> RFdiffusion
#   -> geometry filter
#   -> ProteinMPNN
#   -> RF2
#   -> scorefiles
#   -> final PDB extraction
# ============================================================

RFAB="$HOME/RFantibody"

BASE="$HOME/Documents/RFantibody_outputs/6i58_long_binding1_campaign"

PREPDIR="$BASE/00_target"
RFDIR="$BASE/04_rfdiffusion"
FILTERDIR="$BASE/04b_filtered"
MPNNDIR="$BASE/05_mpnn"
RF2DIR="$BASE/06_rf2"
FINALDIR="$BASE/07_final_pdbs"

SOURCE="$PREPDIR/6i58_exported.pdb"
TARGET="$PREPDIR/6i58_clean_chainA.pdb"

FRAMEWORK="$RFAB/scripts/examples/example_inputs/h-NbBCII10.pdb"

mkdir -p \
    "$PREPDIR" \
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



# GPU 2
export CUDA_VISIBLE_DEVICES=GPU-589685a4-8d09-fe8d-b4d2-5268717630e0


# ============================================================
# CAMPAIGN SETTINGS
# ============================================================

NUM_BACKBONES=100
SEQS_PER_BACKBONE=5
MPNN_TEMP=0.2

MINDIST_MAX=6.0
AVERAGEMIN_MAX=10.0

HOTSPOTS="A181,A182,A183,A184,A185,A186,A187,A188,A189,A190,A191,A192"

REGIMES=(
    "long|H1:7,H2:6,H3:10-16"
)


# ============================================================
# HELPER
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
# PREFLIGHT
# ============================================================

echo
echo "============================================================"
echo "PREFLIGHT"
echo "============================================================"

if [ ! -s "$SOURCE" ]; then
    echo "ERROR: Source PDB not found:"
    echo "$SOURCE"
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
    raise SystemExit("ERROR: CUDA unavailable")

print("GPU:", torch.cuda.get_device_name(0))
PY


# ============================================================
# STEP 0 — CLEAN TARGET
# ============================================================

echo
echo "============================================================"
echo "STEP 0: Cleaning 6i58_exported.pdb"
echo "============================================================"

SOURCE="$SOURCE" TARGET="$TARGET" uv run python - <<'PY'
import os
from pathlib import Path

src = Path(os.environ["SOURCE"])
dst = Path(os.environ["TARGET"])

kept = []
seen_model = False

with src.open(errors="ignore") as f:
    for line in f:

        if line.startswith("MODEL"):
            if seen_model:
                break
            seen_model = True
            continue

        if line.startswith("ENDMDL"):
            if seen_model:
                break
            continue

        if not line.startswith("ATOM"):
            continue

        # Chain A only
        if line[21:22] != "A":
            continue

        # Keep blank/A altloc only
        altloc = line[16:17]

        if altloc not in (" ", "A"):
            continue

        if altloc == "A":
            line = line[:16] + " " + line[17:]

        kept.append(line.rstrip("\n"))

if not kept:
    raise SystemExit(
        "ERROR: No chain A ATOM records found."
    )

with dst.open("w") as out:
    for line in kept:
        out.write(line + "\n")

    out.write("TER\n")
    out.write("END\n")

print(f"Wrote {len(kept)} atoms")
print(dst)
PY


# ============================================================
# VERIFY HOTSPOTS
# ============================================================

echo
echo "============================================================"
echo "Checking A180-A190"
echo "============================================================"

TARGET="$TARGET" uv run python - <<'PY'
import os
from pathlib import Path

pdb = Path(os.environ["TARGET"])

residues = {}

with pdb.open() as f:
    for line in f:

        if not line.startswith("ATOM"):
            continue

        if line[21:22] != "A":
            continue

        try:
            resnum = int(line[22:26])
        except ValueError:
            continue

        resname = line[17:20].strip()
        atom = line[12:16].strip()

        residues.setdefault(
            resnum,
            {"name": resname, "atoms": set()}
        )

        residues[resnum]["atoms"].add(atom)

bad = []

for n in range(180, 191):

    r = residues.get(n)

    if r is None:
        print(f"A{n}: MISSING")
        bad.append(n)
        continue

    backbone_ok = all(
        atom in r["atoms"]
        for atom in ("N", "CA", "C")
    )

    status = "OK" if backbone_ok else "INCOMPLETE"

    print(f"A{n}: {r['name']:3s} {status}")

    if not backbone_ok:
        bad.append(n)

if bad:
    raise SystemExit(
        "ERROR: Missing/incomplete hotspot residues: "
        + ", ".join(map(str, bad))
    )

print("All hotspot residues OK.")
PY


# ============================================================
# STEP 4 — RFDIFFUSION
# ============================================================

echo
echo "============================================================"
echo "STEP 4: RFdiffusion"
echo "============================================================"

for ITEM in "${REGIMES[@]}"; do

    NAME="${ITEM%%|*}"
    LOOPS="${ITEM#*|}"

    OUTQV="$RFDIR/${NAME}_rfdiffusion.qv"
    OUTSC="$RFDIR/${NAME}_rfdiffusion.sc"

    echo
    echo "------------------------------------------------------------"
    echo "REGIME: $NAME"
    echo "LOOPS: $LOOPS"
    echo "------------------------------------------------------------"

    EXISTING="$(qv_count "$OUTQV")"

    if [ "$EXISTING" -eq "$NUM_BACKBONES" ]; then

        echo "RFdiffusion already complete: $EXISTING"

    else

        rm -f "$OUTQV" "$OUTSC"

        uv run rfdiffusion \
            --target "$TARGET" \
            --framework "$FRAMEWORK" \
            --output-quiver "$OUTQV" \
            --num-designs "$NUM_BACKBONES" \
            --design-loops "$LOOPS" \
            --hotspots "$HOTSPOTS"

        COUNT="$(qv_count "$OUTQV")"

        if [ "$COUNT" -ne "$NUM_BACKBONES" ]; then
            echo "ERROR: Expected $NUM_BACKBONES designs."
            echo "Found: $COUNT"
            exit 1
        fi
    fi

    rm -f "$OUTSC"

    uv run qvscorefile "$OUTQV"

    if [ ! -s "$OUTSC" ]; then
        echo "ERROR: Scorefile not created:"
        echo "$OUTSC"
        exit 1
    fi

done



# ============================================================
# STEP 4B — FILTER
# ============================================================

echo
echo "============================================================"
echo "STEP 4B: Geometry filtering"
echo "mindist <= $MINDIST_MAX"
echo "averagemin <= $AVERAGEMIN_MAX"
echo "============================================================"

for ITEM in "${REGIMES[@]}"; do

    NAME="${ITEM%%|*}"

    INPUT_QV="$RFDIR/${NAME}_rfdiffusion.qv"
    INPUT_SC="$RFDIR/${NAME}_rfdiffusion.sc"

    FILTERED_QV="$FILTERDIR/${NAME}_filtered.qv"
    SUMMARY="$FILTERDIR/${NAME}_filter_summary.tsv"
    TEMPDIR="$FILTERDIR/${NAME}_temp_pdbs"

    rm -rf "$TEMPDIR"
    mkdir -p "$TEMPDIR"

    uv run qvextract \
        "$INPUT_QV" \
        -o "$TEMPDIR"

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
average_max = float(os.environ["AVERAGEMIN_MAX"])

with sc.open() as f:
    header = f.readline().split()

    mindist_i = header.index("mindist")
    average_i = header.index("averagemin")
    tag_i = header.index("tag")

    rows = []
    passing = set()

    for line in f:

        if not line.strip():
            continue

        cols = line.split()

        try:
            mindist = float(cols[mindist_i])
            average = float(cols[average_i])
            tag = cols[tag_i]
        except (ValueError, IndexError):
            continue

        passed = (
            mindist <= mindist_max
            and average <= average_max
        )

        rows.append((tag, mindist, average, passed))

        if passed:
            passing.add(tag)

with summary.open("w") as out:

    out.write("tag\tmindist\taveragemin\tpass\n")

    for tag, mindist, average, passed in rows:
        out.write(
            f"{tag}\t{mindist:.3f}\t"
            f"{average:.3f}\t{passed}\n"
        )

for pdb in tempdir.glob("*.pdb"):

    if pdb.stem not in passing:
        pdb.unlink()

print(f"Passed {len(passing)}/{len(rows)} designs")

for tag, mindist, average, passed in rows:

    if passed:
        print(
            f" PASS {tag:25s} "
            f"mindist={mindist:.2f} "
            f"averagemin={average:.2f}"
        )
PY

    PASS_COUNT="$(
        find "$TEMPDIR" \
        -maxdepth 1 \
        -name '*.pdb' \
        | wc -l
    )"

    echo "Passing backbones: $PASS_COUNT"

    rm -f "$FILTERED_QV"

    if [ "$PASS_COUNT" -eq 0 ]; then
        echo "No passing designs for $NAME."
        continue
    fi

    uv run qvfrompdbs \
        "$TEMPDIR"/*.pdb \
        > "$FILTERED_QV"

    CREATED="$(qv_count "$FILTERED_QV")"

    if [ "$CREATED" -ne "$PASS_COUNT" ]; then
        echo "ERROR: Filtered Quiver count mismatch."
        exit 1
    fi

done


# ============================================================
# STEP 5 — PROTEINMPNN
# ============================================================

echo
echo "============================================================"
echo "STEP 5: ProteinMPNN"
echo "============================================================"

for ITEM in "${REGIMES[@]}"; do

    NAME="${ITEM%%|*}"

    INPUT="$FILTERDIR/${NAME}_filtered.qv"
    OUTPUT="$MPNNDIR/${NAME}_mpnn.qv"

    if [ ! -s "$INPUT" ]; then
        echo "$NAME: no filtered backbones, skipping."
        continue
    fi

    BACKBONES="$(qv_count "$INPUT")"
    EXPECTED=$((BACKBONES * SEQS_PER_BACKBONE))

    EXISTING=0

    if [ -s "$OUTPUT" ]; then
        EXISTING="$(qv_count "$OUTPUT")"
    fi

    if [ "$EXISTING" -eq "$EXPECTED" ]; then
        echo "$NAME ProteinMPNN already complete."
        continue
    fi

    rm -f "$OUTPUT"

    uv run proteinmpnn \
        -q "$INPUT" \
        --output-quiver "$OUTPUT" \
        -n "$SEQS_PER_BACKBONE" \
        -t "$MPNN_TEMP"

    echo "$NAME ProteinMPNN: $(qv_count "$OUTPUT") sequences"

done


# ============================================================
# STEP 6 — RF2
# ============================================================

echo
echo "============================================================"
echo "STEP 6: RF2"
echo "============================================================"

for ITEM in "${REGIMES[@]}"; do

    NAME="${ITEM%%|*}"

    INPUT="$MPNNDIR/${NAME}_mpnn.qv"
    OUTPUT="$RF2DIR/${NAME}_rf2.qv"
    SCORE="$RF2DIR/${NAME}_rf2.sc"

    if [ ! -s "$INPUT" ]; then
        echo "$NAME: no MPNN output, skipping."
        continue
    fi

    EXPECTED="$(qv_count "$INPUT")"

    EXISTING=0

    if [ -s "$OUTPUT" ]; then
        EXISTING="$(qv_count "$OUTPUT")"
    fi

    if [ "$EXISTING" -ne "$EXPECTED" ]; then

        rm -f "$OUTPUT"

        uv run rf2 \
            -q "$INPUT" \
            --output-quiver "$OUTPUT"
    else
        echo "$NAME RF2 already complete."
    fi

    rm -f "$SCORE"

    uv run qvscorefile "$OUTPUT"

    if [ ! -s "$SCORE" ]; then
        echo "ERROR: RF2 scorefile missing:"
        echo "$SCORE"
        exit 1
    fi

done


# ============================================================
# STEP 7 — EXTRACT FINAL PDBS
# ============================================================

echo
echo "============================================================"
echo "STEP 7: Extract final RF2 PDBs"
echo "============================================================"

for ITEM in "${REGIMES[@]}"; do

    NAME="${ITEM%%|*}"

    INPUT="$RF2DIR/${NAME}_rf2.qv"
    OUTDIR="$FINALDIR/$NAME"

    if [ ! -s "$INPUT" ]; then
        continue
    fi

    rm -rf "$OUTDIR"
    mkdir -p "$OUTDIR"

    uv run qvextract \
        "$INPUT" \
        -o "$OUTDIR"

    echo "$NAME: $(find "$OUTDIR" -maxdepth 1 -name '*.pdb' | wc -l) PDBs"

done


# ============================================================
# SUMMARY
# ============================================================

echo
echo "============================================================"
echo "SUMMARY"
echo "============================================================"
echo "6I58 H3 CAMPAIGN COMPLETE"
echo "============================================================"

printf "%-10s %-12s %-12s %-12s %-12s\n" \
    "REGIME" \
    "RFDIFF" \
    "FILTERED" \
    "MPNN" \
    "RF2"

for ITEM in "${REGIMES[@]}"; do

    NAME="${ITEM%%|*}"

    RFN="$(qv_count "$RFDIR/${NAME}_rfdiffusion.qv")"
    FN="$(qv_count "$FILTERDIR/${NAME}_filtered.qv")"
    MN="$(qv_count "$MPNNDIR/${NAME}_mpnn.qv")"
    RN="$(qv_count "$RF2DIR/${NAME}_rf2.qv")"

    printf "%-10s %-12s %-12s %-12s %-12s\n" \
        "$NAME" "$RFN" "$FN" "$MN" "$RN"

done

echo
echo "Clean target:"
echo "$TARGET"

echo
echo "RF2 scorefiles:"
find "$RF2DIR" -maxdepth 1 -name '*_rf2.sc' -print | sort