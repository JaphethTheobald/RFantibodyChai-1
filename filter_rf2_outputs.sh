#!/usr/bin/env bash
set -euo pipefail

read -r -e -p "Path to the RF2 output directory: " RF2_DIR
RF2_DIR="${RF2_DIR%/}"

[[ -d "$RF2_DIR" ]] || {
  echo "ERROR: Not a directory: $RF2_DIR" >&2
  exit 1
}

export RF2_DIR

python3 <<'PY'
import csv
import os
import shutil
import sys
from pathlib import Path

root = Path(os.environ["RF2_DIR"]).resolve()
outdir = root / "filtered_interaction_pae_lt10_TAARMSD_lt2A"

if outdir.exists():
    sys.exit(f"ERROR: Output folder already exists:\n  {outdir}")

# Find RF2 score files such as 1jbu_vhh_50designs_rf2.sc.
score_files = list(root.rglob("*_rf2.sc"))
if not score_files:
    score_files = list(root.rglob("*.sc"))

if not score_files:
    sys.exit(f"ERROR: No RF2 .sc score file found under:\n  {root}")

outdir.mkdir()
selected = []
unmatched = []

for score_file in score_files:
    with score_file.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")

        required = {
            "interaction_pae",
            "target_aligned_antibody_rmsd",
            "tag",
        }
        actual = set(reader.fieldnames or [])
        missing = required - actual
        if missing:
            print(f"Skipping {score_file.name}: missing {', '.join(sorted(missing))}")
            continue

        for row in reader:
            interaction_pae = float(row["interaction_pae"])
            ta_armsd = float(row["target_aligned_antibody_rmsd"])
            tag = row["tag"]

            # Strict filters:
            # RF2 interface confidence and retention of designed binding pose.
            if not (interaction_pae < 10 and ta_armsd < 2):
                continue

            # RF2 PDB normally has the same basename as its score-file tag.
            candidates = list(root.rglob(f"{tag}.pdb"))
            if not candidates:
                unmatched.append(tag)
                continue

            pdb = candidates[0]
            destination = outdir / pdb.name
            if destination.exists():
                destination = outdir / f"{pdb.parent.name}_{pdb.name}"

            shutil.copy2(pdb, destination)
            selected.append({
                "pdb_file": destination.name,
                "tag": tag,
                "interaction_pae": interaction_pae,
                "target_aligned_antibody_rmsd": ta_armsd,
                "source_score_file": score_file.name,
            })

manifest = outdir / "selected_designs.tsv"
with manifest.open("w", newline="") as handle:
    fields = [
        "pdb_file",
        "tag",
        "interaction_pae",
        "target_aligned_antibody_rmsd",
        "source_score_file",
    ]
    writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
    writer.writeheader()
    writer.writerows(selected)

print(f"\nCopied {len(selected)} passing RF2 PDB(s) to:\n  {outdir}")
print(f"Manifest:\n  {manifest}")

if unmatched:
    print(f"\nWARNING: {len(unmatched)} passing design(s) lacked a matching PDB.")
    print("\n".join(unmatched[:20]))
PY