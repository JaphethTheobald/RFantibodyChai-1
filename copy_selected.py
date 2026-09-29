#!/usr/bin/env python3
from pathlib import Path
import re
import shutil
import sys


# RUN: 
# python3 copy_selected.py "/home/schrogpu/Documents/RFantibody_outputs/6i58_long_no_clash_with_light_chain_binding1_500_campaign/07_final_pdbs"
# paste in
# cntrl + D



if len(sys.argv) != 2:
    print(f"Usage: {Path(sys.argv[0]).name} /path/to/07_final_pdbs")
    raise SystemExit(2)

source_root = Path(sys.argv[1]).expanduser()
destination = source_root / "selected"
destination.mkdir(parents=True, exist_ok=True)

pasted_text = sys.stdin.read()
names = list(dict.fromkeys(re.findall(
    r"samples_design_\d+_dldesign_\d+_best",
    pasted_text,
)))

if not names:
    print("No design names found in pasted input.")
    raise SystemExit(1)

for name in names:
    matches = list(source_root.rglob(f"{name}.pdb"))

    if len(matches) == 1:
        shutil.copy2(matches[0], destination / matches[0].name)
        print(f"Copied: {matches[0].name}")
    elif not matches:
        print(f"MISSING: {name}.pdb")
    else:
        print(f"AMBIGUOUS — not copied: {name}.pdb")
        for match in matches:
            print(f"  {match}")

print(f"\nDone. Selected PDBs are in:\n{destination}")