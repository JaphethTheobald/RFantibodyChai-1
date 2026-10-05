#!/usr/bin/env python3
"""Filter final RF2 designs, duplicate backbones, and copy matching PDBs."""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Design:
    tag: str
    cdr_rmsd: float
    antibody_rmsd: float
    interaction_pae: float
    source_scorefile: Path

    @property
    def composite(self) -> float:
        return self.cdr_rmsd * 2 + self.antibody_rmsd * 1.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter RF2 scorefiles and copy matching final PDBs."
    )
    parser.add_argument("score_dir", type=Path, help="Directory containing RF2 .sc files")
    parser.add_argument("pdb_dir", type=Path, help="Directory containing final PDBs")
    parser.add_argument("output_dir", type=Path, help="Directory for filtered PDBs")
    parser.add_argument("--cdr-rmsd-max", type=float, default=2.05)
    parser.add_argument("--antibody-rmsd-max", type=float, default=4.0)
    parser.add_argument("--interaction-pae-max", type=float, default=10.0)
    parser.add_argument("--duplicate-min-count", type=int, default=4)
    parser.add_argument("--duplicate-keep-count", type=int, default=2)
    parser.add_argument("--pae-delta", type=float, default=3.0)
    parser.add_argument("--composite-delta", type=float, default=0.3)
    return parser.parse_args()


def read_designs(scorefiles: list[Path], args: argparse.Namespace) -> list[Design]:
    required = {
        "tag",
        "target_aligned_cdr_rmsd",
        "target_aligned_antibody_rmsd",
        "interaction_pae",
    }
    designs = []
    for scorefile in scorefiles:
        with scorefile.open(encoding="utf-8") as handle:
            header = handle.readline().split()
            if not required.issubset(header):
                missing = required - set(header)
                raise ValueError(f"{scorefile}: missing score columns: {sorted(missing)}")
            for line in handle:
                values = line.split()
                if not values:
                    continue
                if len(values) != len(header):
                    raise ValueError(f"{scorefile}: invalid score row: {line.rstrip()}")
                row = dict(zip(header, values))
                try:
                    design = Design(
                        tag=row["tag"],
                        cdr_rmsd=float(row["target_aligned_cdr_rmsd"]),
                        antibody_rmsd=float(row["target_aligned_antibody_rmsd"]),
                        interaction_pae=float(row["interaction_pae"]),
                        source_scorefile=scorefile,
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"{scorefile}: invalid score row: {row}") from exc
                if (
                    design.cdr_rmsd <= args.cdr_rmsd_max
                    and design.antibody_rmsd <= args.antibody_rmsd_max
                    and design.interaction_pae <= args.interaction_pae_max
                ):
                    designs.append(design)
    return designs


def backbone_id(tag: str) -> str:
    match = re.fullmatch(r"samples_design_(\d+)_dldesign_\d+_best", tag)
    if not match:
        raise ValueError(
            f"Tag does not match samples_design_AAAAA_dldesign_BBBBB_best: {tag}"
        )
    return match.group(1)


def reduce_duplicate_group(
    designs: list[Design], args: argparse.Namespace
) -> list[Design]:
    keep_count = (
        args.duplicate_keep_count
        if len(designs) >= args.duplicate_min_count
        else 1
    )
    if len(designs) <= keep_count:
        return designs
    remaining = list(designs)
    while len(remaining) > keep_count:
        smallest_pae = min(design.interaction_pae for design in remaining)
        reference = min(
            remaining,
            key=lambda design: (design.interaction_pae, design.composite, design.tag),
        )
        candidates = [design for design in remaining if design != reference]
        worst = max(
            candidates,
            key=lambda design: (
                2 if design.interaction_pae - smallest_pae > args.pae_delta
                else 1 if abs(design.composite - reference.composite) > args.composite_delta
                else 0,
                design.interaction_pae if design.interaction_pae - smallest_pae > args.pae_delta
                else design.composite if abs(design.composite - reference.composite) > args.composite_delta
                else design.interaction_pae,
                design.tag,
            ),
        )
        remaining.remove(worst)
    if len(remaining) not in {1, 2}:
        raise AssertionError(f"Duplicate filtering kept {len(remaining)} designs")
    return sorted(remaining, key=lambda design: design.tag)


def select_designs(designs: list[Design], args: argparse.Namespace) -> list[Design]:
    groups: dict[str, list[Design]] = defaultdict(list)
    for design in designs:
        groups[backbone_id(design.tag)].append(design)
    selected = []
    for group in groups.values():
        selected.extend(reduce_duplicate_group(group, args))
    return sorted(selected, key=lambda design: design.tag)


def main() -> int:
    args = parse_args()
    if args.duplicate_keep_count not in {1, 2}:
        raise ValueError("--duplicate-keep-count must be 1 or 2")
    scorefiles = sorted(args.score_dir.glob("*_rf2.sc"))
    if not scorefiles:
        raise ValueError(f"No RF2 scorefiles found in {args.score_dir}")
    designs = read_designs(scorefiles, args)
    selected = select_designs(designs, args)

    pdbs_by_tag: dict[str, list[Path]] = defaultdict(list)
    for pdb in sorted(args.pdb_dir.rglob("*.pdb")):
        pdbs_by_tag[pdb.stem].append(pdb)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for existing_pdb in args.output_dir.glob("*.pdb"):
        existing_pdb.unlink()
    copied = []
    for design in selected:
        matches = pdbs_by_tag.get(design.tag, [])
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one PDB for {design.tag}, found {len(matches)}"
            )
        destination = args.output_dir / matches[0].name
        shutil.copy2(matches[0], destination)
        copied.append((design, destination))

    report = args.output_dir / "filter_report.tsv"
    with report.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["tag", "cdr_rmsd", "antibody_rmsd", "interaction_pae", "composite", "source_scorefile", "pdb"])
        for design, pdb in copied:
            writer.writerow([
                design.tag, design.cdr_rmsd, design.antibody_rmsd,
                design.interaction_pae, design.composite,
                design.source_scorefile, pdb,
            ])
    print(f"Score rows passing thresholds: {len(designs)}")
    print(f"Selected after duplicate filtering: {len(selected)}")
    print(f"Copied PDBs: {len(copied)}")
    print(f"Report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())