#!/usr/bin/env python3
"""Convert protein PDB files into AlphaFold Server batch JSON.

Each input PDB becomes one AlphaFold Server job. Each protein chain becomes a
separate proteinChain entity. Coordinates are deliberately not supplied as
templates, so the resulting AlphaFold prediction is an independent sequence-
based check of the designed complex.

python3 /home/schrogpu/RFantibody/pdbs_to_afserver_json.py \
  /home/schrogpu/Documents/RFantibody_outputs/7qot_chai_templated2/08_filtered_PDBs_score_review \
  --output /home/schrogpu/Documents/RFantibody_outputs/7qot_chai_templated2/08_filtered_PDBs_score_review/afserver_jobs.json \
  --recursive
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import OrderedDict
from pathlib import Path


AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    # Common protein residue aliases that AlphaFold expects as canonical AAs.
    "MSE": "M", "HSD": "H", "HSE": "H", "HSP": "H", "SEC": "C",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Make upload-ready AlphaFold Server JSON from final protein PDBs. "
            "Every PDB is one job and every selected chain is one protein entity."
        )
    )
    parser.add_argument(
        "inputs", nargs="+", type=Path,
        help="PDB file(s), and/or directories containing PDB files",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("afserver_jobs.json"),
        help="output JSON path (default: afserver_jobs.json)",
    )
    parser.add_argument(
        "--glob", default="*.pdb",
        help="filename pattern used for directory inputs (default: *.pdb)",
    )
    parser.add_argument(
        "--recursive", action="store_true",
        help="search directory inputs recursively",
    )
    parser.add_argument(
        "--chains",
        help="comma-separated PDB chain IDs to keep, in the desired JSON order",
    )
    parser.add_argument(
        "--allow-templates", action="store_true",
        help="allow AlphaFold Server to use structure templates (off by default)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=30,
        help="maximum jobs per JSON file; 0 means no splitting (default: 30)",
    )
    parser.add_argument(
        "--manifest", type=Path,
        help="TSV manifest path (default: <output_stem>_manifest.tsv)",
    )
    return parser.parse_args()


def collect_pdbs(inputs: list[Path], pattern: str, recursive: bool) -> list[Path]:
    found: list[Path] = []
    for item in inputs:
        if item.is_file():
            found.append(item)
        elif item.is_dir():
            iterator = item.rglob(pattern) if recursive else item.glob(pattern)
            found.extend(p for p in iterator if p.is_file())
        else:
            raise ValueError(f"Input does not exist: {item}")
    # Resolve before de-duplicating so the same file is not submitted twice.
    return sorted(set(p.resolve() for p in found), key=lambda p: str(p))


def pdb_chain_sequences(path: Path) -> OrderedDict[str, str]:
    """Extract unique residues from ATOM/HETATM records in the first model."""
    chains: OrderedDict[str, list[str]] = OrderedDict()
    seen_residues: set[tuple[str, str, str]] = set()
    saw_model = False

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, 1):
            record = line[0:6].strip()
            if record == "MODEL":
                if saw_model:
                    break
                saw_model = True
                continue
            if record == "ENDMDL":
                break
            if record not in {"ATOM", "HETATM"} or len(line) < 27:
                continue

            residue_name = line[17:20].strip().upper()
            if residue_name not in AA3_TO_1:
                # Waters, glycans, ions, and other non-protein HETATMs are ignored.
                if record == "HETATM":
                    continue
                raise ValueError(
                    f"{path}:{line_number}: unsupported protein residue "
                    f"{residue_name!r}; convert it to a standard amino acid first"
                )

            chain_id = line[21].strip() or "_"
            residue_number = line[22:26].strip()
            insertion_code = line[26].strip()
            residue_key = (chain_id, residue_number, insertion_code)
            if residue_key in seen_residues:
                continue
            seen_residues.add(residue_key)
            chains.setdefault(chain_id, []).append(AA3_TO_1[residue_name])

    if not chains:
        raise ValueError(f"No protein residues found in {path}")
    return OrderedDict((chain, "".join(sequence)) for chain, sequence in chains.items())


def safe_job_name(stem: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("_.-")
    return name or "alphafold_job"


def output_paths(base: Path, job_count: int, batch_size: int) -> list[Path]:
    if not base.suffix:
        base = base.with_suffix(".json")
    if batch_size <= 0 or job_count <= batch_size:
        return [base]
    number = (job_count + batch_size - 1) // batch_size
    width = max(3, len(str(number)))
    return [
        base.with_name(f"{base.stem}_part{i:0{width}d}{base.suffix}")
        for i in range(1, number + 1)
    ]


def main() -> int:
    args = parse_args()
    if args.batch_size < 0:
        raise ValueError("--batch-size must be zero or a positive integer")

    requested_chains = None
    if args.chains:
        requested_chains = [x.strip() or "_" for x in args.chains.split(",")]
        if len(requested_chains) != len(set(requested_chains)):
            raise ValueError("--chains contains a duplicate chain ID")

    pdbs = collect_pdbs(args.inputs, args.glob, args.recursive)
    if not pdbs:
        raise ValueError("No PDB files matched the supplied inputs")

    jobs = []
    manifest_rows = []
    used_names: dict[str, int] = {}

    for pdb_path in pdbs:
        all_chains = pdb_chain_sequences(pdb_path)
        if requested_chains is None:
            selected = all_chains
        else:
            missing = [c for c in requested_chains if c not in all_chains]
            if missing:
                raise ValueError(
                    f"{pdb_path}: requested chain(s) not found: {', '.join(missing)}; "
                    f"available: {', '.join(all_chains)}"
                )
            selected = OrderedDict((c, all_chains[c]) for c in requested_chains)

        base_name = safe_job_name(pdb_path.stem)
        used_names[base_name] = used_names.get(base_name, 0) + 1
        job_name = base_name
        if used_names[base_name] > 1:
            job_name = f"{base_name}_{used_names[base_name]}"

        entities = []
        for chain_id, sequence in selected.items():
            protein = {
                "sequence": sequence,
                "count": 1,
                "useStructureTemplate": bool(args.allow_templates),
            }
            entities.append({"proteinChain": protein})
            manifest_rows.append({
                "job_name": job_name,
                "source_pdb": str(pdb_path),
                "chain_id": chain_id,
                "length": len(sequence),
                "sequence": sequence,
            })

        jobs.append({
            "name": job_name,
            "modelSeeds": [],
            "sequences": entities,
            "dialect": "alphafoldserver",
            "version": 1,
        })

    paths = output_paths(args.output, len(jobs), args.batch_size)
    chunk_size = len(jobs) if args.batch_size <= 0 else args.batch_size
    for index, path in enumerate(paths):
        path.parent.mkdir(parents=True, exist_ok=True)
        chunk = jobs[index * chunk_size:(index + 1) * chunk_size]
        with path.open("w", encoding="utf-8") as handle:
            json.dump(chunk, handle, indent=2)
            handle.write("\n")

    manifest = args.manifest or args.output.with_name(
        f"{args.output.stem}_manifest.tsv"
    )
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["job_name", "source_pdb", "chain_id", "length", "sequence"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Converted {len(pdbs)} PDB(s) into {len(jobs)} job(s).")
    for path in paths:
        print(f"JSON: {path}")
    print(f"Manifest: {manifest}")
    print("Templates: " + ("allowed" if args.allow_templates else "disabled"))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
