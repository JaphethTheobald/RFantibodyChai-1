#!/usr/bin/env python3
"""Recursively run Chai-1 validation on VHH complex PDBs.

Each input PDB must contain a VHH in chain H.  Every other protein chain is
treated as target context.  By default Chai receives sequences only, with no
templates.  ``--template-target`` is available when a target conformation must
be preserved; it templates only the non-H target chains, never the VHH or
complex.

Examples
--------
# Blind broad screen: one diffusion sample per PDB, five trunk recycles.
CUDA_VISIBLE_DEVICES=2 \
  ~/RFantibody/.chai-venv/bin/python chai_recursive_vhh_validate.py \
  /home/schrogpu/Documents/RFantibody_outputs/Best //home/schrogpu/Documents/RFantibody_outputs

# Re-run just the survivors with five Chai samples each.
CUDA_VISIBLE_DEVICES=2 \
  ~/RFantibody/.chai-venv/bin/python chai_recursive_vhh_validate.py \
  /path/to/survivors /path/to/chai_revalidation_5x --num-samples 5

# Optional: preserve each input target's conformation without templating H.
CUDA_VISIBLE_DEVICES=2 \
  ~/RFantibody/.chai-venv/bin/python chai_recursive_vhh_validate.py \
  /path/to/survivors /path/to/chai_revalidation_templated --template-target
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import shutil
import sys
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np


AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M",
}


def open_pdb(path: Path):
    return gzip.open(path, "rt") if path.suffix == ".gz" else path.open()


def pdb_protein_chains(path: Path) -> OrderedDict[str, str]:
    """Return polymer sequences built from coordinate-bearing PDB residues."""
    residues: OrderedDict[str, OrderedDict[tuple[str, str], str]] = OrderedDict()
    with open_pdb(path) as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")) or len(line) < 27:
                continue
            resname = line[17:20].strip().upper()
            if resname not in AA3_TO_1:
                continue
            chain = line[21].strip()
            if not chain:
                raise ValueError(f"{path}: blank chain IDs are not supported")
            residue_id = (line[22:26].strip(), line[26].strip())
            residues.setdefault(chain, OrderedDict()).setdefault(residue_id, AA3_TO_1[resname])
    return OrderedDict((chain, "".join(sequence.values())) for chain, sequence in residues.items() if sequence)


def safe_stem(input_root: Path, pdb: Path) -> str:
    rel = pdb.relative_to(input_root).as_posix()
    for suffix in (".gz", ".pdb"):
        if rel.lower().endswith(suffix):
            rel = rel[: -len(suffix)]
    return rel.replace("/", "__")


def write_fasta(path: Path, chains: OrderedDict[str, str]) -> None:
    lines: list[str] = []
    for chain, sequence in chains.items():
        lines.extend((f">protein|name={chain}", sequence))
    path.write_text("\n".join(lines) + "\n")


def template_identifier(pdb: Path) -> str:
    # No underscores: Chai parses the m8 subject as IDENTIFIER_CHAIN.
    return "T" + hashlib.sha1(str(pdb.resolve()).encode()).hexdigest()[:15].upper()


def write_target_template(
    pdb: Path,
    target_chains: OrderedDict[str, str],
    cif_dir: Path,
    m8_dir: Path,
) -> Path:
    """Write one custom template plus its target-chain-only m8 hits."""
    try:
        import gemmi
    except ImportError as exc:
        raise SystemExit("Target templating requires gemmi (included in the Chai environment).") from exc

    identifier = template_identifier(pdb)
    cif = cif_dir / f"{identifier}.cif"
    cif_gz = cif_dir / f"{identifier}.cif.gz"
    structure = gemmi.read_structure(str(pdb))
    structure.make_mmcif_document().write_file(str(cif))
    with cif.open("rb") as source, gzip.open(cif_gz, "wb") as destination:
        shutil.copyfileobj(source, destination)

    # Both names are intentional: Chai 0.6.1 has code paths that look for
    # .cif, whereas the current public documentation specifies .cif.gz.
    rows = []
    for chain, sequence in target_chains.items():
        n = len(sequence)
        rows.append("\t".join((
            chain, f"{identifier}_{chain}", "100.000", str(n), "0", "0",
            "1", str(n), "1", str(n), "0.0", "1000.0",
        )))
    m8 = m8_dir / f"{identifier}.m8"
    m8.write_text("\n".join(rows) + "\n")
    return m8


@contextmanager
def temporary_environment(name: str, value: str) -> Iterator[None]:
    old_value = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if old_value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old_value


def scalar_scores(output_dir: Path) -> list[dict[str, object]]:
    """Extract scalar values without assuming a particular Chai score schema."""
    rows: list[dict[str, object]] = []
    for score_path in sorted(output_dir.glob("scores.model_idx_*.npz")):
        row: dict[str, object] = {"score_file": str(score_path)}
        with np.load(score_path, allow_pickle=False) as scores:
            for key in scores.files:
                value = scores[key]
                if value.size == 1:
                    item = value.item()
                    row[key] = item.item() if hasattr(item, "item") else item
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_dir", type=Path, help="Recursively searched directory of PDB or PDB.GZ files")
    parser.add_argument("output_dir", type=Path, help="New parent directory for Chai results")
    parser.add_argument("--vhh-chain", default="H", help="VHH PDB chain ID (default: H)")
    parser.add_argument("--num-recycles", type=int, default=5, help="Chai trunk recycles (default: 5)")
    parser.add_argument("--num-samples", type=int, default=1, help="Chai diffusion samples/PDB (default: 1)")
    parser.add_argument("--timesteps", type=int, default=200, help="Chai diffusion steps (default: 200)")
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--template-target", action="store_true", help="Template non-H target chains; H remains untemplated")
    parser.add_argument("--resume", action="store_true", help="Skip PDBs with existing Chai score NPZ files")
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {args.input_dir}")
    if args.output_dir.resolve() == args.input_dir.resolve():
        raise SystemExit("Output directory must differ from input directory")
    if args.num_recycles < 1 or args.num_samples < 1 or args.timesteps < 1:
        raise SystemExit("--num-recycles, --num-samples, and --timesteps must be positive")

    pdbs = sorted({*args.input_dir.rglob("*.pdb"), *args.input_dir.rglob("*.pdb.gz")})
    if not pdbs:
        raise SystemExit(f"No .pdb or .pdb.gz files found beneath {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fasta_dir = args.output_dir / "fastas"
    raw_dir = args.output_dir / "raw"
    work_dir = args.output_dir / "template_work"
    fasta_dir.mkdir(exist_ok=True)
    raw_dir.mkdir(exist_ok=True)

    jobs: list[tuple[Path, str, OrderedDict[str, str], Path | None]] = []
    cif_dir = work_dir / "cifs"
    m8_dir = work_dir / "m8"
    if args.template_target:
        cif_dir.mkdir(parents=True, exist_ok=True)
        m8_dir.mkdir(parents=True, exist_ok=True)

    for pdb in pdbs:
        chains = pdb_protein_chains(pdb)
        if args.vhh_chain not in chains:
            print(f"SKIP (no VHH chain {args.vhh_chain!r}): {pdb}", file=sys.stderr)
            continue
        if len(chains[args.vhh_chain]) < 60:
            print(f"SKIP (chain {args.vhh_chain!r} is only {len(chains[args.vhh_chain])} aa): {pdb}", file=sys.stderr)
            continue
        targets = OrderedDict((chain, sequence) for chain, sequence in chains.items() if chain != args.vhh_chain)
        if not targets:
            print(f"SKIP (no protein target chains besides {args.vhh_chain!r}): {pdb}", file=sys.stderr)
            continue
        stem = safe_stem(args.input_dir, pdb)
        fasta = fasta_dir / f"{stem}.fasta"
        write_fasta(fasta, chains)
        m8 = write_target_template(pdb, targets, cif_dir, m8_dir) if args.template_target else None
        jobs.append((pdb, stem, chains, m8))

    if not jobs:
        raise SystemExit("No valid VHH–target PDBs remained after parsing")

    # Chai 0.6.1 may resolve this variable while importing template modules,
    # so set it before importing chai_lab.
    environment = temporary_environment("CHAI_TEMPLATE_CIF_FOLDER", str(cif_dir)) if args.template_target else temporary_environment("CHAI_TEMPLATE_CIF_FOLDER", "")
    summary_rows: list[dict[str, object]] = []
    with environment:
        from chai_lab.chai1 import run_inference

        for job_index, (pdb, stem, chains, m8) in enumerate(jobs):
            output = raw_dir / stem
            existing = sorted(output.glob("scores.model_idx_*.npz")) if output.is_dir() else []
            if existing and args.resume:
                print(f"RESUME: {pdb}", flush=True)
            elif output.exists() and any(output.iterdir()):
                print(f"SKIP (partial/nonempty output; remove it or use --resume if complete): {output}", file=sys.stderr)
                continue
            else:
                print(f"[{job_index + 1}/{len(jobs)}] Chai: {pdb}", flush=True)
                run_inference(
                    fasta_file=fasta_dir / f"{stem}.fasta",
                    output_dir=output,
                    template_hits_path=m8,
                    use_templates_server=False,
                    use_esm_embeddings=True,
                    num_trunk_recycles=args.num_recycles,
                    num_diffn_timesteps=args.timesteps,
                    num_diffn_samples=args.num_samples,
                    seed=args.seed + job_index,
                    device=args.device,
                )

            for scores in scalar_scores(output):
                summary_rows.append({
                    "input_pdb": str(pdb),
                    "vhh_chain": args.vhh_chain,
                    "target_chains": ",".join(chain for chain in chains if chain != args.vhh_chain),
                    "template_target": args.template_target,
                    "raw_output_dir": str(output),
                    **scores,
                })

    summary = args.output_dir / "chai_summary.tsv"
    fields = sorted({key for row in summary_rows for key in row})
    with summary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary_rows)
    manifest = args.output_dir / "run_manifest.json"
    manifest.write_text(json.dumps({
        "input_dir": str(args.input_dir.resolve()),
        "vhh_chain": args.vhh_chain,
        "num_recycles": args.num_recycles,
        "num_samples": args.num_samples,
        "timesteps": args.timesteps,
        "template_target": args.template_target,
        "valid_jobs": len(jobs),
    }, indent=2) + "\n")
    print(f"\nCompleted {len(summary_rows)} model(s). Summary: {summary}")


if __name__ == "__main__":
    main()
