#!/usr/bin/env python3
"""Run the existing Chai campaign validator with a target-only template.

The wrapped validator still creates the same RF2-compatible scorefile and
selects the same single PDB per design.  This adapter supplies exactly one
template hit for the target entity (normally ``T``), while deliberately leaving
the designed VHH untemplated and adding no inter-chain restraints.

It is written as an adapter so ``chai_validate_campaign.py`` remains the
single owner of scoring and output selection.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path

import gemmi

AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M",
}


def read_fasta(path: Path) -> str:
    sequence = "".join(
        line.strip().replace(" ", "").upper()
        for line in path.read_text().splitlines()
        if line.strip() and not line.startswith(">")
    )
    if not sequence or any(letter not in "ACDEFGHIKLMNPQRSTVWY" for letter in sequence):
        raise ValueError(f"{path} must contain one canonical protein sequence")
    return sequence


def chain_sequence(chain: gemmi.Chain) -> str:
    return "".join(
        AA3_TO_1[residue.name.upper()]
        for residue in chain
        if residue.name.upper() in AA3_TO_1
    )


def smith_waterman_matches(reference: str, query: str) -> list[tuple[int, int]]:
    """Return exact-residue matches from a simple local alignment."""
    score = [[0] * (len(query) + 1) for _ in range(len(reference) + 1)]
    trace = [[0] * (len(query) + 1) for _ in range(len(reference) + 1)]
    best = (0, 0, 0)
    for i, ref in enumerate(reference, start=1):
        for j, obs in enumerate(query, start=1):
            choices = (
                0,
                score[i - 1][j - 1] + (3 if ref == obs else -3),
                score[i - 1][j] - 2,
                score[i][j - 1] - 2,
            )
            step = max(range(4), key=choices.__getitem__)
            score[i][j] = choices[step]
            trace[i][j] = step
            if score[i][j] > best[0]:
                best = (score[i][j], i, j)
    _, i, j = best
    matches: list[tuple[int, int]] = []
    while i and j and score[i][j] > 0:
        step = trace[i][j]
        if step == 1:
            if reference[i - 1] == query[j - 1]:
                matches.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif step == 2:
            i -= 1
        elif step == 3:
            j -= 1
        else:
            break
    return list(reversed(matches))


def choose_target_chain(structure: gemmi.Structure, target_sequence: str) -> tuple[str, str, list[tuple[int, int]]]:
    candidates: list[tuple[int, str, str, list[tuple[int, int]]]] = []
    for chain in structure[0]:
        sequence = chain_sequence(chain)
        if sequence:
            matches = smith_waterman_matches(target_sequence, sequence)
            candidates.append((len(matches), chain.name, sequence, matches))
    if not candidates:
        raise ValueError("No protein chains were found in the template PDB")
    matched, chain_name, sequence, mapping = max(candidates, key=lambda item: item[0])
    minimum = max(25, int(0.60 * min(len(target_sequence), len(sequence))))
    if matched < minimum:
        raise ValueError(
            "No template chain maps sufficiently to the target FASTA; "
            f"best chain {chain_name!r} has {matched} exact aligned residues"
        )
    return chain_name, sequence, mapping


def write_custom_template(
    template_pdb: Path,
    target_fasta: Path,
    work_dir: Path,
    identifier: str,
    target_entity_name: str,
    prepared_cif_gz: Path | None = None,
) -> tuple[Path, Path]:
    if not identifier.isalnum():
        raise ValueError("--template-identifier must contain only letters and digits")
    target_sequence = read_fasta(target_fasta)
    if prepared_cif_gz is not None:
        if not prepared_cif_gz.is_file():
            raise FileNotFoundError(f"Prepared template CIF not found: {prepared_cif_gz}")
        with gzip.open(prepared_cif_gz, "rb") as source:
            cif_text = source.read().decode("utf-8")
        prepared_cif = work_dir / f"{identifier}.prepared.cif"
        prepared_cif.parent.mkdir(parents=True, exist_ok=True)
        prepared_cif.write_text(cif_text)
        structure = gemmi.read_structure(str(prepared_cif))
        prepared_cif.unlink()
        structure.name = str(prepared_cif_gz)
    else:
        structure = gemmi.read_structure(str(template_pdb))
        structure.name = str(template_pdb)
    source_chain, source_sequence, mapping = choose_target_chain(structure, target_sequence)

    # Chai reads custom structures from <identifier>.cif.gz.  The m8 records
    # only this source chain, so any other chains in the source PDB cannot act
    # as templates for the VHH or as a complex template.
    cif_dir = work_dir / "cif"
    cif_dir.mkdir(parents=True, exist_ok=True)
    cif_gz = cif_dir / f"{identifier}.cif.gz"
    if prepared_cif_gz is not None:
        shutil.copy2(prepared_cif_gz, cif_gz)
    else:
        cif = cif_dir / f"{identifier}.cif"
        structure.make_mmcif_document().write_file(str(cif))
        with cif.open("rb") as source, gzip.open(cif_gz, "wb") as destination:
            destination.write(source.read())
        cif.unlink()

    # Chai's template parser uses the input entity name as the m8 query and
    # the identifier_chain pair as the subject.  The alignment bounds are
    # 1-based inclusive, following the usual MMseqs m8 convention.
    query_positions = [query for query, _ in mapping]
    template_positions = [template for _, template in mapping]
    query_start, query_end = min(query_positions) + 1, max(query_positions) + 1
    template_start, template_end = min(template_positions) + 1, max(template_positions) + 1
    alignment_length = len(mapping)
    identity = 100.0 * alignment_length / max(1, max(query_end - query_start + 1, template_end - template_start + 1))
    m8 = work_dir / "target_only_template.m8"
    m8.parent.mkdir(parents=True, exist_ok=True)
    m8.write_text(
        "\t".join(
            (
                target_entity_name,
                f"{identifier}_{source_chain}",
                f"{identity:.3f}",
                str(alignment_length),
                "0", "0",
                str(query_start), str(query_end),
                str(template_start), str(template_end),
                "0.0", "1000.0",
            )
        )
        + "\n"
    )
    manifest = {
        "identifier": identifier,
        "target_entity_name": target_entity_name,
        "source_pdb": str(template_pdb.resolve()),
        "prepared_cif_gz": str(prepared_cif_gz.resolve()) if prepared_cif_gz else None,
        "source_chain": source_chain,
        "target_fasta": str(target_fasta.resolve()),
        "target_length": len(target_sequence),
        "template_chain_length": len(source_sequence),
        "aligned_exact_residues": alignment_length,
        "target_sha256": hashlib.sha256(target_sequence.encode()).hexdigest(),
        "m8": str(m8),
        "cif_gz": str(cif_gz),
    }
    (work_dir / "template_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return m8, cif_dir


@contextmanager
def temporary_environment(name: str, value: str):
    old_value = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if old_value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old_value


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--template-pdb", type=Path, required=True)
    parser.add_argument("--template-cif-gz", type=Path)
    parser.add_argument("--template-target-fasta", type=Path, required=True)
    parser.add_argument("--template-work-dir", type=Path, required=True)
    parser.add_argument("--template-identifier", required=True)
    parser.add_argument("--template-target-name", default="T")
    template_args, validator_args = parser.parse_known_args()
    for path in (template_args.template_pdb, template_args.template_target_fasta):
        if not path.is_file() or path.stat().st_size == 0:
            raise SystemExit(f"Template input is missing or empty: {path}")
    if template_args.template_cif_gz is not None and not template_args.template_cif_gz.is_file():
        raise SystemExit(f"Prepared template CIF is missing: {template_args.template_cif_gz}")

    m8, cif_dir = write_custom_template(
        template_args.template_pdb,
        template_args.template_target_fasta,
        template_args.template_work_dir,
        template_args.template_identifier,
        template_args.template_target_name,
        template_args.template_cif_gz,
    )
    print(f"Chai target-only template: {m8}", flush=True)

    # Import the existing validator from the same RFantibody/scripts directory.
    # Its per-design function imports run_inference lazily, so replacing the
    # module attribute here cleanly supplies the target template to every job.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import chai_lab.chai1 as chai1
    import chai_lab.data.io.rcsb as rcsb
    import chai_validate_campaign as validator

    original_run_inference = chai1.run_inference
    original_download_cif_file = rcsb.download_cif_file

    def download_local_template_or_rcsb(pdb_id: str, directory: Path) -> Path:
        if pdb_id.upper() == template_args.template_identifier.upper():
            local_cif = cif_dir / f"{template_args.template_identifier}.cif.gz"
            if not local_cif.is_file():
                raise FileNotFoundError(f"Prepared local template is missing: {local_cif}")
            return local_cif
        return original_download_cif_file(pdb_id, directory)

    rcsb.download_cif_file = download_local_template_or_rcsb

    def run_inference_with_target_template(*args, **kwargs):
        supplied = kwargs.get("template_hits_path")
        if supplied is not None and Path(supplied) != m8:
            raise ValueError("The templated adapter owns template_hits_path")
        kwargs["template_hits_path"] = m8
        kwargs["use_templates_server"] = False
        return original_run_inference(*args, **kwargs)

    chai1.run_inference = run_inference_with_target_template
    sys.argv = [sys.argv[0], *validator_args]
    
    with temporary_environment("CHAI_TEMPLATE_CIF_FOLDER", str(cif_dir)):
        validator.main()


if __name__ == "__main__":
    main()
