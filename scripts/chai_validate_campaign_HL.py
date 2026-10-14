#!/usr/bin/env python3
"""Validate RFantibody/ProteinMPNN VHH designs with Chai-1.

This program deliberately keeps the RFantibody scorefile contract: one row and
one PDB per design tag. Chai itself is run without a complex template or
restraint; the RFdiffusion PDB is used only *after* prediction as the reference
for pose-recovery RMSDs. An optional second target chain can be supplied as
structural context; pose RMSD is still evaluated against primary target T.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import gemmi
import numpy as np


AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M",
}
LEGACY_COLUMNS = [
    "interaction_pae", "pae", "pred_lddt", "target_aligned_antibody_rmsd",
    "target_aligned_cdr_rmsd", "framework_aligned_antibody_rmsd",
    "framework_aligned_cdr_rmsd", "framework_aligned_H1_rmsd",
    "framework_aligned_H2_rmsd", "framework_aligned_H3_rmsd",
    "framework_aligned_L1_rmsd", "framework_aligned_L2_rmsd",
    "framework_aligned_L3_rmsd", "tag",
]


def read_fasta(path: Path) -> str:
    seq: list[str] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith(">"):
            seq.append(line.replace(" ", "").upper())
    result = "".join(seq)
    if not result or any(c not in "ACDEFGHIKLMNPQRSTVWY" for c in result):
        raise ValueError(f"{path} must contain one canonical protein sequence")
    return result


def chain_records(structure: gemmi.Structure, name: str) -> tuple[str, list[np.ndarray | None]]:
    for chain in structure[0]:
        if chain.name != name:
            continue
        sequence: list[str] = []
        coords: list[np.ndarray | None] = []
        for residue in chain:
            aa = AA3_TO_1.get(residue.name.upper())
            if aa is None:
                continue
            sequence.append(aa)
            ca = next((atom for atom in residue if atom.name == "CA"), None)
            coords.append(None if ca is None else np.array([ca.pos.x, ca.pos.y, ca.pos.z], dtype=float))
        if sequence:
            return "".join(sequence), coords
    raise ValueError(f"Chain {name!r} was not found in {structure.name or 'structure'}")


def local_matches(reference: str, query: str) -> dict[int, int]:
    """Smith-Waterman mapping, retaining only exact aligned residues."""
    n, m = len(reference), len(query)
    score = np.zeros((n + 1, m + 1), dtype=np.int32)
    trace = np.zeros((n + 1, m + 1), dtype=np.int8)
    best = (0, 0, 0)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            values = (0, score[i - 1, j - 1] + (3 if reference[i - 1] == query[j - 1] else -3), score[i - 1, j] - 2, score[i, j - 1] - 2)
            choice = int(np.argmax(values))
            score[i, j] = values[choice]
            trace[i, j] = choice
            if score[i, j] > best[0]:
                best = (int(score[i, j]), i, j)
    _, i, j = best
    pairs: dict[int, int] = {}
    while i and j and score[i, j] > 0:
        step = trace[i, j]
        if step == 1:
            if reference[i - 1] == query[j - 1]:
                pairs[i - 1] = j - 1
            i -= 1
            j -= 1
        elif step == 2:
            i -= 1
        elif step == 3:
            j -= 1
        else:
            break
    return pairs


def kabsch(mobile: np.ndarray, fixed: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cm = mobile.mean(axis=0)
    cf = fixed.mean(axis=0)
    u, _, vt = np.linalg.svd((mobile - cm).T @ (fixed - cf))
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    return rotation, cm, cf


def transformed_rmsd(mobile: np.ndarray, fixed: np.ndarray, rotation: np.ndarray, cm: np.ndarray, cf: np.ndarray) -> float:
    if not len(mobile):
        return math.nan
    moved = (mobile - cm) @ rotation + cf
    return float(np.sqrt(np.mean(np.sum((moved - fixed) ** 2, axis=1))))


def parse_cdrs(path: Path, h_length: int) -> dict[str, list[int]]:
    cdrs = {name: [] for name in ("H1", "H2", "H3")}
    for line in path.read_text().splitlines():
        if not line.startswith("REMARK PDBinfo-LABEL"):
            continue
        fields = line.split()
        if len(fields) < 4:
            continue
        try:
            index = int(fields[2]) - 1
        except ValueError:
            continue
        label = fields[3].upper()
        if label in cdrs and 0 <= index < h_length:
            cdrs[label].append(index)
    for label in cdrs:
        cdrs[label] = sorted(set(cdrs[label]))
    if not all(cdrs.values()):
        raise ValueError(f"Missing H1/H2/H3 REMARK PDBinfo-LABEL annotations in {path}")
    return cdrs


def scalar(value: Any) -> float:
    if hasattr(value, "detach"):
        value = value.detach().cpu().item()
    if isinstance(value, np.ndarray):
        value = value.item()
    return float(value)


def ranking_value(ranking: Any, field: str, default: float = math.nan) -> float:
    try:
        return scalar(getattr(ranking, field))
    except (AttributeError, TypeError, ValueError):
        return default


def score_candidate(
    ref_t_seq: str, ref_t_ca: list[np.ndarray | None], ref_h_seq: str, ref_h_ca: list[np.ndarray | None],
    pred_t_seq: str, pred_t_ca: list[np.ndarray | None], pred_h_seq: str, pred_h_ca: list[np.ndarray | None],
    cdrs: dict[str, list[int]], pae: np.ndarray, plddt: np.ndarray, target_context_length: int,
) -> dict[str, float]:
    t_map = local_matches(ref_t_seq, pred_t_seq)
    h_map = local_matches(ref_h_seq, pred_h_seq)
    common_t = [(i, j) for i, j in t_map.items() if ref_t_ca[i] is not None and pred_t_ca[j] is not None]
    if len(common_t) < 20:
        raise ValueError(f"Only {len(common_t)} mapped target CA atoms; target/reference sequence mapping failed")
    mobile_t = np.array([pred_t_ca[j] for _, j in common_t])
    fixed_t = np.array([ref_t_ca[i] for i, _ in common_t])
    rotation, cm_t, cf_t = kabsch(mobile_t, fixed_t)

    def h_rmsd(indices: list[int], rotation_: np.ndarray, cm_: np.ndarray, cf_: np.ndarray) -> float:
        pairs = [(i, h_map[i]) for i in indices if i in h_map and ref_h_ca[i] is not None and pred_h_ca[h_map[i]] is not None]
        if not pairs:
            return math.nan
        return transformed_rmsd(np.array([pred_h_ca[j] for _, j in pairs]), np.array([ref_h_ca[i] for i, _ in pairs]), rotation_, cm_, cf_)

    all_h = list(range(len(ref_h_seq)))
    cdr_all = sorted({i for values in cdrs.values() for i in values})
    framework = [i for i in all_h if i not in set(cdr_all)]
    fw_pairs = [(i, h_map[i]) for i in framework if i in h_map and ref_h_ca[i] is not None and pred_h_ca[h_map[i]] is not None]
    if len(fw_pairs) < 10:
        raise ValueError("Too few VHH framework atoms for framework alignment")
    fw_mobile = np.array([pred_h_ca[j] for _, j in fw_pairs])
    fw_fixed = np.array([ref_h_ca[i] for i, _ in fw_pairs])
    rotation_fw, cm_fw, cf_fw = kabsch(fw_mobile, fw_fixed)

    # Input order is T, optional L, then VHH H. Treat both T and L as target
    # context when calculating interface PAE.
    cross = np.concatenate((
        pae[:target_context_length, target_context_length:],
        pae[target_context_length:, :target_context_length],
    ), axis=None)
    mean_plddt = float(np.mean(plddt))
    if mean_plddt > 1.5:
        mean_plddt /= 100.0
    return {
        "interaction_pae": float(np.mean(cross)),
        "pae": float(np.mean(pae)),
        "pred_lddt": mean_plddt,
        "target_aligned_antibody_rmsd": h_rmsd(all_h, rotation, cm_t, cf_t),
        "target_aligned_cdr_rmsd": h_rmsd(cdr_all, rotation, cm_t, cf_t),
        "framework_aligned_antibody_rmsd": h_rmsd(all_h, rotation_fw, cm_fw, cf_fw),
        "framework_aligned_cdr_rmsd": h_rmsd(cdr_all, rotation_fw, cm_fw, cf_fw),
        "framework_aligned_H1_rmsd": h_rmsd(cdrs["H1"], rotation_fw, cm_fw, cf_fw),
        "framework_aligned_H2_rmsd": h_rmsd(cdrs["H2"], rotation_fw, cm_fw, cf_fw),
        "framework_aligned_H3_rmsd": h_rmsd(cdrs["H3"], rotation_fw, cm_fw, cf_fw),
        "framework_aligned_L1_rmsd": math.nan,
        "framework_aligned_L2_rmsd": math.nan,
        "framework_aligned_L3_rmsd": math.nan,
    }


def load_reference(pdb: Path, target_chain: str, vhh_chain: str) -> tuple[str, list[np.ndarray | None], str, list[np.ndarray | None], dict[str, list[int]]]:
    structure = gemmi.read_structure(str(pdb))
    structure.name = str(pdb)
    h_seq, h_ca = chain_records(structure, vhh_chain)
    t_seq, t_ca = chain_records(structure, target_chain)
    return t_seq, t_ca, h_seq, h_ca, parse_cdrs(pdb, len(h_seq))


def candidate_chains(
    cif: Path,
    target_sequence: str,
    vhh_sequence: str,
    additional_target_sequence: str | None = None,
) -> tuple[str, list[np.ndarray | None], str, list[np.ndarray | None], gemmi.Structure]:
    """Identify Chai output chains by sequence, not version-specific CIF IDs.

    Older Chai releases write A/B/C even when the FASTA entities are named
    T/L/H.  The returned structure is renamed to the RFantibody convention so
    the saved selected PDB remains usable by the existing final filter.
    """
    structure = gemmi.read_structure(str(cif))
    structure.name = str(cif)
    records: list[tuple[gemmi.Chain, str, list[np.ndarray | None]]] = []
    for chain in structure[0]:
        sequence, coords = chain_records(structure, chain.name)
        records.append((chain, sequence, coords))

    def choose(reference: str, used: set[int], label: str) -> int:
        choices = [
            (len(local_matches(reference, sequence)), index)
            for index, (_, sequence, _) in enumerate(records)
            if index not in used
        ]
        if not choices:
            raise ValueError(f"No predicted chain remains for {label} in {cif}")
        score, index = max(choices)
        minimum = max(20, int(0.60 * min(len(reference), len(records[index][1]))))
        if score < minimum:
            observed = ", ".join(f"{chain.name}:{len(sequence)}" for chain, sequence, _ in records)
            raise ValueError(
                f"Could not identify predicted {label} by sequence in {cif}; "
                f"best exact alignment has {score} residues (chains {observed})"
            )
        return index

    used: set[int] = set()
    target_index = choose(target_sequence, used, "primary target")
    used.add(target_index)
    additional_index: int | None = None
    if additional_target_sequence is not None:
        additional_index = choose(additional_target_sequence, used, "additional target")
        used.add(additional_index)
    vhh_index = choose(vhh_sequence, used, "VHH")

    # Canonicalize selected output PDB chain names for downstream RFantibody
    # scripts. Do this only after sequence matching, because names may start
    # as A/B/C in older Chai versions.
    records[target_index][0].name = "T"
    if additional_index is not None:
        records[additional_index][0].name = "L"
    records[vhh_index][0].name = "H"
    _, target_seq, target_ca = records[target_index]
    _, vhh_seq, vhh_ca = records[vhh_index]
    return target_seq, target_ca, vhh_seq, vhh_ca, structure


def save_selected_pdb(structure: gemmi.Structure, path: Path) -> None:
    # fasta_names_as_cif_chains=True makes these chain names deterministic.
    structure.write_pdb(str(path))


def finite_or_inf(value: float) -> float:
    return value if math.isfinite(value) else float("inf")


def one_design(
    pdb: Path, target_sequence: str, additional_target_sequence: str | None,
    raw_root: Path, pdb_out: Path, num_samples: int,
    recycles: int, timesteps: int, seed: int | None, resume: bool,
    target_chain: str, vhh_chain: str,
) -> dict[str, Any]:
    tag = pdb.stem
    result_path = raw_root / tag / "chai_result.json"
    if resume and result_path.is_file() and (pdb_out / f"{tag}.pdb").is_file():
        return json.loads(result_path.read_text())
    if result_path.parent.exists():
        shutil.rmtree(result_path.parent)
    result_path.parent.mkdir(parents=True)
    ref_t_seq, ref_t_ca, ref_h_seq, ref_h_ca, cdrs = load_reference(pdb, target_chain, vhh_chain)
    if not local_matches(ref_t_seq, target_sequence):
        raise ValueError(f"The reference target in {pdb} does not map to the full-target FASTA")
    fasta_dir = raw_root / "_fastas"
    fasta_dir.mkdir(exist_ok=True)
    fasta = fasta_dir / f"{tag}.fasta"
    fasta_parts = [f">protein|name=T\n{target_sequence}\n"]
    if additional_target_sequence is not None:
        fasta_parts.append(f">protein|name=L\n{additional_target_sequence}\n")
    fasta_parts.append(f">protein|name=H\n{ref_h_seq}\n")
    fasta.write_text("".join(fasta_parts))
    from chai_lab.chai1 import run_inference

    candidates = run_inference(
        fasta_file=fasta, output_dir=result_path.parent, use_esm_embeddings=True,
        use_msa_server=False, use_templates_server=False,
        num_trunk_recycles=recycles, num_diffn_timesteps=timesteps,
        num_diffn_samples=num_samples, seed=seed, device="cuda:0", low_memory=True,
    )
    pae_all = candidates.pae.detach().cpu().numpy()
    plddt_all = candidates.plddt.detach().cpu().numpy()
    np.savez_compressed(result_path.parent / "chai_tensors.npz", pae=pae_all, plddt=plddt_all)
    sample_rows: list[dict[str, Any]] = []
    for index, (cif, ranking) in enumerate(zip(candidates.cif_paths, candidates.ranking_data, strict=True)):
        pred_t_seq, pred_t_ca, pred_h_seq, pred_h_ca, structure = candidate_chains(
            Path(cif), target_sequence, ref_h_seq, additional_target_sequence
        )
        target_context_length = len(target_sequence) + (len(additional_target_sequence) if additional_target_sequence else 0)
        legacy = score_candidate(ref_t_seq, ref_t_ca, ref_h_seq, ref_h_ca, pred_t_seq, pred_t_ca, pred_h_seq, pred_h_ca, cdrs, pae_all[index], plddt_all[index], target_context_length)
        legacy["tag"] = tag
        sample_rows.append({
            "model_index": index, "cif": str(cif), "legacy": legacy,
            "aggregate_score": ranking_value(ranking, "aggregate_score"),
            "ptm": ranking_value(ranking, "ptm"), "iptm": ranking_value(ranking, "iptm"),
            "has_inter_chain_clashes": bool(getattr(ranking, "has_inter_chain_clashes", False)),
        })
    chosen = min(sample_rows, key=lambda row: (
        row["has_inter_chain_clashes"], finite_or_inf(row["legacy"]["target_aligned_cdr_rmsd"]),
        finite_or_inf(row["legacy"]["interaction_pae"]), -finite_or_inf(row["iptm"]),
    ))
    _, _, _, _, chosen_structure = candidate_chains(
        Path(chosen["cif"]), target_sequence, ref_h_seq, additional_target_sequence
    )
    save_selected_pdb(chosen_structure, pdb_out / f"{tag}.pdb")
    result = {"tag": tag, "selected_model_index": chosen["model_index"], "legacy": chosen["legacy"], "samples": sample_rows}
    result_path.write_text(json.dumps(result, indent=2, allow_nan=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-pdb-dir", type=Path, required=True)
    parser.add_argument("--full-target-fasta", type=Path, required=True)
    parser.add_argument("--additional-target-fasta", type=Path, help="Optional second target chain, written to Chai as entity L")
    parser.add_argument("--raw-output-dir", type=Path, required=True)
    parser.add_argument("--pdb-output-dir", type=Path, required=True)
    parser.add_argument("--scorefile", type=Path, required=True)
    parser.add_argument("--details", type=Path, required=True)
    parser.add_argument("--target-chain", default="T")
    parser.add_argument("--vhh-chain", default="H")
    parser.add_argument("--additional-chain", default="L")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--recycles", type=int, default=3)
    parser.add_argument("--timesteps", type=int, default=200)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    target_sequence = read_fasta(args.full_target_fasta)
    additional_target_sequence = read_fasta(args.additional_target_fasta) if args.additional_target_fasta else None
    pdbs = sorted(args.input_pdb_dir.glob("*.pdb"))
    if not pdbs:
        raise SystemExit(f"No PDBs found in {args.input_pdb_dir}")
    args.raw_output_dir.mkdir(parents=True, exist_ok=True)
    args.pdb_output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for number, pdb in enumerate(pdbs, start=1):
        print(f"[{number}/{len(pdbs)}] Chai: {pdb.stem}", flush=True)
        result = one_design(pdb, target_sequence, additional_target_sequence, args.raw_output_dir, args.pdb_output_dir, args.num_samples, args.recycles, args.timesteps, args.seed, args.resume, args.target_chain, args.vhh_chain)
        results.append(result)
    with args.scorefile.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LEGACY_COLUMNS, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(result["legacy"] for result in results)
    with args.details.open("w", newline="") as handle:
        columns = ["tag", "selected_model_index", "model_index", "aggregate_score", "ptm", "iptm", "has_inter_chain_clashes", "interaction_pae", "target_aligned_cdr_rmsd"]
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        for result in results:
            for sample in result["samples"]:
                writer.writerow({"tag": result["tag"], "selected_model_index": result["selected_model_index"], "model_index": sample["model_index"], "aggregate_score": sample["aggregate_score"], "ptm": sample["ptm"], "iptm": sample["iptm"], "has_inter_chain_clashes": sample["has_inter_chain_clashes"], **{k: sample["legacy"][k] for k in ("interaction_pae", "target_aligned_cdr_rmsd")}})
    print(f"Wrote {args.scorefile} and {args.details}")


if __name__ == "__main__":
    main()
