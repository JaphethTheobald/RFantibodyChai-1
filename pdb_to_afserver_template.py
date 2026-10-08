#!/usr/bin/env python3
"""Create an AlphaFold Server custom-template mmCIF and mapping JSON from a PDB.

Examples
--------
Interactive mode:
    python3 pdb_to_afserver_template.py

Command-line mode:
    python3 pdb_to_afserver_template.py ./1jbu_noX_rfantibody.pdb \
        --chain H --query-fasta KVCPKGECPWQVLLLVNGAQLCGGTLINTIWVVSAAHCFDKIKNWRNLIAVLGEHDHDGDEQSRRVAQVIIPSTYVPGTTNHDIALLRLHQPVVLTDHVVPLCLPERTFSERTLAFVRFSLVSGWTALELMVLNVPRLMTQDCLQQSRKVGDSPNITEYMFCAGYSDGSKDSCKGDSGGPHATHYRGTWYLTGIVSWGQGCATVGHFGVYTRVSQYIEWLQKLMRSEPRPGVLLRAPFP --out-dir af_template

The output mmCIF contains exactly one cleaned protein chain.  The mapping JSON
uses the AlphaFold Server format: 0-based queryIndices and templateIndices.
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path


AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M",  # Selenomethionine is converted to MET in the output.
}
ONE_TO_3 = {value: key for key, value in AA3_TO_1.items() if key != "MSE"}


def clean_sequence(text: str) -> str:
    """Read either a bare sequence or a FASTA string/file body."""
    sequence = "".join(
        line.strip() for line in text.splitlines() if not line.lstrip().startswith(">")
    ).replace(" ", "").upper()
    invalid = sorted(set(sequence) - set("ACDEFGHIKLMNPQRSTVWY"))
    if not sequence or invalid:
        raise ValueError(f"Query sequence must contain standard amino acids only; invalid: {invalid}")
    return sequence


def parse_chain(pdb_path: Path, chain_id: str) -> list[dict]:
    """Parse one protein chain and retain one highest-occupancy altloc per atom."""
    residues: OrderedDict[tuple[str, str, str], dict] = OrderedDict()
    with pdb_path.open() as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")) or len(line) < 54:
                continue
            if line[21] != chain_id:
                continue

            resname = line[17:20].strip().upper()
            if resname not in AA3_TO_1:
                continue  # remove waters, glycans, ligands, and non-protein HETATMs

            resid, insertion = line[22:26].strip(), line[26].strip()
            key = (resid, insertion, resname)
            residue = residues.setdefault(
                key,
                {"resname": resname, "resid": resid, "insertion": insertion, "atoms": {}},
            )
            atom_name = line[12:16].strip()
            altloc = line[16].strip()
            try:
                occupancy = float(line[54:60])
            except ValueError:
                occupancy = 1.0
            existing = residue["atoms"].get(atom_name)
            # Blank altloc atoms are shared; otherwise retain the highest occupancy.
            prefer = existing is None or (existing["altloc"] != "" and occupancy > existing["occupancy"])
            if prefer:
                residue["atoms"][atom_name] = {
                    "altloc": altloc,
                    "occupancy": occupancy,
                    "element": line[76:78].strip() or atom_name[0],
                    "x": float(line[30:38]),
                    "y": float(line[38:46]),
                    "z": float(line[46:54]),
                    "b": float(line[60:66]) if line[60:66].strip() else 0.0,
                }

    output = list(residues.values())
    if not output:
        raise ValueError(f"No standard protein residues found for chain {chain_id!r} in {pdb_path}")
    return output


def global_alignment(query: str, template: str) -> list[tuple[int, int]]:
    """Needleman-Wunsch global alignment; returns aligned query/template indices."""
    match, mismatch, gap = 2, -1, -3
    rows, cols = len(query) + 1, len(template) + 1
    scores = [[0] * cols for _ in range(rows)]
    moves = [[""] * cols for _ in range(rows)]
    for i in range(1, rows):
        scores[i][0], moves[i][0] = i * gap, "U"
    for j in range(1, cols):
        scores[0][j], moves[0][j] = j * gap, "L"
    for i in range(1, rows):
        for j in range(1, cols):
            choices = [
                (scores[i - 1][j - 1] + (match if query[i - 1] == template[j - 1] else mismatch), "D"),
                (scores[i - 1][j] + gap, "U"),
                (scores[i][j - 1] + gap, "L"),
            ]
            scores[i][j], moves[i][j] = max(choices, key=lambda item: item[0])

    mapping: list[tuple[int, int]] = []
    i, j = len(query), len(template)
    while i or j:
        move = moves[i][j]
        if move == "D":
            mapping.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif move == "U":
            i -= 1
        elif move == "L":
            j -= 1
        else:
            raise RuntimeError("Alignment traceback failed")
    return list(reversed(mapping))


def cif_token(value: str) -> str:
    """Quote CIF values only when a bare token would be invalid."""
    if not value or any(char.isspace() for char in value) or value in {".", "?"}:
        return "'" + value.replace("'", "''") + "'"
    return value


def write_mmcif(residues: list[dict], output_path: Path) -> str:
    """Write a minimal, standards-compliant one-chain protein mmCIF."""
    sequence = "".join(AA3_TO_1[residue["resname"]] for residue in residues)
    lines = [
        "data_custom_template",
        "#",
        "_entry.id custom_template",
        "#",
        "_entity.id 1",
        "_entity.type polymer",
        "_entity.pdbx_description 'Custom template from PDB'",
        "#",
        "_entity_poly.entity_id 1",
        "_entity_poly.type 'polypeptide(L)'",
        f"_entity_poly.pdbx_seq_one_letter_code_can {cif_token(sequence)}",
        f"_entity_poly.pdbx_seq_one_letter_code {cif_token(sequence)}",
        "#",
        "loop_",
        "_entity_poly_seq.entity_id",
        "_entity_poly_seq.num",
        "_entity_poly_seq.mon_id",
        "_entity_poly_seq.hetero",
    ]
    for index, residue in enumerate(residues, start=1):
        lines.append(f"1 {index} {ONE_TO_3[AA3_TO_1[residue['resname']]]} n")
    lines.extend([
        "#",
        "loop_",
        "_struct_asym.id",
        "_struct_asym.entity_id",
        "A 1",
        "#",
        "loop_",
        "_atom_site.group_PDB",
        "_atom_site.id",
        "_atom_site.type_symbol",
        "_atom_site.label_atom_id",
        "_atom_site.label_alt_id",
        "_atom_site.label_comp_id",
        "_atom_site.label_asym_id",
        "_atom_site.label_entity_id",
        "_atom_site.label_seq_id",
        "_atom_site.pdbx_PDB_ins_code",
        "_atom_site.Cartn_x",
        "_atom_site.Cartn_y",
        "_atom_site.Cartn_z",
        "_atom_site.occupancy",
        "_atom_site.B_iso_or_equiv",
        "_atom_site.auth_seq_id",
        "_atom_site.auth_comp_id",
        "_atom_site.auth_asym_id",
        "_atom_site.auth_atom_id",
        "_atom_site.pdbx_PDB_model_num",
    ])
    atom_id = 1
    for seq_id, residue in enumerate(residues, start=1):
        comp_id = ONE_TO_3[AA3_TO_1[residue["resname"]]]
        for atom_name, atom in residue["atoms"].items():
            # Convert MSE's selenium atom to standard MET sulfur in the template.
            if residue["resname"] == "MSE" and atom_name == "SE":
                atom_name, element = "SD", "S"
            else:
                element = atom["element"]
            lines.append(
                f"ATOM {atom_id} {element} {atom_name} . {comp_id} A 1 {seq_id} "
                f"{residue['insertion'] or '?'} {atom['x']:.3f} {atom['y']:.3f} {atom['z']:.3f} "
                f"{atom['occupancy']:.2f} {atom['b']:.2f} {residue['resid']} {comp_id} A {atom_name} 1"
            )
            atom_id += 1
    lines.append("#")
    output_path.write_text("\n".join(lines) + "\n")
    return sequence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdb", nargs="?", type=Path, help="Input PDB file")
    parser.add_argument("--chain", help="One-character PDB chain ID to use")
    query = parser.add_mutually_exclusive_group()
    query.add_argument("--query-sequence", help="Exact protein sequence entered in AlphaFold Server")
    query.add_argument("--query-fasta", type=Path, help="FASTA containing that exact AF Server sequence")
    parser.add_argument("--out-dir", type=Path, default=Path("afserver_custom_template"))
    args = parser.parse_args()

    pdb_path = args.pdb or Path(input("PDB file path: ").strip())
    if not pdb_path.is_file():
        raise SystemExit(f"PDB file not found: {pdb_path}")

    all_chains = sorted({line[21] for line in pdb_path.read_text().splitlines() if line.startswith(("ATOM  ", "HETATM"))})
    chain = args.chain or input(f"Template chain ID {all_chains}: ").strip()
    if len(chain) != 1 or chain not in all_chains:
        raise SystemExit(f"Choose one listed one-character PDB chain ID: {all_chains}")

    if args.query_fasta:
        query_sequence = clean_sequence(args.query_fasta.read_text())
    elif args.query_sequence:
        query_sequence = clean_sequence(args.query_sequence)
    else:
        query_sequence = clean_sequence(input("Exact AlphaFold Server sequence (paste sequence or FASTA): "))

    residues = parse_chain(pdb_path, chain)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{pdb_path.stem}_chain_{chain}"
    cif_path = args.out_dir / f"{stem}_template.cif"
    template_sequence = write_mmcif(residues, cif_path)
    mapping = global_alignment(query_sequence, template_sequence)
    if not mapping:
        raise SystemExit("No sequence alignment could be made.")

    identical = sum(query_sequence[i] == template_sequence[j] for i, j in mapping)
    identity = identical / len(mapping)
    if identity < 0.50:
        raise SystemExit(
            f"Alignment identity is only {identity:.1%}; check chain and query sequence before using this template."
        )

    mapping_path = args.out_dir / f"{stem}_mapping.json"
    mapping_path.write_text(json.dumps([{
        "name": cif_path.name,
        "queryIndices": [query_index for query_index, _ in mapping],
        "templateIndices": [template_index for _, template_index in mapping],
    }], indent=2) + "\n")

    print(f"Wrote: {cif_path}")
    print(f"Wrote: {mapping_path}")
    print(
        f"Mapped {len(mapping)} residues; identity across mapped positions: {identity:.1%}. "
        f"Query length {len(query_sequence)}, template-coordinate length {len(template_sequence)}."
    )


if __name__ == "__main__":
    main()
