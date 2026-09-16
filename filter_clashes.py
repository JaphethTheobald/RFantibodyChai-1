#!/usr/bin/env python3

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def read_chain_heavy_atoms(pdb_file, chain):
    atoms = []

    with open(pdb_file) as f:
        for line in f:

            if not line.startswith("ATOM"):
                continue

            if line[21:22].strip() != chain:
                continue

            atom_name = line[12:16].strip()

            # Ignore hydrogens if present
            element = line[76:78].strip().upper()

            # Robust hydrogen/deuterium detection.
            # PDB atom names can be H, HB2, 2HB, 1HE2, etc.
            name_no_digits = atom_name.lstrip("0123456789").upper()

            if element in ("H", "D"):
                continue

            if name_no_digits.startswith(("H", "D")):
                continue

            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue

            atoms.append(
                {
                    "coord": np.array([x, y, z]),
                    "atom": atom_name,
                    "resname": line[17:20].strip(),
                    "resi": line[22:26].strip(),
                    "chain": chain,
                }
            )

    return atoms


def analyze_clashes(pdb_file, binder_chain, target_chain):

    binder = read_chain_heavy_atoms(
        pdb_file,
        binder_chain,
    )

    target = read_chain_heavy_atoms(
        pdb_file,
        target_chain,
    )

    if not binder:
        raise RuntimeError(
            f"No heavy atoms found for binder chain {binder_chain}"
        )

    if not target:
        raise RuntimeError(
            f"No heavy atoms found for target chain {target_chain}"
        )

    bxyz = np.array([a["coord"] for a in binder])
    txyz = np.array([a["coord"] for a in target])

    btree = cKDTree(bxyz)
    ttree = cKDTree(txyz)

    # Get all cross-chain pairs within 2.5 A
    neighbors = btree.query_ball_tree(ttree, r=2.5)

    contacts = []

    for bi, target_indices in enumerate(neighbors):

        for ti in target_indices:

            distance = np.linalg.norm(
                bxyz[bi] - txyz[ti]
            )

            contacts.append(
                (
                    distance,
                    binder[bi],
                    target[ti],
                )
            )

    contacts.sort(
        key=lambda x: x[0]
    )

    if contacts:
        min_distance = contacts[0][0]
    else:
        min_distance = float("inf")

    n_lt_18 = sum(
        d < 1.8
        for d, _, _ in contacts
    )

    n_lt_20 = sum(
        d < 2.0
        for d, _, _ in contacts
    )

    n_lt_22 = sum(
        d < 2.2
        for d, _, _ in contacts
    )

    n_lt_25 = len(contacts)

    return {
        "min_distance": min_distance,
        "n_lt_18": n_lt_18,
        "n_lt_20": n_lt_20,
        "n_lt_22": n_lt_22,
        "n_lt_25": n_lt_25,
        "contacts": contacts,
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "input_dir",
        help="Folder containing RF2 PDB files",
    )

    parser.add_argument(
        "output_dir",
        help="Folder to copy passing PDB files into",
    )

    parser.add_argument(
        "--binder-chain",
        default="H",
    )

    parser.add_argument(
        "--target-chain",
        default="T",
    )

    parser.add_argument(
        "--reject-below",
        type=float,
        default=1.8,
        help=(
            "Reject if any binder-target heavy atom pair "
            "is closer than this distance in Angstroms"
        ),
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pdb_files = sorted(
        input_dir.glob("*.pdb")
    )

    if not pdb_files:
        raise SystemExit(
            f"No PDB files found in {input_dir}"
        )

    report = output_dir / "clash_report.tsv"

    results = []

    for pdb in pdb_files:

        try:

            r = analyze_clashes(
                pdb,
                args.binder_chain,
                args.target_chain,
            )

        except Exception as e:

            print(
                f"ERROR {pdb.name}: {e}"
            )

            continue

        passed = (
            r["min_distance"]
            >= args.reject_below
        )

        results.append(
            (
                pdb.name,
                r["min_distance"],
                r["n_lt_18"],
                r["n_lt_20"],
                r["n_lt_22"],
                r["n_lt_25"],
                passed,
            )
        )

        status = (
            "PASS"
            if passed
            else "REJECT"
        )

        print(
            f"{status:6s} "
            f"{pdb.name:55s} "
            f"min={r['min_distance']:.2f} A  "
            f"<1.8={r['n_lt_18']:2d}  "
            f"<2.0={r['n_lt_20']:2d}  "
            f"<2.2={r['n_lt_22']:2d}"
        )

        if passed:
            shutil.copy2(
                pdb,
                output_dir / pdb.name,
            )

        else:

            print("       Worst clashes:")

            for distance, b, t in r["contacts"][:5]:

                print(
                    f"       "
                    f"H:{b['resname']}{b['resi']}:{b['atom']} "
                    f"<-> "
                    f"T:{t['resname']}{t['resi']}:{t['atom']} "
                    f"{distance:.2f} A"
                )

    with open(report, "w", newline="") as f:

        writer = csv.writer(
            f,
            delimiter="\t",
        )

        writer.writerow(
            [
                "pdb",
                "min_interchain_distance",
                "contacts_lt_1.8",
                "contacts_lt_2.0",
                "contacts_lt_2.2",
                "contacts_lt_2.5",
                "pass",
            ]
        )

        writer.writerows(results)

    passed_count = sum(
        x[-1]
        for x in results
    )

    print()
    print(
        f"Passed {passed_count}/{len(results)} structures"
    )

    print(
        f"Report: {report}"
    )

    print(
        f"Passing structures: {output_dir}"
    )


if __name__ == "__main__":
    main()