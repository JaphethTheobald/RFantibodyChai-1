#!/usr/bin/env python3

# cd "$HOME/RFantibody"

# ./renumber_1jbu.py \
#     "$HOME/RFantibody/1jbu_immense_trunc.pdb" \
#     "$HOME/RFantibody/1jbu_immense_trunc_renumbered.pdb" \
#     "$HOME/RFantibody/1jbu_immense_trunc_residue_mapping.tsv"

from pathlib import Path
import argparse


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Renumber PDB residues to unique integer residue numbers, "
            "removing insertion codes while preserving residue order."
        )
    )

    parser.add_argument("input_pdb")
    parser.add_argument("output_pdb")
    parser.add_argument("mapping_tsv")

    args = parser.parse_args()

    src = Path(args.input_pdb)
    dst = Path(args.output_pdb)
    mapping_path = Path(args.mapping_tsv)

    if not src.exists():
        raise SystemExit(f"Input PDB not found: {src}")

    # key:
    # (chain, original_resseq_string, insertion_code)
    #
    # value:
    # new integer residue number
    mapping = {}

    metadata = {}

    # Track previous new residue number separately for each chain.
    previous_new = {}

    output_lines = []

    with src.open() as f:
        for line in f:

            if not line.startswith(("ATOM  ", "HETATM")):
                output_lines.append(line)
                continue

            if len(line) < 27:
                output_lines.append(line)
                continue

            chain = line[21:22]

            old_resseq_text = line[22:26]
            old_resseq_clean = old_resseq_text.strip()
            icode = line[26:27].strip()

            try:
                old_resseq = int(old_resseq_clean)
            except ValueError:
                raise RuntimeError(
                    f"Could not parse residue number from line:\n{line}"
                )

            resname = line[17:20].strip()

            key = (
                chain,
                old_resseq,
                icode,
            )

            if key not in mapping:

                if chain not in previous_new:
                    # Preserve the original number for the first residue
                    # encountered in each chain.
                    new_resseq = old_resseq

                else:
                    # Preserve the original number whenever possible.
                    # If it would collide with a previous residue because
                    # of insertion codes, advance to the next available
                    # integer.
                    new_resseq = max(
                        old_resseq,
                        previous_new[chain] + 1,
                    )

                if new_resseq > 9999:
                    raise RuntimeError(
                        f"PDB residue number exceeded 9999 in chain {chain}"
                    )

                mapping[key] = new_resseq
                previous_new[chain] = new_resseq

                metadata[key] = {
                    "resname": resname,
                }

            new_resseq = mapping[key]

            # Replace:
            # columns 23-26 = residue sequence number
            # column 27     = insertion code
            #
            # Clear the insertion code because the new integer numbering
            # now uniquely identifies the residue.
            newline = (
                line[:22]
                + f"{new_resseq:4d}"
                + " "
                + line[27:]
            )

            output_lines.append(newline)

    dst.parent.mkdir(parents=True, exist_ok=True)
    mapping_path.parent.mkdir(parents=True, exist_ok=True)

    with dst.open("w") as f:
        f.writelines(output_lines)

    with mapping_path.open("w") as f:

        f.write(
            "chain\t"
            "old_resnum\t"
            "insertion_code\t"
            "old_id\t"
            "resname\t"
            "new_resnum\t"
            "new_id\n"
        )

        for key, new_resseq in mapping.items():

            chain, old_resseq, icode = key
            resname = metadata[key]["resname"]

            old_id = f"{chain}{old_resseq}{icode}"
            new_id = f"{chain}{new_resseq}"

            f.write(
                f"{chain}\t"
                f"{old_resseq}\t"
                f"{icode}\t"
                f"{old_id}\t"
                f"{resname}\t"
                f"{new_resseq}\t"
                f"{new_id}\n"
            )

    print()
    print("Created:")
    print(f"  PDB:     {dst}")
    print(f"  mapping: {mapping_path}")
    print()

    # Highlight insertion-coded residues.
    inserted = [
        (key, new)
        for key, new in mapping.items()
        if key[2]
    ]

    if inserted:
        print("Insertion-code mappings:")
        print()

        for (chain, old_resseq, icode), new in inserted:
            resname = metadata[(chain, old_resseq, icode)]["resname"]

            print(
                f"  {chain}{old_resseq}{icode:<2} "
                f"{resname:3s}  ->  {chain}{new}"
            )

    print()


if __name__ == "__main__":
    main()