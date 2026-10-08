#!/usr/bin/env python3
"""
Replace the target protein chain in every AF Server job with two protein chains,
each carrying its own custom mmCIF template and mapping.

Edit the CONFIGURATION block, then run:
    python3 make_afserver_two_chain_templated_jobs.py
"""

import json
from pathlib import Path


# ========================= EDIT THESE VALUES =========================
INPUT_JSON = Path("/home/schrogpu/Documents/RFantibody_outputs/12bn_campaign2/08_filtered_PDBs/alphafold_jobs.json")
OUTPUT_JSON = INPUT_JSON.with_name(
    INPUT_JSON.stem + "_two_chain_templated.json"
)


CHAIN_A_TEMPLATE_CIF = Path("/home/schrogpu/Documents/RFantibody_outputs/12bn_campaign2/08_filtered_PDBs/12bn_clean_chain_A_template.cif")
CHAIN_A_MAPPING_JSON = Path("/home/schrogpu/Documents/RFantibody_outputs/12bn_campaign2/08_filtered_PDBs/12bn_clean_chain_A_mapping.json")
CHAIN_A_SEQUENCE = "ECVTQLLKDTCFEGGDITTVFTPSAKYCQVVCTYHPRCLLFTFTAESPSEDPTRWFTCVLKDSVTETLPRVNRTAAISGYSFKQCSHQISACNKDIYVDLDMKGINYNSSVAKSAQECQERCTDDVHCHFFTYATRQFPSLEHRNICLLKHTQTGTPTRITKLDKVVSGFSLKSCALSNLACIRDIFPNTVFADSNIDSVMAPDAFVCGRICTHHPGCLFFTFFSQEWPKESQRNLCLLKTSESGLPSTRIKKSKALSGFSLQSCRHSIPVFCHSSFYHDTDFLGEELDIVAAKSHEACQKLCTNAVRCQFFTYTPAQASCNEGKGKCYLKLSSNGSPTKILHGRGGISGYTLRLCKMDNECTTKIKPR"


CHAIN_B_TEMPLATE_CIF = Path("/home/schrogpu/Documents/RFantibody_outputs/12bn_campaign2/08_filtered_PDBs/12bn_clean_chain_B_template.cif")
CHAIN_B_MAPPING_JSON = Path("/home/schrogpu/Documents/RFantibody_outputs/12bn_campaign2/08_filtered_PDBs/12bn_clean_chain_B_mapping.json")
CHAIN_B_SEQUENCE = "IVGGTASVRGEWPWQVTLHTTSPTQRHLCGGSIIGNQWILTAAHCFYGVESPKILRVYSGILNQSEIKEDTSFFGVQEIIIHDQYKMAESGYDIALLKLETTVNYTDSQRPICLPSKGDRNVIYTDCWVTGWGYRKLRDKIQNTLQKAKIPLVTNEECQKRYRGHKITHKMICAGYREGGKDACKGDAGGPLSCKHNEVWHLVGITSWGEGCAQRERPGVYTNVVEYVDWILEKTQ"

# Usually: protein chain 0 = VHH, protein chain 1 = target / former T chain.
TARGET_PROTEIN_CHAIN_INDEX = 1
# =====================================================================


def read_json(path):
    with open(path) as handle:
        return json.load(handle)


def clean_sequence(sequence, label):
    sequence = "".join(sequence.split()).upper()
    valid = set("ACDEFGHIKLMNPQRSTVWY")
    if not sequence or "PASTE_" in sequence or set(sequence) - valid:
        raise ValueError(f"Set {label} to a valid standard-amino-acid sequence.")
    return sequence


def load_template(cif_path, mapping_path, query_sequence, label):
    mmcif = cif_path.read_text()
    mapping_file = read_json(mapping_path)

    if not isinstance(mapping_file, list) or len(mapping_file) != 1:
        raise ValueError(f"{label} mapping must be a one-item JSON list.")

    mapping = mapping_file[0]
    query_indices = mapping["queryIndices"]
    template_indices = mapping["templateIndices"]

    if len(query_indices) != len(template_indices):
        raise ValueError(f"{label} mapping index lists differ in length.")

    if max(query_indices) >= len(query_sequence):
        raise ValueError(
            f"{label} mapping needs query position {max(query_indices)}, "
            f"but its supplied sequence is only {len(query_sequence)} residues. "
            "Use the exact sequence for which that mapping was generated."
        )

    # This is the AF Server JSON template schema: raw mmCIF text, not a filepath.
    return {
        "mmcif": mmcif,
        "queryIndices": query_indices,
        "templateIndices": template_indices,
    }


def protein_entity(sequence, template):
    return {
        "proteinChain": {
            "sequence": sequence,
            "count": 1,
            "templates": [template],
        }
    }


def replace_target_with_a_and_b(job, chain_a, chain_b):
    sequences = job["sequences"]

    protein_entity_positions = [
        i for i, entity in enumerate(sequences)
        if "proteinChain" in entity
    ]

    if len(protein_entity_positions) <= TARGET_PROTEIN_CHAIN_INDEX:
        raise ValueError(
            f"Job {job.get('name', '<unnamed>')} does not have protein-chain "
            f"index {TARGET_PROTEIN_CHAIN_INDEX}."
        )

    target_position = protein_entity_positions[TARGET_PROTEIN_CHAIN_INDEX]

    # Removes the former T/target entity and inserts A then B in its place.
    sequences[target_position:target_position + 1] = [chain_a, chain_b]


def main():
    chain_a_sequence = clean_sequence(CHAIN_A_SEQUENCE, "CHAIN_A_SEQUENCE")
    chain_b_sequence = clean_sequence(CHAIN_B_SEQUENCE, "CHAIN_B_SEQUENCE")

    template_a = load_template(
        CHAIN_A_TEMPLATE_CIF,
        CHAIN_A_MAPPING_JSON,
        chain_a_sequence,
        "Chain A",
    )
    template_b = load_template(
        CHAIN_B_TEMPLATE_CIF,
        CHAIN_B_MAPPING_JSON,
        chain_b_sequence,
        "Chain B",
    )

    jobs = read_json(INPUT_JSON)
    if not isinstance(jobs, list):
        raise ValueError("AF Server input JSON must be a list of jobs.")

    new_chain_a = protein_entity(chain_a_sequence, template_a)
    new_chain_b = protein_entity(chain_b_sequence, template_b)

    for job in jobs:
        replace_target_with_a_and_b(
            job,
            new_chain_a,
            new_chain_b,
        )

    OUTPUT_JSON.write_text(json.dumps(jobs, indent=2) + "\n")

    print(f"Wrote {len(jobs)} jobs to: {OUTPUT_JSON}")
    print(f"Chain A: {len(chain_a_sequence)} residues")
    print(f"Chain B: {len(chain_b_sequence)} residues")


if __name__ == "__main__":
    main()