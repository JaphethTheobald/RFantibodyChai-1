#!/usr/bin/env python3
"""Copy AlphaFold Server job JSONs and add a sequence as a new protein chain.

Edit INPUT_DIRECTORY (and, if needed, TARGET_PROTEIN_CHAIN_INDEX) below, then run:
    python3 add_sequence_to_afserver_jsons.py

Original JSON files are never changed. Each processed JSON is written to
OUTPUT_DIRECTORY with the same filename. If a job was made by the older
append-to-the-target version of this script, its terminal copy of the sequence
is removed before the new standalone chain is inserted.
"""

from __future__ import annotations

import json
from pathlib import Path


# ---------------------------- EDIT THESE VALUES ----------------------------
INPUT_DIRECTORY = Path("/home/schrogpu/Documents/RFantibody_outputs/Best/f9_8OL9/8ol9_hydrophobic")
OUTPUT_DIRECTORY = INPUT_DIRECTORY.with_name(INPUT_DIRECTORY.name + "_with_added_chain")

# The new protein chain is inserted immediately after this existing protein
# chain. In the supplied example, index 1 is the second sequence (the FIXa
# target), while index 0 is the VHH.
TARGET_PROTEIN_CHAIN_INDEX = 1

SEQUENCE_TO_APPEND = "TCNIKNGRCEQFCKNSVVCSCTEGYRLAENQKSCEPAVPFPCGRVSVSQTS"

# False processes only JSONs directly in INPUT_DIRECTORY.  Set True to include
# JSONs inside its subdirectories while preserving their folder layout.
RECURSIVE = True
# ---------------------------------------------------------------------------


def protein_chain_entries(job: dict) -> list[tuple[int, dict]]:
    """Return (entity-list index, proteinChain) pairs in AF Server order."""
    entries = []
    for index, entity in enumerate(job.get("sequences", [])):
        chain = entity.get("proteinChain")
        if isinstance(chain, dict):
            entries.append((index, chain))
    return entries


def update_job(job: dict, source: Path) -> bool:
    chains = protein_chain_entries(job)
    if len(chains) <= TARGET_PROTEIN_CHAIN_INDEX:
        raise ValueError(
            f"{source}: contains {len(chains)} protein chains; cannot select "
            f"index {TARGET_PROTEIN_CHAIN_INDEX}."
        )

    target_entity_index, target_chain = chains[TARGET_PROTEIN_CHAIN_INDEX]
    sequence = target_chain.get("sequence")
    if not isinstance(sequence, str):
        raise ValueError(f"{source}: selected protein chain has no string sequence.")

    if any(chain.get("sequence") == SEQUENCE_TO_APPEND for _, chain in chains):
        return False

    # Repair a JSON made by the previous version of this script before adding
    # the standalone chain.
    if sequence.endswith(SEQUENCE_TO_APPEND):
        target_chain["sequence"] = sequence[: -len(SEQUENCE_TO_APPEND)]

    job["sequences"].insert(
        target_entity_index + 1,
        {
            "proteinChain": {
                "sequence": SEQUENCE_TO_APPEND,
                "count": 1,
                "useStructureTemplate": False,
            }
        },
    )
    return True


def main() -> None:
    if not INPUT_DIRECTORY.is_dir():
        raise SystemExit(f"INPUT_DIRECTORY does not exist or is not a directory: {INPUT_DIRECTORY}")

    paths = sorted(INPUT_DIRECTORY.rglob("*.json") if RECURSIVE else INPUT_DIRECTORY.glob("*.json"))
    if not paths:
        raise SystemExit(f"No .json files found in: {INPUT_DIRECTORY}")

    written = added = already_present = 0
    for source in paths:
        try:
            data = json.loads(source.read_text())
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid JSON in {source}: {exc}") from exc

        # AF Server imports a list of jobs; accept one-job JSONs too.
        jobs = data if isinstance(data, list) else [data]
        if not all(isinstance(job, dict) for job in jobs):
            raise SystemExit(f"{source}: expected a job dictionary or a list of job dictionaries.")

        changes = sum(update_job(job, source) for job in jobs)
        destination = OUTPUT_DIRECTORY / source.relative_to(INPUT_DIRECTORY)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(data, indent=2) + "\n")
        written += 1
        added += changes
        already_present += len(jobs) - changes

    print(f"Wrote {written} copied JSON file(s) to: {OUTPUT_DIRECTORY}")
    print(f"Added a separate protein chain to {added} job(s); skipped {already_present} already containing it.")


if __name__ == "__main__":
    main()
