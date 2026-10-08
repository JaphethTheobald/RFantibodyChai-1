#!/usr/bin/env python3
"""Copy AF Server job JSONs, splitting an appended sequence into a new chain.

Edit INPUT_DIRECTORY below, then run:
    python3 split_appended_sequence_into_afserver_chain.py

For each job containing SEQUENCE_TO_SPLIT inside one protein sequence:
  - the original chain becomes everything BEFORE that sequence;
  - a new protein chain containing exactly SEQUENCE_TO_SPLIT is inserted after it;
  - the matched sequence and anything after it are discarded from the original.

Original JSONs are never modified.
"""

from __future__ import annotations

import json
from pathlib import Path


# ---------------------------- EDIT THESE VALUES ----------------------------
INPUT_DIRECTORY = Path("/home/schrogpu/Documents/RFantibody_outputs/Best/f7_1jbu/A183mimic")
OUTPUT_DIRECTORY = INPUT_DIRECTORY.with_name(INPUT_DIRECTORY.name + "_split_tail_chain")

SEQUENCE_TO_SPLIT = "ICVNENGGCEQYCSDHTGTKRSCRCHEGYSLLADGVSCTPTVEYPCGKIPILEK"

# False processes only JSONs directly in INPUT_DIRECTORY. Set True to include
# subfolders, preserving their directory structure in OUTPUT_DIRECTORY.
RECURSIVE = True
# ---------------------------------------------------------------------------


def split_in_job(job: dict, source: Path) -> bool:
    """Split the one protein-chain sequence containing SEQUENCE_TO_SPLIT."""
    entities = job.get("sequences")
    if not isinstance(entities, list):
        raise ValueError(f"{source}: job has no 'sequences' list.")

    matches: list[tuple[int, dict, str]] = []
    already_separate = False
    for index, entity in enumerate(entities):
        chain = entity.get("proteinChain") if isinstance(entity, dict) else None
        if not isinstance(chain, dict):
            continue
        sequence = chain.get("sequence")
        if not isinstance(sequence, str):
            raise ValueError(f"{source}: protein chain at index {index} has no string sequence.")
        if sequence == SEQUENCE_TO_SPLIT:
            already_separate = True
        elif SEQUENCE_TO_SPLIT in sequence:
            matches.append((index, chain, sequence))

    if not matches:
        # This also safely skips jobs that have already been split.
        return False
    if len(matches) != 1:
        raise ValueError(f"{source}: found the sequence embedded in {len(matches)} chains; refusing to guess.")

    index, original_chain, original_sequence = matches[0]
    prefix, _, _discarded_suffix = original_sequence.partition(SEQUENCE_TO_SPLIT)
    if not prefix:
        raise ValueError(f"{source}: sequence to split occurs at the start of its chain.")

    original_chain["sequence"] = prefix
    new_entity = {
        "proteinChain": {
            "sequence": SEQUENCE_TO_SPLIT,
            "count": 1,
            "useStructureTemplate": False,
        }
    }
    entities.insert(index + 1, new_entity)
    return True


def main() -> None:
    if not INPUT_DIRECTORY.is_dir():
        raise SystemExit(f"INPUT_DIRECTORY does not exist or is not a directory: {INPUT_DIRECTORY}")

    sources = sorted(INPUT_DIRECTORY.rglob("*.json") if RECURSIVE else INPUT_DIRECTORY.glob("*.json"))
    if not sources:
        raise SystemExit(f"No .json files found in: {INPUT_DIRECTORY}")

    files_written = jobs_split = jobs_skipped = 0
    for source in sources:
        try:
            data = json.loads(source.read_text())
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid JSON in {source}: {exc}") from exc

        # AF Server accepts a list of jobs; accept a single job dictionary too.
        jobs = data if isinstance(data, list) else [data]
        if not all(isinstance(job, dict) for job in jobs):
            raise SystemExit(f"{source}: expected a job dictionary or list of job dictionaries.")

        for job in jobs:
            if split_in_job(job, source):
                jobs_split += 1
            else:
                jobs_skipped += 1

        destination = OUTPUT_DIRECTORY / source.relative_to(INPUT_DIRECTORY)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(data, indent=2) + "\n")
        files_written += 1

    print(f"Wrote {files_written} copied JSON file(s) to: {OUTPUT_DIRECTORY}")
    print(f"Split {jobs_split} job(s); skipped {jobs_skipped} with no embedded match.")


if __name__ == "__main__":
    main()

