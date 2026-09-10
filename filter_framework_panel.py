#!/usr/bin/env python3

from pathlib import Path
import subprocess
import shutil

RFAB = Path.home() / "RFantibody"

BASE = Path.home() / "Documents/RFantibody_outputs/framework_panel"
RFDIR = BASE / "04_rfdiffusion"
OUTDIR = BASE / "04b_filtered"

OUTDIR.mkdir(parents=True, exist_ok=True)

MINDIST_MAX = 6.0
AVERAGEMIN_MAX = 10.0

FRAMEWORKS = [
    "5hgg",
    "5hvf",
    "5hvg",
    "7a6o",
]


def run(cmd):
    print("$", " ".join(str(x) for x in cmd))
    subprocess.run(
        [str(x) for x in cmd],
        cwd=RFAB,
        check=True,
    )


for fw in FRAMEWORKS:

    sc = RFDIR / f"{fw}_1jbu_rfdiffusion.sc"
    qv = RFDIR / f"{fw}_1jbu_rfdiffusion.qv"

    filtered_qv = OUTDIR / f"{fw}_1jbu_rfdiffusion_filtered.qv"
    summary = OUTDIR / f"{fw}_filter_summary.tsv"

    print()
    print("=" * 60)
    print(f"Framework: {fw}")
    print("=" * 60)

    if not sc.exists():
        print(f"Missing scorefile: {sc}")
        continue

    if not qv.exists():
        print(f"Missing Quiver: {qv}")
        continue

    passing_tags = []
    rows = []

    with open(sc) as f:
        header = f.readline().strip().split()

        try:
            mindist_i = header.index("mindist")
            avg_i = header.index("averagemin")
            tag_i = header.index("tag")
        except ValueError:
            raise RuntimeError(
                f"Could not find expected columns in {sc}"
            )

        for line in f:
            if not line.strip():
                continue

            cols = line.strip().split()

            if len(cols) <= max(mindist_i, avg_i, tag_i):
                continue

            try:
                mindist = float(cols[mindist_i])
                averagemin = float(cols[avg_i])
            except ValueError:
                continue

            tag = cols[tag_i]

            passed = (
                mindist <= MINDIST_MAX
                and averagemin <= AVERAGEMIN_MAX
            )

            rows.append(
                (tag, mindist, averagemin, passed)
            )

            if passed:
                passing_tags.append(tag)

    with open(summary, "w") as f:
        f.write("tag\tmindist\taveragemin\tpass\n")

        for tag, mindist, averagemin, passed in rows:
            f.write(
                f"{tag}\t{mindist:.2f}\t{averagemin:.2f}\t{passed}\n"
            )

    print(f"Passed {len(passing_tags)}/{len(rows)} designs")

    for tag in passing_tags:
        print(" PASS", tag)

    if not passing_tags:
        print("No passing designs. Skipping filtered Quiver.")
        continue

    if filtered_qv.exists():
        filtered_qv.unlink()

    tempdir = OUTDIR / f"{fw}_temp_pdbs"

    if tempdir.exists():
        shutil.rmtree(tempdir)

    tempdir.mkdir()

    run([
        "uv", "run", "qvextract",
        qv,
        "-o", tempdir,
    ])

    extracted = list(tempdir.glob("*.pdb"))

    if not extracted:
        print("ERROR: qvextract produced no PDBs.")
        continue

    for pdb in extracted:
        if pdb.stem not in passing_tags:
            pdb.unlink()

    remaining = sorted(tempdir.glob("*.pdb"))

    if not remaining:
        print(
            "ERROR: passing tags did not match extracted PDB filenames."
        )
        continue

    print(
        f"Building filtered Quiver from {len(remaining)} PDBs"
    )

    cmd = [
        "uv", "run", "qvfrompdbs",
        *remaining,
    ]

    print(
        "$",
        " ".join(str(x) for x in cmd),
        ">",
        filtered_qv
    )

    with open(filtered_qv, "w") as out:
        subprocess.run(
            [str(x) for x in cmd],
            cwd=RFAB,
            check=True,
            stdout=out,
        )

    print("Filtered Quiver created:")
    print(filtered_qv)

    print("Contents:")
    subprocess.run(
        [
            "uv", "run", "qvls",
            str(filtered_qv),
        ],
        cwd=RFAB,
        check=True,
    )

print()
print("Done.")
print("Filtered outputs:")
print(OUTDIR)
