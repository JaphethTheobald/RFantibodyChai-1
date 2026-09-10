from pathlib import Path
import csv
import math
import os
import shutil
import subprocess
import sys


# ============================================================
# USER SETTINGS
# ============================================================

RFAB = Path.home() / "RFantibody"
TARGET = Path.home() / "Downloads" / "1jbu_clean.pdb"
FRAMEWORK = RFAB / "scripts/examples/example_inputs/h-NbBCII10.pdb"

CAMPAIGN = Path.home() / "Documents/RFantibody_outputs/1jbu_campaign"

NUM_BACKBONES = 50
HOTSPOTS = "H237,H240,H242,H244"
DESIGN_LOOPS = "H1:7,H2:6,H3:8-16"

SEQS_PER_BACKBONE = 8

GPU = "0"

# Geometry filters
CLASH_CUTOFF = 2.0
MAX_CLASHES = 2

CONTACT_CUTOFF = 5.0
MIN_CONTACTING_VHH_RESIDUES = 4
MIN_CDR_CONTACTING_RESIDUES = 2
MAX_FRAMEWORK_FRACTION = 0.85

# Approximate VHH CDR ranges for this framework/output numbering.
CDR_RANGES = [
    (25, 40),   # H1
    (50, 65),   # H2
    (95, 115),  # H3
]


# ============================================================
# PATHS
# ============================================================

BACKBONE_DIR = CAMPAIGN / "01_backbones"
BACKBONE_QV = BACKBONE_DIR / "1jbu_backbones.qv"
BACKBONE_PDBS = BACKBONE_DIR / "pdbs"

FILTER_DIR = CAMPAIGN / "02_filtered"
FILTER_PDBS = FILTER_DIR / "pdbs"
FILTER_CSV = FILTER_DIR / "filter_scores.csv"

MPNN_DIR = CAMPAIGN / "03_mpnn"
MPNN_QV = MPNN_DIR / "1jbu_mpnn.qv"
MPNN_PDBS = MPNN_DIR / "pdbs"

RF2_DIR = CAMPAIGN / "04_rf2"
RF2_QV = RF2_DIR / "1jbu_rf2.qv"
RF2_PDBS = RF2_DIR / "pdbs"

LOG = CAMPAIGN / "campaign.log"


# ============================================================
# ENVIRONMENT
# ============================================================

ENV = os.environ.copy()

runtime_lib = (
    Path.home()
    / ".conda/envs/rfantibody-runtime/lib"
)

ENV["LD_LIBRARY_PATH"] = (
    str(runtime_lib)
    + ":"
    + ENV.get("LD_LIBRARY_PATH", "")
)

ENV["CUDA_VISIBLE_DEVICES"] = GPU
ENV["DGLBACKEND"] = "pytorch"

ENV["PATH"] = (
    str(Path.home() / ".local/bin")
    + ":"
    + ENV.get("PATH", "")
)


# ============================================================
# HELPERS
# ============================================================

def log(msg=""):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(msg + "\n")


def run(cmd, check=True, capture=False):
    log("$ " + " ".join(str(x) for x in cmd))

    result = subprocess.run(
        [str(x) for x in cmd],
        cwd=RFAB,
        env=ENV,
        text=True,
        capture_output=capture,
    )

    if capture:
        if result.stdout:
            log(result.stdout.rstrip())
        if result.stderr:
            log(result.stderr.rstrip())

    if check and result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}:\n"
            + " ".join(str(x) for x in cmd)
        )

    return result


def get_help(command):
    r = subprocess.run(
        ["uv", "run", command, "--help"],
        cwd=RFAB,
        env=ENV,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return r.stdout


def has_option(help_text, *names):
    return any(name in help_text for name in names)


def is_cdr(resid):
    return any(start <= resid <= end for start, end in CDR_RANGES)


def parse_atoms(pdb):
    atoms = []

    with open(pdb) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue

            atom_name = line[12:16].strip()
            element = line[76:78].strip().upper()

            if not element:
                element = atom_name[0].upper()

            if element == "H":
                continue

            chain = line[21].strip()
            resname = line[17:20].strip()

            try:
                resid = int(line[22:26].strip())
                xyz = (
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                )
            except ValueError:
                continue

            atoms.append({
                "chain": chain,
                "resname": resname,
                "resid": resid,
                "atom": atom_name,
                "xyz": xyz,
            })

    return atoms


def distance(a, b):
    return math.sqrt(
        (a[0] - b[0]) ** 2
        + (a[1] - b[1]) ** 2
        + (a[2] - b[2]) ** 2
    )


# ============================================================
# FILTER
# ============================================================

def filter_backbones():
    log("")
    log("============================================================")
    log("STEP 2: geometry/interface filtering")
    log("============================================================")

    FILTER_PDBS.mkdir(parents=True, exist_ok=True)

    for p in FILTER_PDBS.glob("*.pdb"):
        p.unlink()

    rows = []

    pdbs = sorted(BACKBONE_PDBS.glob("*.pdb"))

    if not pdbs:
        raise RuntimeError(
            f"No extracted PDBs found in {BACKBONE_PDBS}"
        )

    for pdb in pdbs:
        atoms = parse_atoms(pdb)

        # RFantibody output convention for this run:
        # H = VHH, T = target
        vhh = [a for a in atoms if a["chain"] == "H"]
        target = [a for a in atoms if a["chain"] == "T"]

        if not vhh or not target:
            log(
                f"{pdb.name}: ERROR - expected chains H and T not found"
            )
            continue

        clashes = 0
        min_dist = float("inf")

        contacting_vhh = set()
        contacting_target = set()
        contacting_cdr = set()
        contacting_framework = set()

        for a in vhh:
            for b in target:
                d = distance(a["xyz"], b["xyz"])

                min_dist = min(min_dist, d)

                if d < CLASH_CUTOFF:
                    clashes += 1

                if d < CONTACT_CUTOFF:
                    contacting_vhh.add(a["resid"])
                    contacting_target.add(b["resid"])

                    if is_cdr(a["resid"]):
                        contacting_cdr.add(a["resid"])
                    else:
                        contacting_framework.add(a["resid"])

        total_vhh_contact_res = len(contacting_vhh)

        if total_vhh_contact_res:
            framework_fraction = (
                len(contacting_framework)
                / total_vhh_contact_res
            )
        else:
            framework_fraction = 1.0

        passed = (
            clashes <= MAX_CLASHES
            and total_vhh_contact_res >= MIN_CONTACTING_VHH_RESIDUES
            and len(contacting_cdr) >= MIN_CDR_CONTACTING_RESIDUES
            and framework_fraction <= MAX_FRAMEWORK_FRACTION
        )

        status = "PASS" if passed else "FAIL"

        log(
            f"{pdb.name:28s} "
            f"{status:4s} "
            f"clash={clashes:3d} "
            f"VHHcontact={total_vhh_contact_res:2d} "
            f"CDR={len(contacting_cdr):2d} "
            f"FWfrac={framework_fraction:.2f} "
            f"min={min_dist:.2f}A"
        )

        if passed:
            shutil.copy2(pdb, FILTER_PDBS / pdb.name)

        rows.append({
            "design": pdb.name,
            "pass": passed,
            "clashes_lt_2A": clashes,
            "minimum_distance_A": round(min_dist, 3),
            "contacting_vhh_residues": total_vhh_contact_res,
            "contacting_target_residues": len(contacting_target),
            "contacting_cdr_residues": len(contacting_cdr),
            "contacting_framework_residues": len(contacting_framework),
            "framework_fraction": round(framework_fraction, 3),
        })

    if not rows:
        raise RuntimeError("No designs could be analyzed.")

    with open(FILTER_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    passed = [r for r in rows if r["pass"]]

    log("")
    log(f"Passed {len(passed)}/{len(rows)} backbone designs.")
    log(f"Filter table: {FILTER_CSV}")

    if not passed:
        raise RuntimeError(
            "No backbones passed. Relax the filter thresholds."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    for d in [
        CAMPAIGN,
        BACKBONE_DIR,
        BACKBONE_PDBS,
        FILTER_DIR,
        FILTER_PDBS,
        MPNN_DIR,
        MPNN_PDBS,
        RF2_DIR,
        RF2_PDBS,
    ]:
        d.mkdir(parents=True, exist_ok=True)

    LOG.touch(exist_ok=True)

    log("============================================================")
    log("RFantibody 1JBU campaign")
    log("============================================================")
    log(f"Target: {TARGET}")
    log(f"Framework: {FRAMEWORK}")
    log(f"Campaign: {CAMPAIGN}")
    log("")

    if not TARGET.exists():
        raise RuntimeError(f"Target does not exist: {TARGET}")

    if not FRAMEWORK.exists():
        raise RuntimeError(f"Framework does not exist: {FRAMEWORK}")

    # --------------------------------------------------------
    # CUDA sanity check
    # --------------------------------------------------------

    log("=== CUDA check ===")

    run([
        "uv", "run", "python", "-c",
        (
            "import torch; "
            "print('CUDA:', torch.cuda.is_available()); "
            "print('GPU:', torch.cuda.get_device_name(0))"
        )
    ])

    # --------------------------------------------------------
    # STEP 1: RFdiffusion
    # --------------------------------------------------------

    log("")
    log("============================================================")
    log("STEP 1: RFdiffusion")
    log("============================================================")

    if BACKBONE_QV.exists() and BACKBONE_QV.stat().st_size > 0:
        log("Backbone Quiver already exists; skipping generation.")
    else:
        run([
            "uv", "run", "rfdiffusion",
            "--target", TARGET,
            "--framework", FRAMEWORK,
            "--output-quiver", BACKBONE_QV,
            "--num-designs", str(NUM_BACKBONES),
            "--design-loops", DESIGN_LOOPS,
            "--hotspots", HOTSPOTS,
        ])

    # --------------------------------------------------------
    # Extract
    # --------------------------------------------------------

    log("")
    log("=== Extracting backbone PDBs ===")

    if BACKBONE_PDBS.exists():
        shutil.rmtree(BACKBONE_PDBS)

    BACKBONE_PDBS.mkdir(parents=True)

    run([
        "uv", "run", "qvextract",
        BACKBONE_QV,
        "-o", BACKBONE_PDBS,
    ])

    # --------------------------------------------------------
    # Filter
    # --------------------------------------------------------

    filter_backbones()

    # --------------------------------------------------------
    # ProteinMPNN
    # --------------------------------------------------------

    log("")
    log("============================================================")
    log("STEP 3: ProteinMPNN")
    log("============================================================")

    mpnn_help = get_help("proteinmpnn")

    log("Detected ProteinMPNN CLI.")

    # Prefer PDB-directory mode, because it lets us run only
    # the structures that passed filtering.
    if has_option(mpnn_help, "--pdb-dir", "--pdbdir"):

        cmd = [
            "uv", "run", "proteinmpnn",
        ]

        if "--pdb-dir" in mpnn_help:
            cmd += ["--pdb-dir", FILTER_PDBS]
        else:
            cmd += ["--pdbdir", FILTER_PDBS]

        if "--output-quiver" in mpnn_help:
            cmd += ["--output-quiver", MPNN_QV]

        elif "--outquiver" in mpnn_help:
            cmd += ["--outquiver", MPNN_QV]

        else:
            raise RuntimeError(
                "ProteinMPNN supports PDB-directory input, but I cannot "
                "identify its output-Quiver option. Run:\n"
                "uv run proteinmpnn --help"
            )

        if "--seqs-per-struct" in mpnn_help:
            cmd += [
                "--seqs-per-struct",
                str(SEQS_PER_BACKBONE),
            ]

        elif "--seqs_per_struct" in mpnn_help:
            cmd += [
                "--seqs_per_struct",
                str(SEQS_PER_BACKBONE),
            ]

        run(cmd)

    else:
        raise RuntimeError(
            "\nYour ProteinMPNN command does not expose a PDB-directory "
            "input option that this script recognizes.\n\n"
            "Everything through backbone filtering completed successfully.\n"
            "Please run:\n\n"
            "    uv run proteinmpnn --help\n\n"
            "and paste that output to me. I will wire the exact installed "
            "ProteinMPNN syntax into this script."
        )

    # --------------------------------------------------------
    # Extract MPNN
    # --------------------------------------------------------

    if MPNN_QV.exists():

        if MPNN_PDBS.exists():
            shutil.rmtree(MPNN_PDBS)

        MPNN_PDBS.mkdir(parents=True)

        run([
            "uv", "run", "qvextract",
            MPNN_QV,
            "-o", MPNN_PDBS,
        ])

    # --------------------------------------------------------
    # RF2
    # --------------------------------------------------------

    log("")
    log("============================================================")
    log("STEP 4: RF2")
    log("============================================================")

    rf2_help = get_help("rf2")

    cmd = ["uv", "run", "rf2"]

    if "--input-quiver" in rf2_help:
        cmd += ["--input-quiver", MPNN_QV]

    elif "-q" in rf2_help:
        cmd += ["-q", MPNN_QV]

    else:
        raise RuntimeError(
            "Could not identify RF2 Quiver input option.\n"
            "Run:\n    uv run rf2 --help"
        )

    if "--output-quiver" in rf2_help:
        cmd += ["--output-quiver", RF2_QV]

    elif "--outquiver" in rf2_help:
        cmd += ["--outquiver", RF2_QV]

    else:
        raise RuntimeError(
            "Could not identify RF2 Quiver output option.\n"
            "Run:\n    uv run rf2 --help"
        )

    run(cmd)

    # --------------------------------------------------------
    # Extract RF2
    # --------------------------------------------------------

    if RF2_PDBS.exists():
        shutil.rmtree(RF2_PDBS)

    RF2_PDBS.mkdir(parents=True)

    run([
        "uv", "run", "qvextract",
        RF2_QV,
        "-o", RF2_PDBS,
    ])

    log("")
    log("============================================================")
    log("CAMPAIGN COMPLETE")
    log("============================================================")
    log(f"Backbones:      {BACKBONE_PDBS}")
    log(f"Filter results: {FILTER_CSV}")
    log(f"Filtered PDBs:  {FILTER_PDBS}")
    log(f"MPNN Quiver:    {MPNN_QV}")
    log(f"RF2 Quiver:     {RF2_QV}")
    log(f"RF2 PDBs:       {RF2_PDBS}")
    log(f"Log:            {LOG}")


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print("\nStopped by user.")
        sys.exit(130)

    except Exception as e:
        print("\n============================================================")
        print("PIPELINE STOPPED")
        print("============================================================")
        print(e)
        print()
        print(
            "Anything from completed earlier stages has been preserved in:"
        )
        print(CAMPAIGN)
        sys.exit(1)
