"""
launch.py — Interactive launcher for the Bank EA Text Pipeline.
Run with:  python3 launch.py
"""

import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║      Bank Enforcement Action — Text Analysis Pipeline        ║
║                                                              ║
║  Downloads legal documents from OCC, OTS, and FRB, reads    ║
║  the text, and uses machine learning to find recurring       ║
║  problem categories across ~22,000 enforcement actions.      ║
╚══════════════════════════════════════════════════════════════╝
"""

def ask(prompt, default=None, cast=str, valid=None):
    suffix = f" [{default}]" if default is not None else ""
    while True:
        raw = input(f"{prompt}{suffix}: ").strip()
        if raw == "" and default is not None:
            return default
        try:
            val = cast(raw)
        except (ValueError, TypeError):
            print(f"  ✗ Please enter a valid {cast.__name__}.")
            continue
        if valid and val not in valid:
            print(f"  ✗ Choose one of: {', '.join(str(v) for v in valid)}")
            continue
        return val

def ask_yn(prompt, default=True):
    suffix = " [Y/n]" if default else " [y/N]"
    raw = input(f"{prompt}{suffix}: ").strip().lower()
    if raw == "":
        return default
    return raw.startswith("y")

def main():
    print(BANNER)

    # --- Input file ---
    default_csv = "enforcement_actions_round5.csv"
    csv_files = sorted(SCRIPT_DIR.glob("*.csv"))
    if csv_files:
        default_csv = csv_files[0].name

    print("Step 1: Input data")
    print(f"  CSV files found in this folder: {[f.name for f in csv_files] or 'none'}")
    csv_name = ask("  Which CSV file to use", default=default_csv)
    csv_path = SCRIPT_DIR / csv_name
    if not csv_path.exists():
        print(f"\n  ✗ File not found: {csv_path}")
        sys.exit(1)
    print(f"  ✓ Using: {csv_path.name}\n")

    # --- Sample or full run ---
    print("Step 2: Scope")
    print("  A pilot run processes a small random sample — fast, good for testing.")
    print("  A full run processes ALL non-FDIC documents (can take 30–90 min).")
    mode = ask("  Run mode (pilot / full)", default="pilot", valid=["pilot", "full"])
    sample = None
    if mode == "pilot":
        sample = ask("  How many documents to sample", default=300, cast=int)
    print()

    # --- Topics ---
    print("Step 3: Number of topics")
    print("  How many problem categories should the model look for?")
    print("  Recommended: 6–10. More topics = finer distinctions but noisier.")
    n_topics = ask("  Number of topics", default=6, cast=int)
    print()

    # --- Workers ---
    print("Step 4: Download speed")
    workers = ask("  Parallel download workers (4=safe, 8=fast, 16=aggressive)", default=8, cast=int)
    print()

    # --- Skip download ---
    print("Step 5: Caching")
    skip_dl = ask_yn("  Skip re-downloading already-cached documents?", default=True)
    print()

    # --- Output filename ---
    print("Step 6: Output")
    out_name = ask("  Output CSV filename", default="ea_with_topics.csv")
    print()

    # --- Confirm ---
    print("─" * 60)
    print("  Ready to run with these settings:")
    print(f"    Input CSV   : {csv_name}")
    print(f"    Mode        : {mode}" + (f" (sample={sample})" if sample else ""))
    print(f"    Topics      : {n_topics}")
    print(f"    Workers     : {workers}")
    print(f"    Skip cache  : {skip_dl}")
    print(f"    Output file : ea_pipeline_output/{out_name}")
    print("─" * 60)
    go = ask_yn("\n  Start the pipeline?", default=True)
    if not go:
        print("  Cancelled.")
        sys.exit(0)

    # --- Build command ---
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "ea_text_pipeline.py"),
        "--input", str(csv_path),
        "--output", out_name,
        "--n_topics", str(n_topics),
        "--workers", str(workers),
    ]
    if sample:
        cmd += ["--sample", str(sample)]
    if skip_dl:
        cmd += ["--skip_download"]

    print("\n" + "═" * 60)
    print("  PIPELINE STARTING")
    print("═" * 60 + "\n")

    result = subprocess.run(cmd, cwd=str(SCRIPT_DIR))

    if result.returncode == 0:
        print("\n" + "═" * 60)
        print("  DONE! Outputs saved to:  ea_pipeline_output/")
        print("  Open ea_pipeline_output/ea_with_topics.csv in Excel")
        print("  to see topic scores for each enforcement action.")
        print("═" * 60)
    else:
        print("\n  ✗ Pipeline exited with an error. Check output above.")
        sys.exit(result.returncode)

if __name__ == "__main__":
    main()
