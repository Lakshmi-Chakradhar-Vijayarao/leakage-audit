"""Cross-platform end-to-end pipeline runner.

Runs every stage of the Hallucination Pattern Detection pipeline by
invoking the numbered scripts with the *current* Python interpreter
(`sys.executable`). Works identically on Windows, macOS, and Linux —
no shell, batch, or PowerShell required.

Stage 00 (pre-flight check) is run automatically before any other
stage. If it fails (e.g. torch is CPU-only, bitsandbytes missing,
etc.), the pipeline aborts BEFORE downloading any model weights.

Usage (from the project root, with the `swarm` conda env activated):

    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --skip 04 05      # skip stages 04 and 05
    python scripts/run_pipeline.py --only 02         # run only stage 02
    python scripts/run_pipeline.py --dry-run         # print plan, don't run
    python scripts/run_pipeline.py --check-env       # full pre-flight + exit
    python scripts/run_pipeline.py --stop-on-error   # halt on first failure
    python scripts/run_pipeline.py --skip-preflight  # skip pre-flight (not advised)
"""
from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence


# ---------------------------------------------------------------- env tweaks
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# Project root (one level above this script)
ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"

PREFLIGHT_SCRIPT = "00_check_install.py"

# Ordered list of pipeline stages
STAGES: List[tuple] = [
    ("01", "01_download_data.py",          "download + prepare datasets"),
    ("02", "02_extract_hidden_states.py",  "extract per-layer hidden states"),
    ("03", "03_train_probes.py",           "train SAPLMA-style probes"),
    ("04", "04_run_inside.py",             "INSIDE / semantic entropy"),
    ("05", "05_run_self_consistency.py",   "self-consistency scoring"),
    ("06", "06_analyze_attention.py",      "attention-entropy analysis"),
    ("07", "07_aggregate_results.py",      "aggregate results -> tables"),
    ("08", "08_generate_visualizations.py","generate all figures"),
]

EXPECTED_ENV_NAME = "swarm"


# --------------------------------------------------------------------- env
def _detect_conda_env() -> Optional[str]:
    """Return the active conda env name, if any."""
    name = os.environ.get("CONDA_DEFAULT_ENV")
    if name:
        return name
    prefix = os.environ.get("CONDA_PREFIX")
    if prefix:
        return Path(prefix).name
    return None


def print_env_banner() -> None:
    env_name = _detect_conda_env()
    print("[env]   python            :", sys.executable)
    print("[env]   python version    :", sys.version.split()[0])
    print("[env]   platform          :", platform.platform())
    print("[env]   conda env (active):", env_name or "(none detected)")
    print("[env]   working directory :", os.getcwd())
    if env_name != EXPECTED_ENV_NAME:
        print(
            f"[env]   WARNING: expected conda env '{EXPECTED_ENV_NAME}' "
            f"but active env is '{env_name}'.\n"
            f"          conda activate {EXPECTED_ENV_NAME}\n"
            f"          python scripts/run_pipeline.py"
        )


def run_preflight() -> int:
    """Run scripts/00_check_install.py with the current interpreter."""
    pre = SCRIPTS_DIR / PREFLIGHT_SCRIPT
    if not pre.exists():
        print(f"[preflight] missing {pre}; skipping")
        return 0
    print("\n[preflight] running scripts/00_check_install.py ...")
    proc = subprocess.run([sys.executable, str(pre)], cwd=str(ROOT))
    return proc.returncode


# ------------------------------------------------------------------- runner
def run_stage(stage_id: str, filename: str, description: str,
              dry_run: bool = False) -> int:
    """Invoke one stage with `sys.executable`. Return the exit code."""
    script_path = SCRIPTS_DIR / filename
    if not script_path.exists():
        print(f"[run]   [{stage_id}] script not found: {script_path}")
        return 127
    cmd = [sys.executable, str(script_path)]
    pretty = " ".join(cmd)
    bar = "=" * 78
    print(f"\n{bar}\n[stage {stage_id}] {description}\n{bar}\n[cmd]    {pretty}")
    if dry_run:
        print("[run]   --dry-run set, skipping execution")
        return 0
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(ROOT))
    dt = time.time() - t0
    status = "ok" if proc.returncode == 0 else f"FAILED (exit={proc.returncode})"
    print(f"[run]   [{stage_id}] {status}  ({dt:.1f}s)")
    return proc.returncode


# --------------------------------------------------------------------- main
def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cross-platform pipeline runner for the Hallucination Pattern Detection project."
    )
    p.add_argument("--only", nargs="+", metavar="STAGE_ID",
                   help="Run only these stage IDs (e.g. --only 03 08).")
    p.add_argument("--skip", nargs="+", metavar="STAGE_ID",
                   help="Skip these stage IDs.")
    p.add_argument("--from", dest="from_stage", metavar="STAGE_ID",
                   help="Start at this stage and continue to the end.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the plan without invoking any scripts.")
    p.add_argument("--check-env", action="store_true",
                   help="Run the full pre-flight install check and exit.")
    p.add_argument("--skip-preflight", action="store_true",
                   help="Do NOT run scripts/00_check_install.py before the pipeline.")
    p.add_argument("--stop-on-error", action="store_true",
                   help="Halt the pipeline on the first non-zero exit code.")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)

    print_env_banner()

    # --check-env  ->  run the pre-flight only and exit
    if args.check_env:
        return run_preflight()

    # --skip-preflight skips the install check (not recommended).
    if not args.skip_preflight and not args.dry_run:
        code = run_preflight()
        if code != 0:
            print(
                "\n[preflight] FAILED — aborting pipeline. Fix the issues "
                "printed above, then re-run `python scripts/run_pipeline.py`."
            )
            return code

    # Build the ordered stage list with --only / --skip / --from applied.
    plan: List[tuple] = []
    started = args.from_stage is None
    for sid, fn, desc in STAGES:
        if args.from_stage and sid == args.from_stage:
            started = True
        if not started:
            continue
        if args.only and sid not in args.only:
            continue
        if args.skip and sid in args.skip:
            continue
        plan.append((sid, fn, desc))

    if not plan:
        print("[plan]  nothing to run.")
        return 0

    print("\n[plan]  stages to run:")
    for sid, fn, desc in plan:
        print(f"        [{sid}] {fn}  -- {desc}")

    failures: List[str] = []
    for sid, fn, desc in plan:
        code = run_stage(sid, fn, desc, dry_run=args.dry_run)
        if code != 0:
            failures.append(sid)
            if args.stop_on_error:
                print(f"\n[run]   aborting after failed stage [{sid}] (stop-on-error).")
                break

    print("\n" + "=" * 78)
    if failures:
        print(f"[done]  finished with FAILURES in stages: {', '.join(failures)}")
        return 1
    print("[done]  all stages completed successfully.")
    print("[done]  results under:", (ROOT / "results").as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
