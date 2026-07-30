"""00 — Pre-flight environment check.

Runs in well under a second and tells you exactly what's wrong with
your installation BEFORE any stage downloads 16 GB of model weights.

Checks:
  - Python version (>= 3.10)
  - conda env (warns if not `swarm`)
  - torch installed AND compiled with CUDA AND a GPU is visible
  - torch's compiled arch list covers the GPU's compute capability
    (catches the Blackwell sm_120 / RTX 50-series case)
  - bitsandbytes importable (needed for 4-bit quantization)
  - transformers / datasets / accelerate / sentence-transformers importable
  - PyYAML can parse `config/config.yaml`

If a check fails, prints the exact pip command to fix it.

Exit code:
  0  everything is OK
  1  one or more checks failed (caller should abort the pipeline)
"""
from __future__ import annotations

import importlib
import os
import platform
import sys
import warnings
from pathlib import Path
from typing import List, Tuple


# --------------------------------------------------------------------- env
ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ENV_NAME = "swarm"

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ----------------------------------------------------------------- helpers
def _heading(text: str) -> None:
    print()
    print(text)
    print("-" * len(text))


def _check_python() -> Tuple[bool, str]:
    ver = sys.version_info
    ok = ver >= (3, 10)
    msg = f"Python {ver.major}.{ver.minor}.{ver.micro}"
    return ok, msg


def _check_conda_env() -> Tuple[bool, str]:
    name = os.environ.get("CONDA_DEFAULT_ENV") or (
        Path(os.environ["CONDA_PREFIX"]).name if os.environ.get("CONDA_PREFIX") else None
    )
    ok = name == EXPECTED_ENV_NAME
    msg = f"conda env = {name!r} (expected {EXPECTED_ENV_NAME!r})"
    return ok, msg


def _torch_cu128_fix(reason: str) -> List[str]:
    """The standard 'reinstall torch with the right CUDA wheel' fix command.

    cu128 wheels support every NVIDIA arch from Pascal through Blackwell
    (sm_60 to sm_120), so they're the safest default in 2026 — they work
    on every supported card including the RTX 50-series.
    """
    return [
        f"# {reason}",
        "pip uninstall -y torch torchvision torchaudio",
        "pip install --upgrade torch torchvision torchaudio \\",
        "    --index-url https://download.pytorch.org/whl/cu128",
    ]


def _check_torch_cuda() -> Tuple[bool, str, List[str]]:
    """Return (ok, msg, fix_commands).

    Also catches the case where torch.cuda.is_available() returns True
    but the GPU's compute capability isn't in torch.cuda.get_arch_list()
    — i.e. the wheel doesn't have SASS code for this GPU. That's what
    happens with RTX 50-series + cu124 wheels (Blackwell sm_120).
    """
    try:
        import torch
    except Exception as e:
        return (
            False,
            f"torch not installed: {e}",
            _torch_cu128_fix("torch not installed"),
        )

    cuda_build = torch.backends.cuda.is_built()
    cuda_avail = torch.cuda.is_available()
    if not cuda_build:
        return (
            False,
            f"torch {torch.__version__} is CPU-only (cuda built: {cuda_build})",
            _torch_cu128_fix("Your torch wheel is CPU-only. Reinstall with the CUDA build:"),
        )
    if not cuda_avail:
        return (
            False,
            f"torch {torch.__version__} CUDA-built but torch.cuda.is_available() == False",
            [
                "torch has CUDA support but no GPU is visible.",
                "Check the NVIDIA driver is installed and run `nvidia-smi`.",
            ],
        )

    name = torch.cuda.get_device_name(0)
    cap_t = torch.cuda.get_device_capability(0)
    sm = f"sm_{cap_t[0]}{cap_t[1]}"
    try:
        arch_list = list(torch.cuda.get_arch_list())
    except Exception:
        arch_list = []

    # If torch knows about archs and ours isn't there, the wheel won't
    # actually run kernels on this GPU even though .is_available() lies.
    if arch_list and sm not in arch_list:
        return (
            False,
            (
                f"torch {torch.__version__} compiled for {arch_list}; "
                f"your GPU ({name}) needs {sm}. Tensor ops will silently "
                f"fail or fall back to CPU."
            ),
            _torch_cu128_fix(
                f"Your torch wheel was built without {sm} support. "
                f"For RTX 50-series (Blackwell) cards you need the cu128 wheels:"
            ),
        )

    # Final sanity: actually allocate and add a tensor on the GPU.
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)  # promote the sm_xxx warning to an exception
            a = torch.zeros(8, device="cuda")
            b = a + 1
            _ = b.sum().item()
    except Exception as e:
        return (
            False,
            f"torch {torch.__version__} on {name} ({sm}) but GPU op failed: {e}",
            _torch_cu128_fix(
                f"torch can't run kernels on your GPU ({sm}). Reinstall the "
                f"cu128 wheels (Blackwell-capable):"
            ),
        )

    return True, f"torch {torch.__version__} on {name} ({sm})", []


def _check_module(modname: str, fix_pkg: str) -> Tuple[bool, str, List[str]]:
    try:
        m = importlib.import_module(modname)
        ver = getattr(m, "__version__", "?")
        return True, f"{modname} {ver}", []
    except Exception as e:
        return False, f"{modname} not importable: {e}", [f"pip install --upgrade {fix_pkg}"]


def _check_bitsandbytes() -> Tuple[bool, str, List[str]]:
    try:
        import bitsandbytes as bnb  # noqa
        return True, f"bitsandbytes {bnb.__version__}", []
    except Exception as e:
        return False, f"bitsandbytes not importable: {e}", [
            "pip install --upgrade bitsandbytes"
        ]


def _check_config() -> Tuple[bool, str, List[str]]:
    cfg_path = ROOT / "config" / "config.yaml"
    if not cfg_path.exists():
        return False, f"missing {cfg_path}", []
    try:
        import yaml
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        n_models = len(cfg.get("models", []))
        n_datasets = len(cfg.get("datasets", {}))
        return True, f"config.yaml ({n_models} models, {n_datasets} datasets)", []
    except Exception as e:
        return False, f"config.yaml unreadable: {e}", []


# ------------------------------------------------------------------- main
def main() -> int:
    _heading("Hallucination Pattern Detection — pre-flight check")
    print(f"platform: {platform.platform()}")
    print(f"python  : {sys.executable}")

    results: List[Tuple[str, bool, str, List[str]]] = []

    ok, msg = _check_python();           results.append(("python>=3.10", ok, msg, []))
    ok, msg = _check_conda_env();        results.append(("conda env",     ok, msg, []))
    ok, msg, fix = _check_torch_cuda();  results.append(("torch + CUDA",  ok, msg, fix))
    ok, msg, fix = _check_module("transformers", "transformers>=4.44.0")
    results.append(("transformers", ok, msg, fix))
    ok, msg, fix = _check_module("accelerate", "accelerate>=0.33.0")
    results.append(("accelerate", ok, msg, fix))
    ok, msg, fix = _check_bitsandbytes(); results.append(("bitsandbytes", ok, msg, fix))
    ok, msg, fix = _check_module("datasets", "datasets>=2.20.0")
    results.append(("datasets", ok, msg, fix))
    ok, msg, fix = _check_module("sentence_transformers", "sentence-transformers>=3.0.1")
    results.append(("sentence-transformers", ok, msg, fix))
    ok, msg, fix = _check_module("sklearn", "scikit-learn>=1.4")
    results.append(("scikit-learn", ok, msg, fix))
    ok, msg, fix = _check_module("yaml", "PyYAML>=6.0")
    results.append(("PyYAML", ok, msg, fix))
    ok, msg, fix = _check_config();      results.append(("config.yaml", ok, msg, fix))

    _heading("Results")
    fails: List[Tuple[str, str, List[str]]] = []
    for name, ok, msg, fix in results:
        status = "[OK]   " if ok else "[FAIL] "
        soft = name in ("python>=3.10", "conda env")
        if not ok and soft:
            status = "[WARN] "
        print(f"  {status} {name:<24} {msg}")
        if not ok and not soft:
            fails.append((name, msg, fix))

    if not fails:
        _heading("Pre-flight: PASS")
        print("All required components are installed correctly. Safe to run the pipeline.")
        return 0

    _heading("Pre-flight: FAIL")
    print(f"{len(fails)} required check(s) failed. Fix commands:\n")
    for name, msg, fix in fails:
        print(f"  * {name}: {msg}")
        for cmd in fix:
            for line in cmd.splitlines():
                print(f"        {line}")
        print()
    print(
        "After fixing, re-run:\n"
        "    python scripts/00_check_install.py\n"
        "and once it passes:\n"
        "    python scripts/run_pipeline.py"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
