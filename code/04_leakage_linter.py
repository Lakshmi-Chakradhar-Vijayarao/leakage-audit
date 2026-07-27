"""
Paper 2 -- final-audit stretch goal: a scoped, semi-automated prevalence
check for the four leakage patterns this paper's own §5 checklist
documents, run against real external hidden-state-probing repos rather
than only the four already-manually-audited case studies.

HONEST SCOPE, stated before any result: this is a REGEX-BASED HEURISTIC
SCANNER, not a validated static-analysis tool. It flags candidate lines
for four patterns; every flag is then manually read and classified as a
true or false positive by us before anything is reported as a finding.
We do not claim precision/recall numbers for the scanner itself -- only
for the manually-confirmed results after review. This is a "scoped
prevalence check," not a systematic literature review: 7 external repos,
found via targeted search, not a random or exhaustive sample of the
field, and every caveat that implies (selection bias toward repos that
came up in search, no claim of representativeness) applies to whatever
this finds.

Four detectors, one per case-study pattern already in this paper:
  P1: full-dataset fit-then-score (Case Study 1, HaRP pattern) --
      a `.fit(`/`.fit_transform(` call with no `train_test_split` (or
      equivalent split keyword) anywhere else in the same file.
  P2: CV-based hyperparameter/feature selection reported as the final
      number, no nested/outer held-out evaluation (Case Study 2,
      GUARDIAN pattern) -- GridSearchCV/RandomizedSearchCV/cross_val_score/
      cross_validate present, with no separate held-out `.score(`/`.predict(`
      call on a variable that looks like an outer test set.
  P3: per-fold checkpoint/layer selection re-using the fold later
      reported as held-out (Case Study 3, MultiHaluDet pattern) -- a
      `best_epoch`/`checkpoint` selection driven by a validation-fold
      variable that recurs in a later "final"/"test" evaluation call.
  P4: test-set-driven argmax over many hypotheses, no correction (Case
      Study 4, quantized-LLM pattern) -- an argmax/max selection over a
      loop-computed metric whose variable name contains "test", with no
      multiple-comparison-correction keyword anywhere in the file.

Each detector is a deliberately simple, readable heuristic -- not an AST
analysis -- precisely so a human reviewer (us) can read the flagged
context and judge it, rather than trust an opaque tool.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXTERNAL_DIR = ROOT / "code" / "external"
OUT_PATH = ROOT / "results" / "leakage_linter_report.json"

SPLIT_KEYWORDS = re.compile(r"train_test_split|StratifiedKFold|KFold\(|random_split|\.split\(")
CV_SELECTION_KEYWORDS = re.compile(r"GridSearchCV|RandomizedSearchCV|cross_val_score|cross_validate")
CORRECTION_KEYWORDS = re.compile(r"bonferroni|holm|benjamini|fdr|multiple.?comparison|correction", re.IGNORECASE)
FIT_CALL = re.compile(r"\.fit(_transform)?\(\s*([A-Za-z_][A-Za-z0-9_\.]*)")
TEST_VAR = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*test[A-Za-z0-9_]*\b", re.IGNORECASE)
ARGMAX_KEYWORDS = re.compile(r"argmax|best_layer|best_epoch|best_checkpoint", re.IGNORECASE)
CHECKPOINT_KEYWORDS = re.compile(r"best_epoch|best_checkpoint|checkpoint", re.IGNORECASE)
FINAL_EVAL_KEYWORDS = re.compile(r"\.score\(|\.predict\(|\.evaluate\(")


def iter_py_files(repo_dir):
    for p in repo_dir.rglob("*.py"):
        if any(part in {".git", "venv", "__pycache__", "node_modules", ".venv"} for part in p.parts):
            continue
        yield p


def detect_p1(repo_dir):
    """Full-dataset fit-then-score: .fit(/.fit_transform( with no split keyword anywhere in the file."""
    flags = []
    for f in iter_py_files(repo_dir):
        try:
            text = f.read_text(errors="ignore")
        except Exception:
            continue
        if not FIT_CALL.search(text):
            continue
        if SPLIT_KEYWORDS.search(text):
            continue  # a split exists somewhere in this file -- not flagged
        for i, line in enumerate(text.splitlines(), 1):
            m = FIT_CALL.search(line)
            if m:
                flags.append({"file": str(f.relative_to(repo_dir)), "line": i, "snippet": line.strip()[:160]})
    return flags


def detect_p2(repo_dir):
    """CV-based selection reported as final: CV-selection keyword present, no distinct held-out eval call."""
    flags = []
    for f in iter_py_files(repo_dir):
        try:
            text = f.read_text(errors="ignore")
        except Exception:
            continue
        if not CV_SELECTION_KEYWORDS.search(text):
            continue
        if FINAL_EVAL_KEYWORDS.search(text) and re.search(r"test", text, re.IGNORECASE):
            continue  # a distinct test-set evaluation call also appears -- not flagged
        for i, line in enumerate(text.splitlines(), 1):
            if CV_SELECTION_KEYWORDS.search(line):
                flags.append({"file": str(f.relative_to(repo_dir)), "line": i, "snippet": line.strip()[:160]})
    return flags


def detect_p3(repo_dir):
    """Checkpoint/layer selection reusing the fold later reported as held-out: checkpoint keyword
    and a 'test'-named variable co-occurring within a few lines."""
    flags = []
    for f in iter_py_files(repo_dir):
        try:
            lines = f.read_text(errors="ignore").splitlines()
        except Exception:
            continue
        for i, line in enumerate(lines):
            if CHECKPOINT_KEYWORDS.search(line):
                window = "\n".join(lines[max(0, i - 3):i + 4])
                if TEST_VAR.search(window):
                    flags.append({"file": str(f.relative_to(repo_dir)), "line": i + 1, "snippet": line.strip()[:160]})
    return flags


def detect_p4(repo_dir):
    """Test-set-driven argmax over many hypotheses, no correction keyword anywhere in the file."""
    flags = []
    for f in iter_py_files(repo_dir):
        try:
            text = f.read_text(errors="ignore")
        except Exception:
            continue
        if not ARGMAX_KEYWORDS.search(text):
            continue
        if not TEST_VAR.search(text):
            continue
        if CORRECTION_KEYWORDS.search(text):
            continue  # correction language present somewhere -- not flagged
        for i, line in enumerate(text.splitlines(), 1):
            if ARGMAX_KEYWORDS.search(line) and TEST_VAR.search(line):
                flags.append({"file": str(f.relative_to(repo_dir)), "line": i, "snippet": line.strip()[:160]})
    return flags


def main():
    repos = sorted([d for d in EXTERNAL_DIR.iterdir() if d.is_dir() and not d.name.startswith(".")])
    print(f"Scanning {len(repos)} repos: {[r.name for r in repos]}")

    report = {}
    for repo_dir in repos:
        n_files = len(list(iter_py_files(repo_dir)))
        p1 = detect_p1(repo_dir)
        p2 = detect_p2(repo_dir)
        p3 = detect_p3(repo_dir)
        p4 = detect_p4(repo_dir)
        report[repo_dir.name] = {
            "n_py_files": n_files,
            "P1_full_dataset_fit_no_split": p1,
            "P2_cv_selection_no_held_out_eval": p2,
            "P3_checkpoint_reuses_test_var": p3,
            "P4_test_argmax_no_correction": p4,
            "any_flag": bool(p1 or p2 or p3 or p4),
        }
        print(f"\n{repo_dir.name} ({n_files} .py files): "
              f"P1={len(p1)} P2={len(p2)} P3={len(p3)} P4={len(p4)}")

    n_any_flag = sum(1 for r in report.values() if r["any_flag"])
    print(f"\n{n_any_flag}/{len(repos)} repos have at least one raw (unreviewed) flag in some pattern.")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
