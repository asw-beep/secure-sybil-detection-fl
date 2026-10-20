"""Pick detector distance thresholds from the seed-0 tuning runs (validation only).

Reads results/suite/tune/{clean_s0,attack_s0}/shadow_detection.csv, sums detection counts over
all rounds for every (method, k, tau), and for each detector variant chooses

    tau* = argmax F1(attack run)  subject to  FPR(clean run) <= MAX_FPR and FPR(attack run) <= MAX_FPR,

breaking ties by the larger tau (more margin against evasion). Variants with no feasible tau
fall back to the tau with the lowest clean FPR and are marked infeasible.
Writes configs/tuned_thresholds.yaml (read by run_suite.py) and results/suite/tune/tuning_table.csv.
The held-out seeds 1-3 are never used here.

Usage:  python experiments/tune_thresholds.py
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import yaml

from sybil_fl.metrics import detection_scores

REPO = Path(__file__).resolve().parents[1]
TUNE = REPO / "results" / "suite" / "tune"
MAX_FPR = 0.05


def totals(path: Path) -> dict[tuple[str, int, float], dict[str, int]]:
    acc: dict = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            key = (r["method"], int(r["k"]), float(r["tau"]))
            for c in ("tp", "fp", "fn", "tn"):
                acc[key][c] += int(r[c])
    return acc


def main() -> None:
    clean = totals(TUNE / "clean_s0" / "shadow_detection.csv")
    attack = totals(TUNE / "attack_s0" / "shadow_detection.csv")
    rows = []
    for key in sorted(attack):
        a, c = detection_scores(**attack[key]), detection_scores(**clean[key])
        rows.append({"method": key[0], "k": key[1], "tau": key[2], "f1_attack": a["f1"],
                     "precision_attack": a["precision"], "recall_attack": a["recall"],
                     "fpr_attack": a["fpr"], "fpr_clean": c["fpr"]})

    chosen = {}
    for variant in sorted({(r["method"], r["k"]) for r in rows}):
        cand = [r for r in rows if (r["method"], r["k"]) == variant]
        feasible = [r for r in cand if r["fpr_clean"] <= MAX_FPR and r["fpr_attack"] <= MAX_FPR]
        if feasible:
            best = max(feasible, key=lambda r: (r["f1_attack"], r["tau"]))
        else:
            best = min(cand, key=lambda r: (r["fpr_clean"], -r["tau"]))
        name = "full" if variant[0] == "full" else f"projected_k{variant[1]}"
        chosen[name] = {"tau": best["tau"], "feasible": bool(feasible), "f1_attack": best["f1_attack"],
                        "fpr_attack": best["fpr_attack"], "fpr_clean": best["fpr_clean"]}

    with open(TUNE / "tuning_table.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    header = (
        "# Detector distance thresholds tuned on seed 0 only (experiments/tune_thresholds.py).\n"
        f"# Rule: max F1 on the attacked run s.t. FPR <= {MAX_FPR} on the clean and attacked runs; ties -> larger tau.\n"
        "# Validation scores per variant are in the comments; full table: results/suite/tune/tuning_table.csv\n"
    )
    body = {name: v["tau"] for name, v in chosen.items()}
    notes = "".join(
        f"#   {name}: tau {v['tau']}, feasible {v['feasible']}, F1 {v['f1_attack']:.3f}, "
        f"FPR attack {v['fpr_attack']:.3f}, FPR clean {v['fpr_clean']:.3f}\n"
        for name, v in chosen.items()
    )
    (REPO / "configs" / "tuned_thresholds.yaml").write_text(
        header + notes + yaml.safe_dump(body, sort_keys=False), encoding="utf-8", newline="\n")
    print(header + notes + yaml.safe_dump(body, sort_keys=False))


if __name__ == "__main__":
    main()
