"""Run the study's experiment groups (docs/EXECUTION_PLAN.md) in parallel worker processes.

Groups (each run = a base config file + recorded overrides + a seed):
  tune  seed 0: FedAvg trajectories (clean, attack) with shadow scoring over SHADOW_KS x TAU_GRID;
        experiments/tune_thresholds.py picks the defence thresholds from these.
  e2    held-out seeds 1-3: {fedavg, median, full detector, projected detector} x {clean, attack};
        the undefended FedAvg runs also carry shadow scoring (held-out detection vs k, E1 distortion).
  e3    seeds 1-2: clean and attacked FedAvg at Dirichlet alpha 0.1 and 5.0 with shadow scoring
        (alpha 0.5 comes from e2): false positives under label skew.
  e6    seeds 1-2: projected defence (k = 64) under evasion {known, guessed projection, gaussian}
        x EVASION_BUDGETS; the full-dimensional detector is scored in shadow on the same updates.
Defended runs read their thresholds from configs/tuned_thresholds.yaml (written by tune_thresholds.py).

Each run writes to results/suite/<group>/<name>/ and is skipped if its meta.json already has a
summary (so an interrupted suite resumes). Every run gets its own server key (generated unless
SYBIL_FL_KEY is set); only the key source is recorded.

Usage:  python experiments/run_suite.py --group tune [--workers 4 --threads 4]
"""

from __future__ import annotations

import argparse
import copy
import json
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
SUITE_DIR = REPO / "results" / "suite"
TUNED = REPO / "configs" / "tuned_thresholds.yaml"

TAU_GRID = [0.02, 0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
SHADOW_KS = [16, 32, 64, 128, 256]
HELD_OUT = [1, 2, 3]
EVASION_MODES = ["known_projection", "guessed_projection", "gaussian"]
EVASION_BUDGETS = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
SHADOW = {"evaluation.shadow_ks": SHADOW_KS, "evaluation.shadow_thresholds": TAU_GRID}


def tuned() -> dict:
    if not TUNED.exists():
        raise SystemExit(f"{TUNED.relative_to(REPO)} not found: run the tune group and tune_thresholds.py first")
    return yaml.safe_load(TUNED.read_text())


def defended(detector: str, t: dict) -> dict:
    tau = t["full"] if detector == "full" else t["projected_k64"]
    return {"defence.detector": detector, "defence.distance_threshold": tau}


def jobs_for(group: str) -> list[tuple[str, str, dict]]:
    """(run name, base config file, overrides) for one group."""
    jobs = []
    if group == "tune":
        for scen, base in (("clean", "default.yaml"), ("attack", "label_flip.yaml")):
            jobs.append((f"{scen}_s0", base, {"seed": 0, **SHADOW}))
    elif group == "e2":
        t = tuned()
        for s in HELD_OUT:
            for scen, base in (("clean", "default.yaml"), ("attack", "label_flip.yaml")):
                jobs.append((f"{scen}_fedavg_s{s}", base, {"seed": s, **SHADOW}))
                jobs.append((f"{scen}_median_s{s}", base, {"seed": s, "defence.aggregator": "median"}))
                jobs.append((f"{scen}_full_s{s}", base, {"seed": s, **defended("full", t)}))
                jobs.append((f"{scen}_projected_s{s}", base, {"seed": s, **defended("projected", t)}))
    elif group == "e3":
        for alpha in (0.1, 5.0):
            for s in (1, 2):
                for scen, base in (("clean", "default.yaml"), ("attack", "label_flip.yaml")):
                    jobs.append((f"{scen}_alpha{alpha}_s{s}", base,
                                 {"seed": s, "data.dirichlet_alpha": alpha, **SHADOW}))
    elif group == "e6":
        t = tuned()
        shadow = {"evaluation.shadow_ks": [64], "evaluation.shadow_thresholds": sorted({t["full"], t["projected_k64"]})}
        for s in (1, 2):
            for mode in EVASION_MODES:
                for eps in EVASION_BUDGETS:
                    jobs.append((f"{mode}_eps{eps}_s{s}", "label_flip.yaml", {
                        "seed": s, **defended("projected", t), **shadow,
                        "attack.evasion": mode, "attack.evasion_budget": eps,
                    }))
    else:
        raise SystemExit(f"unknown group {group!r}")
    return jobs


def apply_overrides(raw: dict, overrides: dict) -> dict:
    out = copy.deepcopy(raw)
    for dotted, value in overrides.items():
        node = out
        *path, leaf = dotted.split(".")
        for p in path:
            node = node[p]
        if leaf not in node:
            raise KeyError(f"override {dotted!r} does not name an existing config key")
        node[leaf] = value
    return out


_DATA = None


def _run(group: str, name: str, base: str, overrides: dict, threads: int) -> tuple[str, dict | None, str | None]:
    global _DATA
    import torch

    from sybil_fl.config import Config
    from sybil_fl.crypto import load_server_key
    from sybil_fl.data import load_mnist
    from sybil_fl.experiment import run_experiment

    torch.set_num_threads(threads)
    out = SUITE_DIR / group / name
    try:
        raw = yaml.safe_load((REPO / "configs" / base).read_text())
        cfg = Config.from_dict(apply_overrides(raw, overrides))
        if _DATA is None:
            data_dir = Path(cfg.data.data_dir)
            _DATA = load_mnist(data_dir if data_dir.is_absolute() else REPO / data_dir, download=False)
        key, source = load_server_key()
        meta = run_experiment(cfg, _DATA, out, key, source, config_file=f"configs/{base}", label=f"{group}/{name}")
        meta["overrides"] = overrides
        meta["torch_threads"] = threads
        (out / "meta.json").write_text(json.dumps(meta, indent=2))
        return name, meta["summary"], None
    except Exception:  # record and continue with the other runs
        out.mkdir(parents=True, exist_ok=True)
        (out / "error.txt").write_text(traceback.format_exc())
        return name, None, traceback.format_exc().splitlines()[-1]


def done(group: str, name: str) -> bool:
    meta = SUITE_DIR / group / name / "meta.json"
    return meta.exists() and "summary" in json.loads(meta.read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True, choices=["tune", "e2", "e3", "e6"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4, help="torch threads per worker")
    args = ap.parse_args()

    todo = [j for j in jobs_for(args.group) if not done(args.group, j[0])]
    print(f"group {args.group}: {len(todo)} runs to do ({args.workers} workers x {args.threads} threads)", flush=True)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_run, args.group, n, b, o, args.threads) for n, b, o in todo]
        for i, f in enumerate(as_completed(futures), 1):
            name, s, err = f.result()
            elapsed = (time.perf_counter() - t0) / 60
            if err:
                print(f"[{i}/{len(todo)} {elapsed:.1f} min] {name}: ERROR {err}", flush=True)
            else:
                d = s["detection_scores"]
                print(f"[{i}/{len(todo)} {elapsed:.1f} min] {name}: acc {s['last_half_mean_test_acc']:.4f} "
                      f"asr {s['last_half_mean_asr']:.4f} det f1 {d['f1']:.3f} fpr {d['fpr']:.3f} "
                      f"skipped {s['skipped_rounds']}", flush=True)


if __name__ == "__main__":
    main()
