# Execution plan — complete code and results (validated 2026-10-20)

**Status: complete (2026-10-20).** Steps 0–8 done; findings in `docs/RESULTS.md`, log in
`docs/PROGRESS.md`. Nothing was cut from the plan.

Scope agreed with the author: finish the code and all results in one push; the report and
slides are written later from `docs/PROGRESS.md`, `docs/DECISIONS.md` and `docs/RESULTS.md`.
Cryptography is the priority. Work runs end to end without per-phase review stops; each step
is committed to `main` and logged in `docs/PROGRESS.md`.

## Strategy

- Cryptography first: HKDF-SHA256 round keys + keyed SHAKE-256 projection matrix (D3, D4),
  and E6 — what keeping the projection key secret buys against an adaptive attacker — as the
  headline experiment.
- Shadow evaluation: the detector scores itself on every round's updates for several `k` and
  thresholds without affecting aggregation, so one run yields many detection measurements.
  Full defended runs are used where the model outcome (accuracy, ASR) matters.
- Thresholds tuned on seed 0; results reported on held-out seeds 1–3.
- Mixed or negative results are reported as found; no security proof is claimed.

## Steps

| # | Step | Output |
|---|---|---|
| 0 | Commit crypto core; attack default 8 Sybils × 500 pool images | commit |
| 1 | DECISIONS D3/D4 rewritten for HKDF + SHAKE-256; test that the key never reaches outputs | docs, test |
| 2 | Detector (normalise → secret projection → cosine → average-linkage clustering → flag clusters ≥ m → filtered FedAvg, skip-round fallback), full-dimensional counterpart, detection metrics | `detector.py`, tests |
| 3 | Coordinate-wise median baseline; defence selectable by config | `aggregators.py` |
| 4 | Threshold tuning on seed 0 (best F1 with FPR ≤ 5%), then frozen | config |
| 5 | E6 evasion: perturbation budget ε × attacker {knows matrix, guesses matrix, Gaussian noise} | results |
| 6 | E1 (cosine distortion vs k; SHAKE-256 vs PCG64 cost), E2 (FedAvg / median / full / projected under attack, 3 seeds), E3 (clean false positives at α ∈ {0.1, 0.5, 5}), E5 (synthetic runtime vs n, d) | results |
| 7 | Plots from saved results; `docs/RESULTS.md` with tables | plots, docs |
| 8 | Final check: tests, clean reproduction, no secrets, README | commit |

## Cut order if time runs short

E3 α = 5 → E2 down to 1 seed → smaller E5 grid → median baseline.
Never cut: crypto core, E6, E2 main comparison, E3 clean false positives, limitations.

## Out of scope (future work)

Multi-Krum, backdoor, sparse projection, reputation EMA, Fashion-MNIST, Sybil-fraction
sweep (E4), dashboard, report and slides (written later by the author).
