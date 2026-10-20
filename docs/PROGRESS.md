# Progress log

Plan: the revised project plan, kept outside this repository; § numbers refer to it (roadmap §12, working agreement §11).
One phase at a time; each phase ends with a full test run and a pause for review.

## Phase 0 — Design freeze and environment (done, 2026-10-19)

- Repo skeleton (`src/` layout, `pyproject.toml`, `.gitignore`); working agreement from plan §11.
- `configs/default.yaml` + `config.py` (typed, unknown keys rejected, ranges validated).
- Design decisions frozen in `docs/DECISIONS.md` (D1–D8).

## Phase 1 — Data and model (done, 2026-10-19)

Built: `data.py` (MNIST loader, IID + Dirichlet partitions, `validate_partition`),
`models.py` (SmallCNN, flat-parameter helpers), `training.py` (seeded SGD, accuracy),
`utils.py` (`derive_seed`), `experiments/run_central.py` (E0).

Tests: `pytest` → **29 passed** (config 5, data 13 incl. real-MNIST shapes, models/training 11).
Each test was run and seen failing (module missing) before its implementation was written.

E0 result (`results/e0_central_20261019T111952Z.json`, seed 42, default config):

| Check | Result |
|---|---|
| Dirichlet α=0.5, 20 clients | sizes 1092 / 2387 / 6774 (min/median/max); mean majority-class fraction 0.37 |
| Coverage | all 60,000 training samples assigned exactly once |
| Central training | test acc 0.102 → 0.9835 (epoch 1) → 0.9864 (epoch 2); ~30 s on 2 CPU cores |

Environment for these runs: Python 3.13, torch 2.14.1+cpu, torchvision 0.29.1, numpy 2.x, Linux.

Windows re-run via `setup_windows.bat` (`results/e0_central_20261019T122347Z.json`, same config):
Python 3.12.10, torch 2.14.1+cpu, torchvision 0.29.1+cpu, numpy 2.5.2, Windows 11.
Partition identical (1092 / 2387 / 6774, majority fraction 0.37); test acc 0.1018 → 0.9835 → 0.9859;
`pytest` → **29 passed** (the MNIST test now runs). Epoch-2 accuracy differs from Linux by 0.0005:
seeded runs are reproducible on one machine, not bit-identical across platforms (floating-point
differences), so cross-machine comparisons should use metrics, not exact weights.

**Exit criteria met:** the central model learns; the partition tests pass.

### Rulings made in Phases 0–1

Each line: what was decided, why, and what it costs if it is wrong.

1. The plan has phases, not task-by-task steps, so it was executed as plan §12 Phase 0 → 1
   and work stopped there for review (§11: "stop and wait for review before the next phase").
   *Cost if wrong:* Phase 2 starts one review later.
2. Projection RNG: the plan's recommended "course prototype" path (HMAC-SHA256 seed → PCG64),
   with no claim of cryptographic unpredictability (DECISIONS D3). *Cost if wrong:* swap the
   generator inside `projection.py` (Phase 4); no other module changes.
3. CNN sized to d = 46,730 with no BatchNorm (D5), so a dense k=512 projection fits in ~96 MB
   and updates are parameters only. *Cost if wrong:* the runtime study uses the synthetic
   benchmark (E5) to vary d anyway.
4. Added `training.py`, not in the plan's tree, so central training (E0) and FL clients
   (Phase 2) share one training loop. *Cost if wrong:* moving two functions.
5. Added `experiments/run_central.py` (E0) and `tests/test_config.py` / `tests/test_models.py`,
   beyond the plan's listed files. *Cost if wrong:* none functionally.
6. Key handling decided now, implemented in Phase 4: `SYBIL_FL_KEY` env var, else a fresh
   random key per run, never logged (D4). *Cost if wrong:* a change in one function.
7. One git repository at the `Project/` level (branch `main`) versions the plan, the docs and the code
   together. Local config: author Aswin, `core.autocrlf=false`, `core.fileMode=false`; `.gitattributes`
   stores text with LF endings. Generated results, datasets and `.venv` stay out of git.
   *Cost if wrong:* none.

## Phase 2 — Clean FL baseline (done, 2026-10-19)

Built: `client.py` (`local_update` → `ClientUpdate` with the original delta), `aggregators.py`
(sample-count-weighted `fedavg`, explicit input checks), `server.py` (`run_fl` round loop, per-round
CSV), `fl` config section (`rounds`, `local_epochs`), `configs/smoke.yaml`, `utils.environment_info`,
`experiments/run_main.py`.

Tests: `pytest` → **53 passed** (29 earlier + 24 new: aggregators 10, client 5, server 5, config 4).
The new tests were run and seen failing (modules / config section missing) before implementation.
Key checks: FedAvg of `[1,2]`, `[3,4]` with weights 1, 3 is exactly `[2.5, 3.5]`; one server round
equals the global weights plus the FedAvg of independently computed client deltas; a repeated
run gives identical weights.

Results (Windows, environment as in Phase 1, seed 42, Dirichlet α=0.5, 20 clients, all every round):

| Run | Result |
|---|---|
| Smoke (`configs/smoke.yaml`, 3 rounds) | test acc 0.1018 → 0.8713 → 0.9467 → 0.9595 |
| Full (`default.yaml`, 20 rounds, 1 local epoch) `results/fl_20261019T124156Z/` | 0.9595 at round 3 (same as smoke), 0.9820 at round 10, **0.9875** at round 20; ~6.3 s per round |
| E0 re-run after the `environment_info` refactor | unchanged: 0.1018 → 0.9835 → 0.9859 |

The FL run beats E0's 0.9859, but it saw 20 passes over the data against E0's 2, so this is not
evidence that federation helps; it shows only that clean FedAvg learns under this non-IID split.
Single seed so far; mean ± std across seeds comes with the Phase 6 experiments.

**Exit criteria met:** clean FL learns; aggregation matches the hand-checkable toy case.

### Rulings made in Phase 2

1. Every client takes part in every round; no subsampling option (approved before implementation).
   *Cost if wrong:* add a `clients_per_round` field and a seeded selection step in `run_fl`.
2. Plain FedAvg with server learning rate 1 (global += weighted mean delta); no server optimizer.
   *Cost if wrong:* one line in `run_fl`.
3. A fresh SGD optimizer per client per round, so momentum (0 by default) never carries over.
   *Cost if wrong:* clients would need persistent optimizer state.
4. `fedavg` raises on an empty update set instead of falling back; the configured fallback for
   "every client flagged" belongs to Phase 4, when filtering exists. *Cost if wrong:* none yet.
5. `round_seconds` times client training + aggregation, excluding test evaluation.
6. Shared synthetic test data moved to `tests/helpers.py` (not `conftest.py`, which pytest
   advises against importing). `run_fl` gained an optional `on_round` callback for progress
   printing, beyond the approved signature. *Cost if wrong:* none functionally.

## Phase 3 — Label flipping and attack success rate (built, 2026-10-19; awaiting review)

Built: `attacks.py` (`select_sybils`, `label_flip_labels`, `prepare_attack` → `AttackPlan`),
`metrics.py` (`attack_success_rate`), `attack` config section (`type`, `num_sybils`,
`source_class`, `target_class`), `configs/label_flip.yaml`; `run_fl` trains Sybils on the
poisoned labels and logs an `asr` column every round (clean runs too); `run_main.py` records the
Sybil ids and per-Sybil flip counts in `meta.json` and names run folders `fl_<attack>_<timestamp>`.

Tests: `pytest` → **80 passed** (53 earlier + 27 new: attacks 10, metrics 4, server 4, config 9
counting parametrised cases). The new tests were run and seen failing before implementation. Key checks: only Sybils' source-class labels change; ASR
matches a hand calculation with a stub classifier; an attacked round equals FedAvg of
independently computed deltas with Sybils on poisoned labels; `run_fl` refuses a label-flip
config without its plan (and vice versa), so a config cannot silently run clean.

Results (Windows, seed 42, Dirichlet α=0.5, 20 clients, 20 rounds, 7 → 1):

| Run | Final test acc | Final ASR | Max ASR (rounds 1–20) |
|---|---|---|---|
| Clean `results/fl_none_20261019T132154Z/` (accuracies identical to Phase 2) | 0.9875 | 0.0019 | 0.0058 |
| 4 Sybils `results/fl_label_flip_20261019T132356Z/` | 0.9872 | 0.0019 | 0.0088 |

Sybils were clients 0, 3, 5, 6, holding 174 / 69 / 104 / **0** sevens: 347 of the 6,265 training
sevens (5.5%). Client 6 flipped nothing, so it behaves exactly like an honest client.

**Finding: at this setting the attack has no measurable effect.** ASR stays at the clean
baseline (≈ 2 of 1,028 test sevens predicted as 1) and accuracy differs by 0.0003, on one seed.
The honest clients' correctly labelled sevens outweigh the flipped ones under FedAvg. This does
not show that label flipping is harmless in general; it shows that randomly chosen Sybils on
their own Dirichlet shards carry too little source-class data here to move the model.

**Exit criteria:** attack code and metrics pass unit tests and outcomes are logged (met).
"Confirm the attack has measurable effects" (plan §12) is **not met**; how to strengthen the
threat model is open for review before Phase 4, because a Sybil that flips nothing is not a
meaningful detection target.

### Phase 3 diagnostic: what makes the attack bite

Five extra 20-round runs, seed 42, each `label_flip.yaml` or `default.yaml` with only
`data.partition` and/or `attack.num_sybils` changed (full configs in each run's `meta.json`).
Sybil selection does not depend on the partition, so the 8-Sybil runs use the same clients.

| Split | Sybils | Sevens held by Sybils | Final acc | Final ASR | ASR over rounds 1–20 |
|---|---|---|---|---|---|
| Dirichlet | 0 | — | 0.9875 | 0.0019 | ≤ 0.006 |
| Dirichlet | 4 | 347 (5.5%) | 0.9872 | 0.0019 | ≤ 0.009 |
| Dirichlet | 8 | 1,999 (31.9%) | 0.9652 | 0.1975 | 0.14–0.72, oscillating |
| Dirichlet | 10 | 5,169 (82.5%) | 0.8865 | 0.9757 | 0.80 → 0.98 |
| IID | 0 | — | 0.9882 | 0.0039 | ≤ 0.007 |
| IID | 4 | 1,240 (19.8%) | 0.9859 | 0.0107 | 0.01–0.02 |
| IID | 8 | 2,524 (40.3%) | 0.9756 | 0.1021 | 0.06–0.14 |

Reading (one seed, so indicative only): ASR tracks the **share of source-class training samples
the Sybils control**, not the Sybil count. Under the Dirichlet split that share is decided by
which clients happen to be chosen, so attack strength varies wildly between configurations and
would vary between seeds too. Run folders under `results/`: Dirichlet 8 / 10 Sybils
`fl_label_flip_20261019T133050Z` / `T133258Z`; IID clean `fl_none_20261019T133454Z`;
IID 4 / 8 Sybils `fl_label_flip_20261019T133655Z` / `T133928Z`.

### Phase 3 revision: attacker-owned pool (DECISIONS D9)

Built: `data.reserve_attacker_pool` and `data.make_federated_split` (pool withheld in every run);
`attack.sybil_data` (`attacker_pool` | `own_shard`) and `attack.samples_per_sybil`;
`data.attacker_pool_size`; `AttackPlan.client_indices` (the full client roster, which `run_fl`
now trains instead of the honest partition when an attack is given). Defaults: pool 1,000
sevens, 4 Sybils joining 20 honest clients (24 total), 500 pool images each.

Tests: `pytest` → **97 passed** (80 + 17 new: data 5, attacks 5, server 1, config 6), all seen
failing first. Regression: `own_shard` with `attacker_pool_size: 0` reproduces the committed
Phase 3 rounds 1–3 exactly (0.8702 / 0.9444 / 0.9577, ASR 0.0088 / 0.0078 / 0.0068).

Results (20 rounds, seed 42; honest clients identical in both runs):

| Run | Final acc | Final ASR | ASR over rounds 1–20 |
|---|---|---|---|
| Clean, pool withheld `results/fl_none_20261019T163030Z/` | 0.9867 | 0.0029 | 0.003–0.007 |
| 4 pool Sybils `results/fl_label_flip_20261019T163219Z/` | 0.9845 | 0.0175 | 0.007–0.028 |

Sybils were clients 0, 4, 6, 7 of 24, each flipping all 500 of its images. The attack now has a
consistent, measurable effect (ASR about 3–6× the clean run in every round from round 3 on), but
it is small: the Sybils hold 2,000 of ~61,000 training samples, so FedAvg gives them ~3% of the
weight against 5,265 correctly labelled honest sevens. Strength is now set by `num_sybils` and
`samples_per_sybil` rather than by which clients were chosen. One seed.

### Phase 3 attack-strength sweep (2026-10-20)

`label_flip.yaml` with only `attack.num_sybils` and `attack.samples_per_sybil` changed; seed 42,
20 rounds, same 20 honest clients and pool in every run. "Sybil weight" = Sybil samples / all
training samples, i.e. their share of FedAvg. ASR is summarised over rounds 11–20 because it
swings strongly between rounds; a single final-round value is misleading (the 10 × 1,000 run read
ASR 0.92 at round 17 and 0.012 at round 20).

| Sybils × images | Clients | Sybil weight | Final acc | ASR r11–20 mean | min – max |
|---|---|---|---|---|---|
| clean | 20 | 0% | 0.9867 | 0.004 | 0.003 – 0.005 |
| 4 × 500 | 24 | 3.3% | 0.9845 | 0.017 | 0.011 – 0.028 |
| 4 × 1,000 | 24 | 6.3% | 0.9820 | 0.108 | 0.038 – 0.202 |
| 8 × 500 | 28 | 6.3% | 0.9784 | 0.120 | 0.057 – 0.196 |
| 10 × 500 | 30 | 7.8% | 0.9853 | 0.245 | 0.012 – 0.732 |
| 8 × 1,000 | 28 | 11.9% | 0.8964 | 0.449 | 0.010 – 0.918 |
| 10 × 1,000 | 30 | 14.5% | 0.9847 | 0.470 | 0.012 – 0.945 |

Reading (one seed): mean ASR rises with the Sybils' share of FedAvg weight; 4 × 1,000 and
8 × 500 have the same share and similar ASR. Above ~8% weight the global model swings between
rounds dominated by honest updates and rounds dominated by Sybils, so ASR and accuracy oscillate.
At 1,000 images every Sybil holds the entire pool, so their data is identical (the easiest case
to detect). Runs: `results/fl_label_flip_20261020T065249Z`, `T065504Z`, `T065712Z`, `T070019Z`,
`T070238Z`.

### Rulings made in Phase 3

1. Flip 7 → 1, 4 of 20 Sybils, Sybils keep their own Dirichlet shards, no backdoor (approved).
2. `attack.type: none` requires `num_sybils: 0` and an attack requires `>= 1`, so a config
   cannot be ambiguous about whether Sybils exist. *Cost if wrong:* one validation rule.
3. Sybils are chosen uniformly at random from their own seed stage `"sybil_selection"`, so
   changing the attack never shifts the partition or model init. *Cost if wrong:* none.
4. `mean_train_loss` averages all clients, Sybils included (their loss is on poisoned labels).
5. Run folders renamed from `fl_<timestamp>` to `fl_<attack>_<timestamp>`; the Phase 2 folder
   keeps its old name.
6. Attacker pool adopted after review (D9, "try C"): pool of source-class images only, Sybils as
   extra clients at random positions, own-shard mode kept for reproducibility. E0
   (`run_central.py`) still trains on all 60,000 images; it is only a pipeline check.
   *Cost if wrong:* see D9.

**Phase 3 closed (2026-10-20):** attack default set to **8 Sybils × 500 pool images** (Sybil
weight 6.3%; ASR rounds 11–20 mean 0.120, range 0.057–0.196 vs clean ≤ 0.005), the smallest
setting in the sweep with a clear, steady effect and non-identical Sybil data. Summary metrics
are reported as means over the last half of the rounds because ASR oscillates (see sweep).

---

# One-day completion push (2026-10-20)

Plan: `docs/EXECUTION_PLAN.md` (validated by the author: code + results now, report and
slides later; cryptography is the priority; no per-phase review stops, a commit per step).

## Step 0 — Cryptographic core

Built:
- `crypto.py`: HKDF-SHA256 (RFC 5869) on the standard-library HMAC; `derive_round_key(K, t,
  purpose)` with salt `sybil-fl|hkdf-salt|v1` and info `sybil-fl|<purpose>|v1|round=<t>`;
  `load_server_key()` (`SYBIL_FL_KEY` = 64 hex chars, else 32 fresh bytes from `secrets`;
  returns only a source label for logging; error messages never echo the value).
- `projection.py`: `projection_matrix(round_key, k, d, generator)` — dense ±1/√k matrix whose
  sign bits are the keyed SHAKE-256 stream of `round_key | "|sybil-fl|projection-matrix|v1|k=..|d=.."`
  (bit i·d+j, MSB first, + for 0); `generator="pcg64"` gives the same distribution from NumPy
  PCG64 for cost comparison only. `project(vectors, R)`.
- `projection` config section (`k: 64`, `generator: shake256`).

Verification: RFC 5869 test cases 1 and 3 (PRK and OKM) were first confirmed against the
independent `cryptography` 46.0.6 library, then hard-coded in `tests/test_crypto.py`. The FIPS 202
SHAKE-256("") vector was checked once against Python's `hashlib` during that cross-check but is
**not** in the tests; instead `tests/test_projection.py` pins the matrix's bit layout to
`hashlib.shake_256` output.
`pytest` → **119 passed** (97 + 22: crypto 10, projection 9, config 3), all seen failing first.

Rulings:
1. Rademacher (±1) entries instead of Gaussian: they come straight from stream bits (no
   float transform of random bits to get right), have variance 1/k after scaling, and satisfy
   the Johnson–Lindenstrauss property (Achlioptas 2003). *Cost if wrong:* swap the bit→value
   map in one function.
2. Prefix-keyed SHAKE-256 instead of KMAC256: `hashlib` has no KMAC; key-first SHAKE is the
   sponge-PRF construction KMAC standardises (KMAC adds encoding/padding). The composition
   HKDF → keyed SHAKE is ours and not externally reviewed; the report must say so.
3. Matrix shape is part of the SHAKE input so matrices of different (k, d) are unrelated
   (tested: a k=8 matrix is not a prefix of the k=16 one).

## Step 1 — Crypto design recorded

DECISIONS D3 rewritten (HKDF-SHA256 round keys → keyed SHAKE-256 sign stream → ±1/√k matrix;
what is and is not claimed; PCG64 kept only as a cost baseline) and D4 (key handling, including
the no-key-in-outputs test). Test `tests/test_experiment.py::test_no_key_material_reaches_any_output_file`
runs an experiment with the fake key and scans every output file for `K` and every round key,
raw and hex (upper and lower case).

## Steps 2–3 — Detector, metrics, median baseline, evasion, server integration

Built:
- `detector.py` (D11): normalised copies → optional projection → cosine → average-linkage
  (scipy) cut at τ → flag clusters ≥ 3; degenerate vectors never flagged; per-stage timings.
- `metrics.py`: `confusion`, `detection_scores` (NaN for undefined ratios; micro-averaging).
- `aggregators.py`: `coordinate_median` (exact median via quantile; even counts average the
  middle two) and `aggregate(name, ...)`.
- `attacks.py`: `evasion_perturbations` (known/guessed projection: minimum-norm row-space
  perturbation; Gaussian) with exact norm budget; `attacker_guess_key` (D10).
- `evaluation.py`: shadow scoring per (method, k, τ) and projection distortion of pairwise
  cosine (E1), from the same submitted updates.
- `server.run_fl`: evasion (attacker side) → non-finite rejection → detector → configured
  aggregator over unflagged → skip-round fallback; per-round detection counts and timings;
  shadow CSVs. Requires the master key when the projection is used; never writes it.
- `experiment.py` (`run_experiment`, `summarize`); `run_main.py` thin; `run_suite.py`
  (groups tune/e2/e3/e6, parallel workers, resumable); `tune_thresholds.py`;
  `run_scalability.py` (E5); `summarize.py`; `make_plots.py`.
- Config sections `defence`, `evaluation`, `attack.evasion*`; matplotlib added as the optional
  `plots` extra in `pyproject.toml` (needed only by `make_plots.py`).

Tests: `pytest` → **166 passed**, all new tests seen failing first. Two test fixes during the
step, both in the tests: (1) the evasion test first used the d = 46,730 gain (≈ 27) for a
d = 4,096 check where the gain is 8; the budget was raised so the expected cosines follow
from the formula; (2) the defence-equivalence test needed τ = 0.1 because on synthetic IID
data at round 1 honest clients have cosine ≈ 0.9 with each other (Sybils ≈ 0.99).

MNIST smoke run (3 rounds, projected detector at the placeholder τ = 0.5): 26–27 of 28 clients
flagged — the placeholder is far too loose; Sybil–Sybil cosine 0.997 vs honest–honest 0.49.

## Step 4 — Threshold tuning (seed 0 only)

Runs: `results/suite/tune/{clean_s0, attack_s0}` (FedAvg, shadow scoring at
k ∈ {16, 32, 64, 128, 256} and τ ∈ {0.02 … 0.5}). Rule (D11): max F1 on the attacked run with
FPR ≤ 5% on both runs; ties → larger τ. Result (`configs/tuned_thresholds.yaml`):

| Variant | τ | perfect-F1 plateau on seed 0 | first τ with false positives |
|---|---|---|---|
| full-dimensional | 0.25 | 0.05 – 0.25 | 0.30 |
| projected k = 16 | 0.12 | 0.08 – 0.12 | 0.15 |
| projected k = 32 | 0.20 | — | — |
| projected k = 64 | 0.15 | 0.08 – 0.15 | 0.20 |
| projected k = 128 | 0.20 | — | — |
| projected k = 256 | 0.20 | — | — |

(k = 32/128/256 plateaus are in `results/suite/tune/tuning_table.csv`.) All variants reach
F1 = 1.000 with FPR 0 on seed 0: the 8 pool Sybils are near-duplicates (cosine ≈ 0.997), so the
unevaded attack is easy; the plateau narrows as k shrinks (projection noise adds spread to the
honest clients' similarities). E6 is where the methods are expected to separate.

## Steps 5–6 — Experiments (E2, E3, E6 suite; E5 benchmark)

Runs: `run_suite.py` groups e2 (24 runs, seeds 1–3), e3 (8 runs, α ∈ {0.1, 5.0}, seeds 1–2),
e6 (42 runs: 3 attackers × 7 budgets × seeds 1–2), 6 workers × 2 torch threads, ~90 min in
total, no failed runs. Each run used its own freshly generated server key (`key_source:
generated` in every meta.json). E5 (`run_scalability.py`, 4 threads) ran afterwards on the idle
machine.

Ruling during E5: the first E5 run showed PCG64 generation *slower* than SHAKE-256, but the PCG64
path drew one byte per bit (`integers(0, 2)`), an unfair baseline. It now draws bytes and uses
the same byte → bit unpacking as the SHAKE path (`projection.py`); no FL experiment used the
PCG64 path, so only E5 was re-run. With the fair baseline the two generators cost the same
within ~12% at d = 46,730. *Cost if wrong:* none for the defence; E5's generator comparison.

## Step 7 — Summaries, figures, findings

`summarize.py` → `results/summary/` (6 CSV tables + `tables.md`); `make_plots.py` → 7 figures
in `plots/`, rendered and inspected (end-label collisions fixed: coinciding lines share one
label; overlapping labels form stacks centred on their true values; round 0 dropped from the
per-round plots; E6 legend moved below the panels). Palette checked with the dataviz validator
(4 slots pass all hard gates; aqua/yellow below 3:1 contrast → markers, direct labels and the
tables as relief). Findings written up in `docs/RESULTS.md`.

Headline numbers (details in RESULTS.md):
- E6: with the projection key leaked, ε = 0.02 evades the projected detector (recall 0, ASR back
  to 0.19 ≈ undefended) while the full-dim detector still catches every Sybil; with the key
  secret, the guessing attacker behaves like Gaussian noise and is fully detected up to ε = 0.2,
  only partly evading at ε = 0.5 (recall 0.08–0.12).
- E2: both detectors reduce ASR from 0.18 to 0.004 with no accuracy cost (projected: F1 1.000,
  FPR 0; full: F1 0.989, FPR 0.009); the coordinate median only reaches ASR 0.075 and costs
  accuracy.
- E1: cosine error ∝ 1/√k (mean 0.090, p95 0.222 at k = 64); detection unaffected down to k = 16.
- E3: false positives rise as data becomes more IID (α = 5: 9.1% full, 3.9% projected).
- E5: projection pays off above n ≈ 100–200 (5× at n = 1,000); stored representations 730×
  smaller; matrix generation ~15–18 ms per round.

## Step 8 — Final check

`pytest` → **166 passed**. Summary tables and figures regenerated from saved runs and committed
(D12); raw runs stay git-ignored. Key-material scan of everything committed: no master or round
keys anywhere (keys were generated per run and never written). A scan of all 66 tracked and
new text files for 64-hex-character strings found only the two RFC 5869 PRK test vectors in
`tests/test_crypto.py`.
README and EXECUTION_PLAN status updated.

## What is left (for the author)

Report and slides, written from `docs/RESULTS.md` (findings, limitations, figure index,
references to verify), `docs/DECISIONS.md` (design and security claims) and this log. Optional
future work listed in `docs/EXECUTION_PLAN.md` (Multi-Krum, backdoor, sparse projection,
reputation, Sybil-fraction sweep, Fashion-MNIST, stronger adaptive attacker in E6).
