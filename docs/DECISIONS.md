# Design decisions (Phase 0 freeze)

Each decision lists what was chosen, why, and what changes if it turns out wrong.
Revise this file when a decision changes; never change behaviour silently.

## D1. Research claim

We test (not promise) the following:

> A secret-seeded, low-dimensional random projection of client updates
> preserves enough pairwise cosine-similarity structure for a clustering-based
> detector to flag colluding Sybils with detection quality close to the same
> detector run on full-dimensional updates, while reducing the cost of the
> pairwise-similarity step and the size of stored detection representations.

Every result is reported against the **full-dimensional counterpart**
(identical detector logic, no projection). Runtime claims always include the
projection cost, `O(n·d·k)`, not just the `O(n²·k)` similarity step.

## D2. Threat model (summary — full version in plan §3)

- One trusted server holds a secret key `K` and sees every individual update
  (no secure aggregation).
- One adversary controls a fraction of the `n` simulated clients. It knows the
  algorithm and all public configuration, but not `K` and not the current
  round's projection before it submits its updates.
- Client identities are stable across rounds; Sybil creation/churn is not solved.
- The detector is a similarity **heuristic**. Similar ≠ malicious; honest
  non-IID clients can look alike, so false-positive rate is a first-class metric.
- Out of scope: transport security (TLS), secure aggregation, DP, HE, formal proofs.

## D3. Projection randomness — cryptographic stream (revised 2026-10-20)

Originally (Phase 0) the plan's "course prototype" path: `HMAC-SHA256(K, label)` seeding
NumPy PCG64. **Revised**, because the author made cryptography the project's priority, to the
plan §3 second path: a cryptographically secure deterministic stream built from standard
primitives, with domain separation, key handling and separate tests.

Construction, per round `t` (`crypto.py`, `projection.py`):

1. Round key: `K_t = HKDF-SHA256(IKM = K, salt = "sybil-fl|hkdf-salt|v1",
   info = "sybil-fl|projection|v1|round=<t>", L = 32)` (RFC 5869; extract then expand).
   The info string is the domain separator (protocol, purpose, version, round); purposes may
   not contain `|`, so labels cannot be made to collide.
2. Sign stream: `SHAKE-256(K_t || "|sybil-fl|projection-matrix|v1|k=<k>|d=<d>")` (FIPS 202),
   read for `k·d` bits. The shape is in the input, so matrices of different shapes are unrelated.
3. Matrix: `R_t[i, j] = (+1 if bit i·d+j is 0 else −1) / √k` (dense Rademacher; variance 1/k;
   Johnson–Lindenstrauss property as for Gaussian entries).

What we **do** claim, under standard assumptions (HMAC-SHA256 is a PRF; keyed SHAKE-256 is a
PRF): without `K`, each `R_t` is computationally indistinguishable from a uniformly random ±1
matrix, independent across rounds and purposes; `R_t` is reproducible from `K` and `t`.

What we **do not** claim: that the detector resists adaptive attackers in general, or any
formal security proof. Unpredictability of `R_t` only removes one attack route (crafting
updates against a known projection); E6 measures how much that matters. The HKDF → keyed SHAKE
composition is ours and has not been externally reviewed; KMAC256 (SP 800-185) is the
standardised keyed XOF but is not in Python's `hashlib`.

Kept for comparison: `generator: pcg64` produces the same sign distribution from
`PCG64(SeedSequence(K_t))`. PCG64 is not a CSPRNG (its state can be recovered from outputs), so
it is used only to measure the cost of the cryptographic stream (E1/E5), never as the defence.

If wrong (e.g. a reviewer requires a standardised keyed XOF): replace step 2 with KMAC256 from
a reviewed library; only `projection.projection_matrix` changes.

## D4. Key handling (revised 2026-10-20)

- The master key `K` (32 bytes) is read at runtime from `SYBIL_FL_KEY` (64 hex characters) or,
  if absent, generated fresh with `secrets.token_bytes(32)` (the OS CSPRNG) for that run.
  Malformed values are rejected with messages that do not repeat the value.
- `K` and round keys are never written to results, logs or configs. Run metadata records only
  `key_source` (`env` / `generated`). A test runs an experiment and scans every output file
  for the key in hex and raw form.
- Per-round keys come from HKDF (D3), so learning one round's `K_t` or `R_t` reveals nothing
  about other rounds' (PRF assumption); compromise of `K` reveals all rounds.
- Experiments with a generated key are not bit-reproducible across runs (the projection
  differs); results are compared statistically. Supplying `SYBIL_FL_KEY` makes a run exactly
  reproducible.
- Tests use the obviously fake key `bytes(range(32))`.

## D5. Model: small CNN, ~46.7k parameters, no BatchNorm

`conv(1→16, 5×5) → ReLU → maxpool 2 → conv(16→32, 5×5) → ReLU → maxpool 2 →
fc(512→64) → ReLU → fc(64→10)`; parameter count **d = 46,730**.

- Small `d` keeps CPU experiments fast and a dense `k × d` projection at
  `k = 512` around 96 MB (float32), so chunking is optional rather than required.
- No BatchNorm: the update `Δ` is exactly the trainable parameters, with no
  running-statistics buffers to aggregate or exclude.

If wrong (need a larger `d` for the runtime study): the synthetic benchmark
(E5) varies `d` independently of the model, so the scaling claim does not
depend on this CNN.

## D6. Data handling

- MNIST is loaded once into in-memory float tensors (normalised with mean
  0.1307, std 0.3081); clients hold **index arrays** into that tensor. This
  makes partitions trivially checkable (disjoint + exact coverage) and avoids
  DataLoader worker nondeterminism.
- Partitions: IID (random equal split) and Dirichlet(α) label skew, with a
  configured minimum client size enforced by bounded resampling.

## D7. Determinism

- One master `seed` in config. Sub-streams (partitioning, model init, client
  minibatch order) are derived from it, so changing one stage does not shift
  the others.
- Minibatch order uses an explicit `torch.Generator`, not global state.

## D8. Environment

- Python ≥ 3.10; dependencies in `pyproject.toml`. Exact versions used for
  each recorded result are written into that run's metadata.
- Versions the Phase 1 tests were run with are listed in `docs/PROGRESS.md`.

## D9. Sybil data: an attacker-owned pool (added in Phase 3 review)

Chosen: Sybils are **extra identities run by one attacker**, not honest clients turned bad.

- Before partitioning, `data.attacker_pool_size` random training images of
  `attack.source_class` are withheld; honest clients partition the rest. The pool is withheld
  in **every** run, clean ones included, so a clean run and an attacked run have identical
  honest clients and the only difference is the Sybils joining.
- An attack adds `num_sybils` clients at seeded random positions in the client list (so client
  order carries no information). Each draws its own `samples_per_sybil` pool images without
  replacement (seed stage `sybil_<id>_pool_sample`; Sybils' samples may overlap) and trains on
  them with every label set to `target_class`. Sybils report their true sample count to FedAvg.
- This is the plan's realism-ladder level 2 (shared malicious objective, independent data and
  minibatch order) and the usual colluding-Sybil setting in the literature.

Why: with Sybils on their own Dirichlet shards (`sybil_data: own_shard`, kept as the Phase 3
baseline), attack strength depended on which clients were chosen: 4 Sybils held 5.5% of the
sevens and ASR stayed at the clean baseline, 10 held 82.5% and ASR reached 0.98 (PROGRESS,
Phase 3 diagnostic). The pool makes strength a configured quantity.

Costs and limits:
- Withholding the pool changes the honest data, so Phase 2 numbers do not carry over; the clean
  baseline is re-run with the pool withheld. With `attacker_pool_size: 0` the split is
  identical to the Phase 2–3 split (tested).
- The pool holds only source-class images, so a Sybil's local data is a single relabelled
  class. That is a strong, easily clustered attack; a mixed-class pool (harder to detect) is a
  possible later option, not built.
- Sybils do not inflate their reported sample counts; that attack is out of scope.

If wrong: switch `sybil_data` back to `own_shard` (no code change), or change the pool's
composition inside `data.reserve_attacker_pool` / `attacks.prepare_attack` only.

## D10. E6 attacker model: what the key protects against (added 2026-10-20)

E6 measures the security value of keeping `R_t` secret. After local training, each Sybil adds
a perturbation `p_i` to the update it submits, with `||p_i|| = ε·||δ_i||` exactly (ε is the
"evasion budget", reported relative to the update norm as plan §6 requires). The server
aggregates the perturbed update, so a large ε also dilutes the Sybil's poisoning. Attackers:

- **known_projection** (key compromised / projection public): `p_i = R_tᵀ (R_t R_tᵀ)⁻¹ v_i`, the
  minimum-norm change whose projection is a random k-dim direction `v_i`. Projection magnifies
  it by about `√(d/k)` (≈ 27 at d = 46,730, k = 64), so a small ε scrambles the Sybils'
  projected similarity while their full-dimensional updates stay almost identical.
- **guessed_projection** (secret key, honest model of a smart attacker): the same construction
  with the attacker's own random matrix (`attacks.attacker_guess_key`). Without `K` its matrix
  is unrelated to `R_t` (D3), so the perturbation acts like random noise under `R_t`.
- **gaussian**: isotropic noise, the plan's baseline evasion.

The full-dimensional detector is scored in shadow on the same perturbed updates, isolating the
effect of the projection. Assumptions: the attacker knows the algorithm, `k`, `d` and the
threshold; the server reveals no flags or scores (no feedback oracle across rounds); a fresh
`R_t` each round means nothing learned in one round transfers to the next. Not modelled: an
attacker that learns from detection feedback, or that optimises perturbations against the
clustering rule directly. Expected (hypothesis, measured in E6): known-projection evasion
defeats the projected detector at ε of a few percent with little loss of attack strength;
guessed and Gaussian need ε large enough to also break full-dimensional similarity.

## D11. Detector design and threshold selection (added 2026-10-20)

- Normalised copies → (optional) secret projection → normalise → cosine similarity →
  average-linkage clustering on cosine distance (scipy), cut at τ → flag every cluster of
  ≥ `min_cluster_size` = 3 clients → FedAvg over unflagged submitted updates; if all are
  flagged the round is skipped. The full-dimensional counterpart is the same code without the
  projection step (not FoolsGold; plan §5).
- Degenerate vectors (zero norm, non-finite, near-zero projection) are never clustered or
  flagged; non-finite updates are rejected from aggregation and counted.
- τ is chosen per detector variant on seed 0 only (`experiments/tune_thresholds.py`): max F1 on
  the attacked run subject to FPR ≤ 5% on both the clean and attacked runs, ties → larger τ.
  Results are reported on held-out seeds 1–3 (E2) or 1–2 (E3, E6). Tuned values are in
  `configs/tuned_thresholds.yaml`.
- Detection scores are micro-averaged over a run's rounds (summed counts), which keeps them
  defined when a round has no flags or no Sybils; undefined ratios are NaN, never 0 or 1.

## D12. Committed results (added 2026-10-20)

Raw run folders (`results/suite/`) stay git-ignored and are regenerated by the scripts. The
generated summary tables (`results/summary/`) and figures (`plots/`) are committed, because
the report will be written from them later; they are always produced by
`experiments/summarize.py` and `experiments/make_plots.py`, never edited by hand.
