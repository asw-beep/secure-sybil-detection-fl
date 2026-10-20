"""Figures for the report, generated only from saved runs (results/suite/) and summaries.

Style: light print figures; categorical colours in fixed order (blue, orange, aqua, yellow;
validated with the dataviz palette checker: adjacent CVD dE >= 9.1). Aqua and yellow are below
3:1 contrast on the surface, so every line also carries a distinct marker and a direct end label,
and every figure has a table twin in results/summary/tables.md. Reference series (clean
baseline, full-dimensional detector) are drawn in neutral grey. One y-axis per panel.

Usage:  python experiments/make_plots.py
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from sybil_fl.metrics import detection_scores  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SUITE = REPO / "results" / "suite"
PLOTS = REPO / "plots"

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MARKERS = ["o", "s", "^", "D"]
REFERENCE = "#898781"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 11,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "legend.labelcolor": INK2, "lines.linewidth": 1.8, "lines.markersize": 5.5,
})


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def meta(run: Path) -> dict:
    return json.loads((run / "meta.json").read_text())


def shadow(run: Path, method: str, k: int, tau: float) -> dict[str, float]:
    tot = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for r in read_csv(run / "shadow_detection.csv"):
        if r["method"] == method and int(r["k"]) == k and math.isclose(float(r["tau"]), tau):
            for c in tot:
                tot[c] += int(r[c])
    return detection_scores(**tot)


def curve(runs: list[Path], col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.array([[float(r[col]) for r in read_csv(run / "rounds.csv")] for run in runs])
    return np.arange(data.shape[1]), data.mean(axis=0), data.std(axis=0)


def end_label(ax, x, y, text, color=INK2):
    ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8.5, color=color)


def end_labels(ax, x, labels: list[tuple[float, str]]) -> None:
    """Direct labels at the line ends without collisions: coinciding lines share one label, and
    labels that would overlap form a stack centred on the mean of their true end values."""
    lo, hi = ax.get_ylim()
    gap = 0.045 * (hi - lo)
    groups: list[list] = []
    for y, text in sorted(labels):
        if groups and abs(y - groups[-1][0]) < 0.25 * gap:
            groups[-1][1].append(text)
        else:
            groups.append([y, [text]])
    blocks: list[list] = []  # each block: list of (true y, text); placed as an evenly spaced stack
    for y, texts in groups:
        blocks.append([(y, " / ".join(texts))])
        while len(blocks) > 1:
            prev, cur = blocks[-2], blocks[-1]
            centre = lambda b: np.mean([v for v, _ in b])
            top_prev = centre(prev) + (len(prev) - 1) / 2 * gap
            bottom_cur = centre(cur) - (len(cur) - 1) / 2 * gap
            if bottom_cur - top_prev >= gap:
                break
            blocks[-2:] = [prev + cur]
    for block in blocks:
        c = np.mean([v for v, _ in block])
        for i, (_, text) in enumerate(block):
            y = c + (i - (len(block) - 1) / 2) * gap
            ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=8.5, color=INK2)


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(PLOTS / name, dpi=160)
    plt.close(fig)
    print(f"saved plots/{name}")


def fig_rounds(col: str, ylabel: str, title: str, name: str) -> None:
    # Rounds 1-20: round 0 is the untrained model and only stretches the axis.
    e2 = SUITE / "e2"
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    labels = []
    x, m, _ = curve([e2 / f"clean_fedavg_s{s}" for s in (1, 2, 3)], col)
    ax.plot(x[1:], m[1:], color=REFERENCE, linewidth=1.4, label="no attack, FedAvg")
    labels.append((m[-1], "no attack"))
    series = [("fedavg", "FedAvg"), ("median", "Coordinate median"), ("full", "Full-dim detector"),
              ("projected", "Projected detector (k=64)")]
    for i, (key, label) in enumerate(series):
        x, m, s = curve([e2 / f"attack_{key}_s{s_}" for s_ in (1, 2, 3)], col)
        ax.fill_between(x[1:], (m - s)[1:], (m + s)[1:], color=SERIES[i], alpha=0.10, linewidth=0)
        ax.plot(x[1:], m[1:], color=SERIES[i], marker=MARKERS[i], markevery=(3, 4), label=label)
        labels.append((m[-1], label.split(" (")[0].replace(" detector", "")))
    ax.set_xlabel("round")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    ax.set_xlim(1, x[-1] + 6)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.legend(loc="best", fontsize=8.5)
    end_labels(ax, x[-1], labels)
    save(fig, name)


def fig_detection_vs_k(tuned: dict) -> None:
    e2 = SUITE / "e2"
    ks = sorted(int(n.split("_k")[1]) for n in tuned if n.startswith("projected_k"))
    f1, fpr = [], []
    for k in ks:
        tau = tuned[f"projected_k{k}"]
        f1.append([shadow(e2 / f"attack_fedavg_s{s}", "projected", k, tau)["f1"] for s in (1, 2, 3)])
        fpr.append([shadow(e2 / f"clean_fedavg_s{s}", "projected", k, tau)["fpr"] for s in (1, 2, 3)])
    full_f1 = np.mean([shadow(e2 / f"attack_fedavg_s{s}", "full", 0, tuned["full"])["f1"] for s in (1, 2, 3)])
    full_fpr = np.mean([shadow(e2 / f"clean_fedavg_s{s}", "full", 0, tuned["full"])["fpr"] for s in (1, 2, 3)])
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), sharex=True)
    for ax, vals, ref, title in ((axes[0], f1, full_f1, "Detection F1 (attacked runs)"),
                                 (axes[1], fpr, full_fpr, "False-positive rate (clean runs)")):
        m, s = np.mean(vals, axis=1), np.std(vals, axis=1)
        ax.axhline(ref, color=REFERENCE, linewidth=1.2)
        ax.annotate("full-dim", (ks[-1], ref), xytext=(0, 5), textcoords="offset points", ha="right",
                    fontsize=8.5, color=INK2)
        ax.errorbar(ks, m, yerr=s, color=SERIES[0], marker="o", capsize=3, label="projected")
        ax.set_xscale("log", base=2)
        ax.set_xticks(ks, [str(k) for k in ks])
        ax.set_xlabel("projected dimension k")
        ax.set_title(title, loc="left")
    axes[0].set_ylim(0, 1.05)
    axes[1].set_ylim(0, max(0.1, axes[1].get_ylim()[1]))
    fig.suptitle("Held-out seeds 1–3, tuned thresholds, shadow scoring", x=0.01, ha="left", fontsize=9, color=INK2)
    save(fig, "fig3_detection_vs_k.png")


def fig_distortion() -> None:
    rows = read_csv(REPO / "results" / "summary" / "e1_distortion.csv")
    rows = [r for r in rows if r["k"] != "full"]
    ks = [int(r["k"]) for r in rows]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for i, (col, label) in enumerate((("mean_abs_err", "mean"), ("p95_abs_err", "95th percentile"))):
        y = [float(r[col]) for r in rows]
        ax.plot(ks, y, color=SERIES[i], marker=MARKERS[i], label=label)
        end_label(ax, ks[-1], y[-1], label)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks, [str(k) for k in ks])
    ax.set_xlim(ks[0] * 0.8, ks[-1] * 1.9)
    ax.set_xlabel("projected dimension k")
    ax.set_ylabel("|cos projected − cos full|")
    ax.set_title("E1 — cosine-similarity distortion of real client updates", loc="left")
    ax.legend(fontsize=8.5)
    save(fig, "fig4_distortion_vs_k.png")


def fig_evasion(tuned: dict) -> None:
    e6 = SUITE / "e6"
    modes = [("known_projection", "knows R (key leaked)"), ("guessed_projection", "guesses R (key secret)"),
             ("gaussian", "Gaussian noise")]
    budgets = sorted({float(p.name.split("_eps")[1].split("_s")[0]) for p in e6.glob("*_eps*_s*")})
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.4), sharex=True)
    for i, (mode, label) in enumerate(modes):
        rec, asr, full_rec = [], [], []
        for eps in budgets:
            runs = [e6 / f"{mode}_eps{eps}_s{s}" for s in (1, 2) if (e6 / f"{mode}_eps{eps}_s{s}" / "meta.json").exists()]
            sums = [meta(r)["summary"] for r in runs]
            rec.append(np.mean([x["detection_scores"]["recall"] for x in sums]))
            asr.append(np.mean([x["last_half_mean_asr"] for x in sums]))
            full_rec.append(np.mean([shadow(r, "full", 0, tuned["full"])["recall"] for r in runs]))
        axes[0].plot(budgets, rec, color=SERIES[i], marker=MARKERS[i], label=f"attacker {label}")
        axes[1].plot(budgets, asr, color=SERIES[i], marker=MARKERS[i])
        if mode == "known_projection":
            axes[0].plot(budgets, full_rec, color=REFERENCE, marker="x",
                         label="full-dim detector vs attacker that knows R (shadow)")
    clean = [meta(SUITE / "e2" / f"clean_fedavg_s{s}")["summary"]["last_half_mean_asr"] for s in (1, 2, 3)]
    axes[1].axhline(np.mean(clean), color=REFERENCE, linewidth=1.2)
    axes[1].annotate("no-attack level", (0.07, np.mean(clean)), xytext=(0, 9), textcoords="offset points",
                     ha="center", fontsize=8.5, color=INK2)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("evasion budget ε = ‖perturbation‖ / ‖update‖")
    axes[0].set_ylim(-0.03, 1.05)
    axes[0].set_title("Sybil recall of the projected defence (k=64)", loc="left")
    axes[1].set_title("Attack success rate under that defence", loc="left")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, fontsize=8.2,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("E6 — what the secret projection key buys (seeds 1–2; ASR = mean of rounds 11–20)",
                 x=0.01, ha="left", fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(PLOTS / "fig5_evasion.png", dpi=160)
    plt.close(fig)
    print("saved plots/fig5_evasion.png")


def fig_runtime() -> None:
    rows = read_csv(SUITE / "e5" / "scalability.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    vn = [r for r in rows if r["grid"] == "vary_n"]
    n = [int(r["n"]) for r in vn]
    for i, (col, label) in enumerate((("full_similarity", "full-dim similarity"),
                                      ("projected_total_shake256", "projection (SHAKE-256 matrix) + similarity"))):
        y = [float(r[col]) * 1e3 for r in vn]
        axes[0].plot(n, y, color=SERIES[i], marker=MARKERS[i], label=label)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("clients n (d = 46,730, k = 64)")
    axes[0].set_ylabel("milliseconds per round")
    axes[0].set_title("E5 — detector similarity cost vs n", loc="left")
    axes[0].legend(fontsize=8)
    vk = [r for r in rows if r["grid"] == "vary_k"]
    k = [int(r["k"]) for r in vk]
    for i, (col, label) in enumerate((("gen_shake256", "keyed SHAKE-256 (cryptographic)"),
                                      ("gen_pcg64", "PCG64 (not cryptographic)"))):
        y = [float(r[col]) * 1e3 for r in vk]
        axes[1].plot(k, y, color=SERIES[i], marker=MARKERS[i], label=label)
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(k, [str(v) for v in k])
    axes[1].set_xlabel("projected dimension k (d = 46,730)")
    axes[1].set_ylabel("milliseconds per matrix")
    axes[1].set_title("Cost of generating R_t", loc="left")
    axes[1].legend(fontsize=8)
    save(fig, "fig6_runtime.png")


def fig_label_skew(tuned: dict) -> None:
    alphas = [0.1, 0.5, 5.0]
    def run(alpha, s):
        return SUITE / "e2" / f"clean_fedavg_s{s}" if alpha == 0.5 else SUITE / "e3" / f"clean_alpha{alpha}_s{s}"
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    width = 0.36
    for i, (method, k, tau, label) in enumerate((("full", 0, tuned["full"], "full-dim"),
                                                  ("projected", 64, tuned["projected_k64"], "projected k=64"))):
        vals = [np.mean([shadow(run(a, s), method, k, tau)["fpr"] for s in (1, 2)]) for a in alphas]
        xs = np.arange(len(alphas)) + (i - 0.5) * (width + 0.02)
        ax.bar(xs, vals, width=width, color=SERIES[i], label=label)
        for x_, v in zip(xs, vals):
            ax.annotate(f"{v:.3f}", (x_, v), xytext=(0, 3), textcoords="offset points", ha="center",
                        fontsize=8, color=INK2)
    ax.set_xticks(np.arange(len(alphas)), [f"α = {a}" for a in alphas])
    ax.set_xlabel("Dirichlet label skew (smaller α = more non-IID)")
    ax.set_ylabel("false-positive rate, no attack")
    ax.set_title("E3 — honest clients flagged vs label skew (seeds 1–2)", loc="left")
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=8.5)
    save(fig, "fig7_fpr_vs_alpha.png")


def main() -> None:
    PLOTS.mkdir(exist_ok=True)
    tuned = yaml.safe_load((REPO / "configs" / "tuned_thresholds.yaml").read_text())
    fig_rounds("test_acc", "clean test accuracy", "Clean accuracy, 8 Sybils attacking (mean ± std, seeds 1–3)",
               "fig1_accuracy_rounds.png")
    fig_rounds("asr", "attack success rate (7 → 1)", "Attack success rate (mean ± std, seeds 1–3)",
               "fig2_asr_rounds.png")
    fig_detection_vs_k(tuned)
    fig_distortion()
    if (SUITE / "e6").exists():
        fig_evasion(tuned)
    if (SUITE / "e5" / "scalability.csv").exists():
        fig_runtime()
    if (SUITE / "e3").exists():
        fig_label_skew(tuned)


if __name__ == "__main__":
    main()
