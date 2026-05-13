import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
from sklearn.metrics import roc_curve, auc

OUTPUT_DIR = "./outputs"
FIGURES_DIR = "./outputs/figures"
CONDITIONS = ["baseline", "kgw_static", "kgw_entropy", "synthid_static", "synthid_entropy"]
COLORS = {
    "baseline":        "#666666",
    "kgw_static":      "#2166ac",
    "kgw_entropy":     "#92c5de",
    "synthid_static":  "#d6604d",
    "synthid_entropy": "#f4a582",
}
LABELS = {
    "baseline":        "Baseline (no watermark)",
    "kgw_static":      "KGW Static",
    "kgw_entropy":     "KGW Entropy-Adaptive",
    "synthid_static":  "SynthID Static",
    "synthid_entropy": "SynthID Entropy-Adaptive",
}

def load_detections():
    path = os.path.join(OUTPUT_DIR, "detections.jsonl")
    data = defaultdict(list)
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            cond = item["condition"]
            data[cond].append(item)
    return data

def load_perplexity():
    path = os.path.join(OUTPUT_DIR, "perplexity.jsonl")
    data = defaultdict(list)
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            if item.get("perplexity") is not None:
                data[item["condition"]].append(item["perplexity"])
    return data

def load_generations():
    path = os.path.join(OUTPUT_DIR, "generations.jsonl")
    data = defaultdict(list)
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            if item.get("output"):
                data[item["condition"]].append(item)
    return data

def plot_auroc_kgw(detections):
    fig, ax = plt.subplots(figsize=(6, 5))

    baseline_z = [d["kgw_z_score"] for d in detections["baseline"]
                  if d.get("kgw_z_score") is not None]

    for condition in ["kgw_static", "kgw_entropy"]:
        wm_z = [d["kgw_z_score"] for d in detections[condition]
                if d.get("kgw_z_score") is not None]

        y_true = [0] * len(baseline_z) + [1] * len(wm_z)
        y_score = baseline_z + wm_z

        fpr, tpr, _ = roc_curve(y_true, y_score)
        auroc = auc(fpr, tpr)

        ax.plot(fpr, tpr,
                color=COLORS[condition],
                linewidth=2,
                label=f"{LABELS[condition]} (AUROC={auroc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random (AUROC=0.500)")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("KGW Watermark Detection ROC Curve", fontsize=13)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "auroc_kgw.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")

def plot_auroc_synthid(detections):
    fig, ax = plt.subplots(figsize=(6, 5))

    baseline_s = [d["synthid_score"] for d in detections["baseline"]
                  if d.get("synthid_score") is not None]

    for condition in ["synthid_static", "synthid_entropy"]:
        wm_s = [d["synthid_score"] for d in detections[condition]
                if d.get("synthid_score") is not None]

        y_true = [0] * len(baseline_s) + [1] * len(wm_s)
        y_score = baseline_s + wm_s

        fpr, tpr, _ = roc_curve(y_true, y_score)
        auroc = auc(fpr, tpr)

        ax.plot(fpr, tpr,
                color=COLORS[condition],
                linewidth=2,
                label=f"{LABELS[condition]} (AUROC={auroc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random (AUROC=0.500)")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("SynthID Watermark Detection ROC Curve", fontsize=13)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "auroc_synthid.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")

def plot_perplexity_box(perplexity):
    fig, ax = plt.subplots(figsize=(8, 5))

    plot_data = []
    plot_labels = []
    plot_colors = []

    for cond in CONDITIONS:
        vals = perplexity.get(cond, [])
        if vals:
            cap = np.percentile(vals, 95)
            capped = [min(v, cap) for v in vals]
            plot_data.append(capped)
            plot_labels.append(LABELS[cond].replace(" ", "\n"))
            plot_colors.append(COLORS[cond])

    bp = ax.boxplot(plot_data, patch_artist=True, notch=False,
                    medianprops=dict(color="black", linewidth=2))

    for patch, color in zip(bp["boxes"], plot_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax.set_xticklabels(plot_labels, fontsize=9)
    ax.set_ylabel("Perplexity (capped at 95th pct)", fontsize=11)
    ax.set_title("Text Quality: Perplexity by Condition", fontsize=13)
    ax.grid(True, axis="y", alpha=0.3)

    for i, vals in enumerate(plot_data):
        ax.scatter(i + 1, np.mean(vals), marker="D", color="black",
                   s=30, zorder=5, label="Mean" if i == 0 else "")

    ax.legend(fontsize=9)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "perplexity_box.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")

def plot_zscore_dist(detections):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)

    kgw_conditions = [
        ("kgw_static",  "KGW Static vs Baseline"),
        ("kgw_entropy", "KGW Entropy-Adaptive vs Baseline"),
    ]

    baseline_z = [d["kgw_z_score"] for d in detections["baseline"]
                  if d.get("kgw_z_score") is not None]

    for ax, (condition, title) in zip(axes, kgw_conditions):
        wm_z = [d["kgw_z_score"] for d in detections[condition]
                if d.get("kgw_z_score") is not None]

        bins = np.linspace(-5, 15, 60)
        ax.hist(baseline_z, bins=bins, alpha=0.6,
                color=COLORS["baseline"], label="Baseline", density=True)
        ax.hist(wm_z, bins=bins, alpha=0.6,
                color=COLORS[condition], label=LABELS[condition], density=True)

        ax.axvline(x=4.0, color="red", linestyle="--",
                   linewidth=1.5, label="Threshold (z=4.0)")

        ax.set_xlabel("Z-score", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("KGW Z-Score Distributions", fontsize=13, y=1.02)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "zscore_dist.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

def plot_synthid_dist(detections):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)

    synthid_conditions = [
        ("synthid_static",  "SynthID Static vs Baseline"),
        ("synthid_entropy", "SynthID Entropy-Adaptive vs Baseline"),
    ]

    baseline_s = [d["synthid_score"] for d in detections["baseline"]
                  if d.get("synthid_score") is not None]

    threshold = 0.5
    for d in detections["synthid_static"]:
        if d.get("synthid_threshold") is not None:
            threshold = d["synthid_threshold"]
            break

    for ax, (condition, title) in zip(axes, synthid_conditions):
        wm_s = [d["synthid_score"] for d in detections[condition]
                if d.get("synthid_score") is not None]

        bins = np.linspace(0.46, 0.56, 50)
        ax.hist(baseline_s, bins=bins, alpha=0.6,
                color=COLORS["baseline"], label="Baseline", density=True)
        ax.hist(wm_s, bins=bins, alpha=0.6,
                color=COLORS[condition], label=LABELS[condition], density=True)

        ax.axvline(x=threshold, color="red", linestyle="--",
                   linewidth=1.5, label=f"Threshold ({threshold:.4f})")

        ax.set_xlabel("Mean G-Value Score", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("SynthID Mean G-Value Score Distributions", fontsize=13, y=1.02)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "synthid_dist.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def compute_tpr_at_fpr(detections, target_fpr=0.01):
    results = {}

    baseline_z = [d["kgw_z_score"] for d in detections["baseline"]
                  if d.get("kgw_z_score") is not None]
    baseline_s = [d["synthid_score"] for d in detections["baseline"]
                  if d.get("synthid_score") is not None]

    for condition in ["kgw_static", "kgw_entropy"]:
        wm_z = [d["kgw_z_score"] for d in detections[condition]
                if d.get("kgw_z_score") is not None]
        y_true = [0] * len(baseline_z) + [1] * len(wm_z)
        y_score = baseline_z + wm_z
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auroc = auc(fpr, tpr)
        idx = np.searchsorted(fpr, target_fpr)
        tpr_at_fpr = tpr[min(idx, len(tpr)-1)]
        results[condition] = {"auroc": auroc, "tpr_at_1fpr": tpr_at_fpr}

    for condition in ["synthid_static", "synthid_entropy"]:
        wm_s = [d["synthid_score"] for d in detections[condition]
                if d.get("synthid_score") is not None]
        y_true = [0] * len(baseline_s) + [1] * len(wm_s)
        y_score = baseline_s + wm_s
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auroc = auc(fpr, tpr)
        idx = np.searchsorted(fpr, target_fpr)
        tpr_at_fpr = tpr[min(idx, len(tpr)-1)]
        results[condition] = {"auroc": auroc, "tpr_at_1fpr": tpr_at_fpr}

    return results

def print_results_table(detections, perplexity, tpr_results):
    lines = []
    lines.append("=" * 80)
    lines.append("ENTROPY-ADAPTIVE WATERMARKING — FULL RESULTS")
    lines.append("=" * 80)
    lines.append(f"\nModel: meta-llama/Llama-3.2-1B-Instruct")
    lines.append(f"Dataset: databricks/databricks-dolly-15k (1000 prompts, 6 categories)")
    lines.append(f"Max tokens: 400 | KGW gamma=0.25, delta=2.0, scheme=minhash")
    lines.append(f"SynthID ngram_len=5, num_keys=20")

    lines.append("\n" + "-" * 80)
    lines.append("DETECTION RESULTS")
    lines.append("-" * 80)
    lines.append(f"  {'Condition':<25} {'Mean Z':>10} {'SynthID':>10} {'AUROC':>8} {'TPR@1%FPR':>12}")
    lines.append(f"  {'-'*70}")

    bz = [d["kgw_z_score"] for d in detections["baseline"] if d.get("kgw_z_score") is not None]
    bs = [d["synthid_score"] for d in detections["baseline"] if d.get("synthid_score") is not None]
    lines.append(f"  {'baseline':<25} {np.mean(bz):>10.3f} {np.mean(bs):>10.4f} {'N/A':>8} {'N/A':>12}")

    for condition in ["kgw_static", "kgw_entropy", "synthid_static", "synthid_entropy"]:
        dets = detections[condition]
        z_scores = [d["kgw_z_score"] for d in dets if d.get("kgw_z_score") is not None]
        s_scores = [d["synthid_score"] for d in dets if d.get("synthid_score") is not None]
        mean_z = np.mean(z_scores) if z_scores else float("nan")
        mean_s = np.mean(s_scores) if s_scores else float("nan")
        auroc = tpr_results.get(condition, {}).get("auroc", float("nan"))
        tpr = tpr_results.get(condition, {}).get("tpr_at_1fpr", float("nan"))
        lines.append(f"  {condition:<25} {mean_z:>10.3f} {mean_s:>10.4f} {auroc:>8.3f} {tpr:>12.3f}")

    lines.append("\n" + "-" * 80)
    lines.append("PERPLEXITY RESULTS (lower = better quality)")
    lines.append("-" * 80)
    lines.append(f"  {'Condition':<25} {'Mean PPL':>10} {'Median PPL':>12} {'vs Baseline':>14}")
    lines.append(f"  {'-'*65}")

    baseline_ppl = np.mean(perplexity.get("baseline", [np.nan]))
    for cond in CONDITIONS:
        vals = perplexity.get(cond, [])
        if vals:
            mean_ppl = np.mean(vals)
            med_ppl = np.median(vals)
            delta = mean_ppl - baseline_ppl
            delta_str = f"{delta:+.2f}"
            lines.append(f"  {cond:<25} {mean_ppl:>10.2f} {med_ppl:>12.2f} {delta_str:>14}")

    lines.append("\n" + "-" * 80)
    lines.append("SUMMARY")
    lines.append("-" * 80)

    kgw_ppl_improvement = np.mean(perplexity["kgw_static"]) - np.mean(perplexity["kgw_entropy"])
    kgw_auroc_static = tpr_results["kgw_static"]["auroc"]
    kgw_auroc_entropy = tpr_results["kgw_entropy"]["auroc"]
    synthid_ppl_diff = np.mean(perplexity["synthid_entropy"]) - np.mean(perplexity["synthid_static"])
    synthid_auroc_static = tpr_results["synthid_static"]["auroc"]
    synthid_auroc_entropy = tpr_results["synthid_entropy"]["auroc"]

    lines.append(f"\n  KGW:")
    lines.append(f"    Perplexity improvement (static→entropy): {kgw_ppl_improvement:.2f} PPL")
    lines.append(f"    AUROC (static): {kgw_auroc_static:.3f}")
    lines.append(f"    AUROC (entropy): {kgw_auroc_entropy:.3f}")
    lines.append(f"    AUROC cost of entropy adaptation: {kgw_auroc_static - kgw_auroc_entropy:.3f}")

    lines.append(f"\n  SynthID:")
    lines.append(f"    Perplexity change (static→entropy): {synthid_ppl_diff:+.2f} PPL")
    lines.append(f"    AUROC (static): {synthid_auroc_static:.3f}")
    lines.append(f"    AUROC (entropy): {synthid_auroc_entropy:.3f}")
    lines.append(f"    AUROC cost of entropy adaptation: {synthid_auroc_static - synthid_auroc_entropy:.3f}")

    lines.append("\n" + "=" * 80)

    output = "\n".join(lines)
    print(output)

    path = os.path.join(FIGURES_DIR, "results_table.txt")
    with open(path, "w") as f:
        f.write(output)
    print(f"\nSaved: {path}")

def plot_tradeoff_kgw(detections, perplexity, tpr_results):
    fig, ax = plt.subplots(figsize=(7, 5))

    conditions_to_plot = ["baseline", "kgw_static", "kgw_entropy"]
    point_labels = {
        "baseline":    "Baseline\n(no watermark)",
        "kgw_static":  "KGW Static\n(delta=2.0)",
        "kgw_entropy": "KGW Entropy-\nAdaptive",
    }

    xs, ys, aurocs, labels = [], [], [], []
    for cond in conditions_to_plot:
        ppl_vals = perplexity.get(cond, [])
        z_vals = [d["kgw_z_score"] for d in detections[cond]
                  if d.get("kgw_z_score") is not None]
        if not ppl_vals or not z_vals:
            continue
        xs.append(np.mean(ppl_vals))
        ys.append(np.mean(z_vals))
        aurocs.append(tpr_results.get(cond, {}).get("auroc", 0.0))
        labels.append(point_labels[cond])

    auroc_arr = np.array(aurocs)

    scatter = ax.scatter(xs, ys,
                         c=auroc_arr,
                         cmap="RdYlBu_r",
                         vmin=0.0, vmax=1.0,
                         s=300,
                         zorder=5,
                         edgecolors="black",
                         linewidths=0.5)

    for x, y, label in zip(xs, ys, labels):
        ax.annotate(label,
                    xy=(x, y),
                    xytext=(8, 4),
                    textcoords="offset points",
                    fontsize=8,
                    va="bottom")

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("ROC-AUC", fontsize=10)

    ax.set_xlabel("Mean Perplexity", fontsize=11)
    ax.set_ylabel("Mean Z-Score", fontsize=11)
    ax.set_title("KGW: Detection vs Distortion Tradeoff", fontsize=13)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "tradeoff_kgw.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")

def plot_tradeoff_synthid(detections, perplexity, tpr_results):
    fig, ax = plt.subplots(figsize=(7, 5))

    conditions_to_plot = ["baseline", "synthid_static", "synthid_entropy"]
    point_labels = {
        "baseline":        "Baseline\n(no watermark)",
        "synthid_static":  "SynthID Static",
        "synthid_entropy": "SynthID Entropy-\nAdaptive",
    }

    xs, ys, aurocs, labels = [], [], [], []
    for cond in conditions_to_plot:
        ppl_vals = perplexity.get(cond, [])
        s_vals = [d["synthid_score"] for d in detections[cond]
                  if d.get("synthid_score") is not None]
        if not ppl_vals or not s_vals:
            continue
        xs.append(np.mean(ppl_vals))
        ys.append(np.mean(s_vals))
        aurocs.append(tpr_results.get(cond, {}).get("auroc", 0.0))
        labels.append(point_labels[cond])

    auroc_arr = np.array(aurocs)

    scatter = ax.scatter(xs, ys,
                         c=auroc_arr,
                         cmap="RdYlBu_r",
                         vmin=0.0, vmax=1.0,
                         s=300,
                         zorder=5,
                         edgecolors="black",
                         linewidths=0.5)

    for x, y, label in zip(xs, ys, labels):
        ax.annotate(label,
                    xy=(x, y),
                    xytext=(8, 4),
                    textcoords="offset points",
                    fontsize=8,
                    va="bottom")

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("ROC-AUC", fontsize=10)

    ax.set_xlabel("Mean Perplexity", fontsize=11)
    ax.set_ylabel("Mean G-Value Score", fontsize=11)
    ax.set_title("SynthID: Detection vs Distortion Tradeoff", fontsize=13)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "tradeoff_synthid.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")

def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("Loading data...")
    detections = load_detections()
    perplexity = load_perplexity()

    print("Computing TPR @ 1% FPR...")
    tpr_results = compute_tpr_at_fpr(detections)

    print("Generating plots...")
    plot_auroc_kgw(detections)
    plot_auroc_synthid(detections)
    plot_perplexity_box(perplexity)
    plot_zscore_dist(detections)
    plot_synthid_dist(detections)

    print("Generating tradeoff plots...")
    plot_tradeoff_kgw(detections, perplexity, tpr_results)
    plot_tradeoff_synthid(detections, perplexity, tpr_results)

    print("Generating results table...")
    print_results_table(detections, perplexity, tpr_results)

    print(f"\nAll outputs saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()