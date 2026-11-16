"""Reproduce results.png from the committed results.json."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(output_path="results.png"):
    models = json.loads(Path(__file__).with_name("results.json").read_text())["models"]
    names = list(models)
    auc = [model["auc"] for model in models.values()]
    errors = [
        [model["auc"] - model["auc_ci95"][0] for model in models.values()],
        [model["auc_ci95"][1] - model["auc"] for model in models.values()],
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    axes[0].barh(names, auc, xerr=errors, ecolor="black", capsize=3)
    axes[0].set_title("Test ROC-AUC (95% bootstrap CI)")
    axes[0].set_xlim(0.72, 0.80)
    axes[1].barh(names, [model["cost_per_account"] for model in models.values()])
    axes[1].set_title("Expected cost per account (FN=5x FP)")
    axes[1].set_xlim(0.50, 0.65)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
