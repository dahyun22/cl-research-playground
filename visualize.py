"""
Visualization Module
====================
Generates plots for any subset of datasets and continual learning methods.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec


METHOD_NAMES = {
    "finetune": "Fine-tuning",
    "ewc": "EWC",
    "derpp": "DER++",
    "hat": "HAT",
}

METHOD_COLORS = {
    "finetune": "#FF6B6B",
    "ewc": "#4ECDC4",
    "derpp": "#45B7D1",
    "hat": "#F4A261",
}

PREFERRED_METHOD_ORDER = ["finetune", "ewc", "derpp", "hat"]


def _collect_datasets_and_methods(all_results):
    """Infer dataset and method ordering from the results dictionary."""
    datasets = list(all_results.keys())

    discovered_methods = []
    for dataset in datasets:
        for method in all_results[dataset].keys():
            if method not in discovered_methods:
                discovered_methods.append(method)

    ordered_methods = [method for method in PREFERRED_METHOD_ORDER if method in discovered_methods]
    ordered_methods.extend(
        method for method in discovered_methods if method not in ordered_methods
    )

    return datasets, ordered_methods


def _format_method_name(method):
    """Return a human-readable method label."""
    return METHOD_NAMES.get(method, method.upper())


def _format_method_color(method, index):
    """Return a plotting color, falling back to a tab20 color cycle."""
    if method in METHOD_COLORS:
        return METHOD_COLORS[method]
    return plt.cm.tab20(index % 20)


def _pad_accuracy_matrix(accuracy_matrix):
    """Pad ragged triangular accuracy matrix with NaNs for heatmap plotting."""
    max_len = max(len(row) for row in accuracy_matrix)
    return np.array([row + [np.nan] * (max_len - len(row)) for row in accuracy_matrix])


def plot_results(all_results, output_path="continual_learning_results.png"):
    """
    Create visualization for the datasets/methods present in the results.

    Args:
        all_results (dict): Results dict of shape {dataset: {method: results_dict}}
        output_path (str): File path for the saved figure
    """
    if not all_results:
        raise ValueError("No results available to plot.")

    datasets, methods = _collect_datasets_and_methods(all_results)
    if not methods:
        raise ValueError("No method results found to plot.")

    num_heatmap_rows = len(datasets)
    num_heatmap_cols = max(len(methods), 1)
    num_columns = max(num_heatmap_cols, 2)

    fig_width = max(5 * num_columns, 12)
    fig_height = max(3.5 * (num_heatmap_rows + 2), 10)
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = GridSpec(num_heatmap_rows + 2, num_columns, figure=fig, hspace=0.45, wspace=0.35)

    # Heatmaps
    for dataset_idx, dataset in enumerate(datasets):
        dataset_methods = [method for method in methods if method in all_results[dataset]]

        for method_idx, method in enumerate(dataset_methods):
            ax = fig.add_subplot(gs[dataset_idx, method_idx])
            accuracy_matrix = all_results[dataset][method]["accuracy_matrix"]
            acc_array = _pad_accuracy_matrix(accuracy_matrix)

            im = ax.imshow(acc_array, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
            ax.set_xlabel("Task Evaluated", fontsize=10)
            ax.set_ylabel("Task Learned", fontsize=10)
            ax.set_title(
                f"{dataset} - {_format_method_name(method)}",
                fontsize=11,
                fontweight="bold",
            )

            num_tasks = acc_array.shape[1]
            ax.set_xticks(range(num_tasks))
            ax.set_yticks(range(len(accuracy_matrix)))
            ax.set_xticklabels(range(num_tasks))
            ax.set_yticklabels(range(len(accuracy_matrix)))

            for row_idx in range(len(accuracy_matrix)):
                for col_idx in range(len(accuracy_matrix[row_idx])):
                    ax.text(
                        col_idx,
                        row_idx,
                        f"{accuracy_matrix[row_idx][col_idx]:.2f}",
                        ha="center",
                        va="center",
                        color="black",
                        fontsize=8,
                    )

            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Accuracy", fontsize=9)

        for empty_col in range(len(dataset_methods), num_columns):
            ax = fig.add_subplot(gs[dataset_idx, empty_col])
            ax.axis("off")

    # Average accuracy curves
    ax_avg = fig.add_subplot(gs[num_heatmap_rows, : max(num_columns - 1, 1)])
    style_markers = ["o", "s", "^", "D", "P", "X"]
    style_lines = ["-", "--", "-.", ":"]

    for dataset_idx, dataset in enumerate(datasets):
        linestyle = style_lines[dataset_idx % len(style_lines)]
        marker = style_markers[dataset_idx % len(style_markers)]

        for method_idx, method in enumerate(methods):
            if method not in all_results[dataset]:
                continue

            avg_accs = all_results[dataset][method]["avg_accuracy"]
            x_pos = np.arange(len(avg_accs)) + 1
            label = f"{_format_method_name(method)} ({dataset})"

            ax_avg.plot(
                x_pos,
                avg_accs,
                label=label,
                color=_format_method_color(method, method_idx),
                marker=marker,
                linestyle=linestyle,
                linewidth=2,
                markersize=6,
            )

    ax_avg.set_xlabel("Task Index", fontsize=11)
    ax_avg.set_ylabel("Average Accuracy", fontsize=11)
    ax_avg.set_title("Average Accuracy Across Learned Tasks", fontsize=12, fontweight="bold")
    ax_avg.grid(True, alpha=0.3)
    ax_avg.legend(loc="best", fontsize=9, ncol=2)

    max_task_count = 1
    for dataset in datasets:
        for method in methods:
            if method in all_results[dataset]:
                max_task_count = max(max_task_count, len(all_results[dataset][method]["avg_accuracy"]))
    ax_avg.set_xticks(range(1, max_task_count + 1))

    # Weight drift
    ax_drift = fig.add_subplot(gs[num_heatmap_rows, max(num_columns - 1, 1)])
    for dataset_idx, dataset in enumerate(datasets):
        marker = style_markers[dataset_idx % len(style_markers)]

        for method_idx, method in enumerate(methods):
            if method not in all_results[dataset]:
                continue

            weight_drift = all_results[dataset][method].get("weight_drift", {})
            if not weight_drift:
                continue

            first_layer = list(weight_drift.keys())[0]
            drifts = weight_drift[first_layer]
            x_pos = np.arange(len(drifts)) + 1
            label = f"{_format_method_name(method)} ({dataset})"

            ax_drift.plot(
                x_pos,
                drifts,
                label=label,
                color=_format_method_color(method, method_idx),
                marker=marker,
                linewidth=2,
                markersize=6,
            )

    ax_drift.set_xlabel("Task Index", fontsize=11)
    ax_drift.set_ylabel("L2 Parameter Drift", fontsize=11)
    ax_drift.set_title("Weight Drift (First Recorded Layer)", fontsize=12, fontweight="bold")
    ax_drift.grid(True, alpha=0.3)
    if ax_drift.lines:
        ax_drift.legend(fontsize=8)
        ax_drift.set_xticks(range(1, max_task_count + 1))
    else:
        ax_drift.text(0.5, 0.5, "No weight drift data", ha="center", va="center", transform=ax_drift.transAxes)

    # Backward transfer
    ax_bwt = fig.add_subplot(gs[num_heatmap_rows + 1, :])
    if len(datasets) == 1:
        width = 0.6
    else:
        width = min(0.8 / len(datasets), 0.35)

    x_pos = np.arange(len(methods))

    for dataset_idx, dataset in enumerate(datasets):
        offsets = (dataset_idx - (len(datasets) - 1) / 2.0) * width
        bwt_values = []

        for method in methods:
            dataset_method_results = all_results[dataset].get(method)
            bwt_values.append(np.nan if dataset_method_results is None else dataset_method_results["bwt"])

        bars = ax_bwt.bar(
            x_pos + offsets,
            bwt_values,
            width,
            label=dataset,
            alpha=0.8,
            edgecolor="black",
            linewidth=1.2,
        )

        for bar, value in zip(bars, bwt_values):
            if np.isnan(value):
                continue
            ax_bwt.text(
                bar.get_x() + bar.get_width() / 2.0,
                value,
                f"{value:.3f}",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=8,
                fontweight="bold",
            )

    ax_bwt.set_ylabel("Backward Transfer (BWT)", fontsize=11)
    ax_bwt.set_title("Backward Transfer by Dataset and Method", fontsize=12, fontweight="bold")
    ax_bwt.set_xticks(x_pos)
    ax_bwt.set_xticklabels([_format_method_name(method) for method in methods])
    ax_bwt.axhline(y=0, color="black", linestyle="-", linewidth=1)
    ax_bwt.grid(True, alpha=0.3, axis="y")
    ax_bwt.legend(fontsize=9)

    dataset_summary = ", ".join(datasets)
    fig.suptitle(
        f"Continual Learning Results ({dataset_summary})",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n[✓] Visualization saved to {output_path}")

    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    for dataset in datasets:
        print(f"\n{dataset}:")
        print("-" * 70)

        for method in methods:
            if method not in all_results[dataset]:
                continue

            results = all_results[dataset][method]
            final_acc = np.mean(results["accuracy_matrix"][-1])
            bwt = results["bwt"]

            print(
                f"  {_format_method_name(method):15s} | "
                f"Final Avg Acc: {final_acc:.4f} | BWT: {bwt:.4f}"
            )

    print("=" * 70)
