"""
Continual Learning Experiment Runner
====================================
Runs configurable continual learning experiments on supported datasets/models.

Examples:
  python run_all.py
  python run_all.py --methods gem
  python run_all.py --dataset cifar10 --model cnn --no-plot
  python run_all.py --plot-only results.pkl
"""

import argparse
import pickle
import random

import numpy as np
import torch

from datasets import SplitCIFAR10, SplitMNIST
from models import CIFAR10_CNN, MNIST_MLP
from train import TaskIncrementalLearner
from visualize import plot_results


METHOD_DISPLAY_NAMES = {
    "finetune": "Fine-tuning",
    "ewc": "EWC",
    "derpp": "DER++",
    "hat": "HAT",
    "co2l": "Co2L",
    "gem": "GEM",
    "lwf": "LwF",
    "si": "SI",
}

DEFAULT_METHODS = ["finetune", "ewc", "derpp", "hat", "co2l", "gem", "lwf", "si"]

EXPERIMENT_REGISTRY = {
    "mnist": {
        "dataset_name": "MNIST",
        "dataset_class": SplitMNIST,
        "models": {
            "mlp": MNIST_MLP,
            "mnist_mlp": MNIST_MLP,
        },
        "default_model": "mlp",
    },
    "cifar10": {
        "dataset_name": "CIFAR-10",
        "dataset_class": SplitCIFAR10,
        "models": {
            "cnn": CIFAR10_CNN,
            "cifar10_cnn": CIFAR10_CNN,
        },
        "default_model": "cnn",
    },
}


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def canonicalize_methods(methods):
    """Normalize and validate method names while preserving order."""
    normalized = []

    for method in methods:
        method_name = method.lower()
        if method_name not in METHOD_DISPLAY_NAMES:
            valid_methods = ", ".join(METHOD_DISPLAY_NAMES.keys())
            raise ValueError(f"Unsupported method '{method}'. Choose from: {valid_methods}")

        if method_name not in normalized:
            normalized.append(method_name)

    return normalized


def resolve_experiment_config(dataset_key, model_key=None):
    """Resolve dataset/model classes from registry and validate compatibility."""
    dataset_config = EXPERIMENT_REGISTRY[dataset_key]

    if model_key is None:
        model_key = dataset_config["default_model"]

    normalized_model_key = model_key.lower()
    if normalized_model_key not in dataset_config["models"]:
        valid_models = ", ".join(dataset_config["models"].keys())
        dataset_name = dataset_config["dataset_name"]
        raise ValueError(
            f"Model '{model_key}' is not available for {dataset_name}. "
            f"Choose from: {valid_models}"
        )

    return {
        "dataset_name": dataset_config["dataset_name"],
        "dataset_class": dataset_config["dataset_class"],
        "model_name": normalized_model_key,
        "model_class": dataset_config["models"][normalized_model_key],
    }


def run_experiment(config, methods, device="cpu", learning_rate=0.001, epochs=5):
    """
    Run continual learning experiment on a single dataset using selected methods.

    Args:
        config (dict): Resolved dataset/model configuration
        methods (list[str]): Continual learning methods to run
        device (str): "cpu" or "cuda"
        learning_rate (float): Learning rate passed to the learner
        epochs (int): Epochs per task

    Returns:
        dict: Results keyed by method name
    """
    dataset_name = config["dataset_name"]
    model_name = config["model_name"]

    print(f"\n{'=' * 60}")
    print(f"Running Continual Learning on {dataset_name}")
    print(f"Model: {model_name}")
    print(f"Methods: {', '.join(methods)}")
    print(f"{'=' * 60}")

    dataset = config["dataset_class"]()
    learner = TaskIncrementalLearner(
        model_class=config["model_class"],
        num_tasks=dataset.num_tasks,
        dataset=dataset,
        device=device,
        learning_rate=learning_rate,
        epochs=epochs,
    )

    results = {}
    for method in methods:
        print(f"\n[{dataset_name}] Training with {METHOD_DISPLAY_NAMES[method]}...")
        results[method] = learner.train_all_tasks(method_name=method, verbose=True)

    return results


def _format_percentage(value):
    """Format a ratio value as a percentage string."""
    return f"{value * 100:.2f}%"


def _build_text_table(headers, rows):
    """Build a simple ASCII table without extra dependencies."""
    widths = [len(header) for header in headers]

    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(str(cell)))

    def make_row(values):
        padded = [str(value).ljust(widths[idx]) for idx, value in enumerate(values)]
        return f"| {' | '.join(padded)} |"

    separator = f"+-{'-+-'.join('-' * width for width in widths)}-+"
    lines = [separator, make_row(headers), separator]
    lines.extend(make_row(row) for row in rows)
    lines.append(separator)
    return "\n".join(lines)


def print_results_summary(all_results):
    """Print compact per-dataset summary tables to the console."""
    print(f"\n{'=' * 60}")
    print("Results Summary")
    print(f"{'=' * 60}")

    for dataset_name, dataset_results in all_results.items():
        print(f"\n[{dataset_name}]")

        headers = ["Method", "Final Avg Acc", "Final Task Acc", "Best Avg Acc", "BWT"]
        rows = []

        for method_name, result in dataset_results.items():
            avg_accuracy = result.get("avg_accuracy", [])
            accuracy_matrix = result.get("accuracy_matrix", [])
            final_row = accuracy_matrix[-1] if accuracy_matrix else []

            final_avg_acc = avg_accuracy[-1] if avg_accuracy else 0.0
            final_task_acc = final_row[-1] if final_row else 0.0
            best_avg_acc = max(avg_accuracy) if avg_accuracy else 0.0
            bwt = result.get("bwt", 0.0)

            rows.append([
                METHOD_DISPLAY_NAMES.get(method_name, method_name),
                _format_percentage(final_avg_acc),
                _format_percentage(final_task_acc),
                _format_percentage(best_avg_acc),
                f"{bwt:+.4f}",
            ])

        print(_build_text_table(headers, rows))


def save_results(results, output_path):
    """Persist experiment results to disk."""
    with open(output_path, "wb") as file_obj:
        pickle.dump(results, file_obj)
    print(f"[✓] Saved results to {output_path}")


def load_results(results_path):
    """Load previously saved experiment results."""
    with open(results_path, "rb") as file_obj:
        return pickle.load(file_obj)


def build_argument_parser():
    """Create CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Run continual learning experiments with selectable datasets, models, and methods."
    )
    parser.add_argument(
        "--dataset",
        choices=["all", *EXPERIMENT_REGISTRY.keys()],
        default="all",
        help="Dataset to run. Use 'all' to run every registered dataset.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model key for the selected dataset, e.g. 'mlp' or 'cnn'.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=DEFAULT_METHODS,
        help="Methods to run, e.g. finetune ewc derpp.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Epochs per task.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate for the learner.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Training device. 'auto' uses CUDA when available.",
    )
    parser.add_argument(
        "--output",
        default="results.pkl",
        help="Path to save experiment results.",
    )
    parser.add_argument(
        "--plot-output",
        default="continual_learning_results.png",
        help="Path to save generated plot.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip visualization after running experiments.",
    )
    parser.add_argument(
        "--plot-only",
        default=None,
        metavar="RESULTS_PKL",
        help="Load a saved results file and only generate plots.",
    )
    return parser


def main():
    """Run configurable experiments or visualize saved results."""
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.plot_only is not None:
        print(f"Loading results from {args.plot_only}...")
        saved_results = load_results(args.plot_only)
        print_results_summary(saved_results)
        plot_results(saved_results, output_path=args.plot_output)
        return

    methods = canonicalize_methods(args.methods)
    set_seed(args.seed)

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if args.device == "auto" and device == "auto":
        device = "cpu"

    print(f"Using device: {device}")
    print(f"PyTorch version: {torch.__version__}")

    dataset_keys = list(EXPERIMENT_REGISTRY.keys()) if args.dataset == "all" else [args.dataset]
    all_results = {}

    for dataset_key in dataset_keys:
        config = resolve_experiment_config(dataset_key, args.model)
        dataset_results = run_experiment(
            config=config,
            methods=methods,
            device=device,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
        )
        all_results[config["dataset_name"]] = dataset_results

    print_results_summary(all_results)

    print(f"\n{'=' * 60}")
    print("Saving results...")
    print(f"{'=' * 60}")
    save_results(all_results, args.output)

    if not args.no_plot:
        print(f"\n{'=' * 60}")
        print("Generating visualizations...")
        print(f"{'=' * 60}")
        plot_results(all_results, output_path=args.plot_output)

    print(f"\n{'=' * 60}")
    print("Experiment completed successfully!")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
