"""
Utility to visualize training logs from TensorBoard.
Supports both single-run config (config.yml) and multi-run config (model_runs.yml).

Usage:
  - For one run:     python utils/tensorboard_plot_utils.py --config configs/config.yml
  - For many runs:   python utils/tensorboard_plot_utils.py --runs_config configs/model_runs.yml
"""

import os
import yaml
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from datetime import datetime
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # fix for KMP DLL conflicts

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def find_event_file(log_dir):
    for root, _, files in os.walk(log_dir):
        for f in files:
            if f.startswith("events"):
                return os.path.join(root, f)
    return None

def extract_scalar_from_event(event_path, tag):
    ea = EventAccumulator(event_path)
    ea.Reload()
    if tag in ea.Tags()["scalars"]:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        return steps, values
    return [], []

def plot_runs(run_list, save_path):
    """
    Plot multiple runs from a list of dicts with keys 'log_dir' and 'label'.
    """
    plt.figure(figsize=(10, 5))

    # --- Loss Plot ---
    plt.subplot(1, 2, 1)
    for run in run_list:
        event_path = find_event_file(run['log_dir'])
        if event_path:
            train_steps, train_loss = extract_scalar_from_event(event_path, "Loss/Train")
            val_steps, val_loss = extract_scalar_from_event(event_path, "Loss/Val")
            if train_steps and train_loss:
                plt.plot(train_steps, train_loss, label=f"{run['label']} Train")
            if val_steps and val_loss:
                plt.plot(val_steps, val_loss, label=f"{run['label']} Val")
    plt.title("Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()

    # --- Accuracy Plot ---
    plt.subplot(1, 2, 2)
    for run in run_list:
        event_path = find_event_file(run['log_dir'])
        if event_path:
            train_steps, train_acc = extract_scalar_from_event(event_path, "Accuracy/Train")
            val_steps, val_acc = extract_scalar_from_event(event_path, "Accuracy/Val")
            if train_steps and train_acc:
                plt.plot(train_steps, train_acc, label=f"{run['label']} Train")
            if val_steps and val_acc:
                plt.plot(val_steps, val_acc, label=f"{run['label']} Val")
    plt.title("Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✅ Plot saved at {save_path}")
    plt.close()

def plot_from_tensorboard(config):
    """
    Plot a single run given a config dict or YAML path.
    """
    if isinstance(config, str):
        config = load_yaml(config)

    run_label = config['logging'].get('run_label', 'default')
    base_log_dir = config['logging']['log_dir']
    save_dir = os.path.join(base_log_dir, run_label)
    os.makedirs(save_dir, exist_ok=True)

    filename = f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    save_path = os.path.join(save_dir, filename)

    run = {
        "log_dir": base_log_dir,
        "label": run_label
    }

    plot_runs([run], save_path)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='./configs/config.yml', help="Path to config.yml")
    parser.add_argument("--runs_config", type=str, default=None, help="Path to model_runs.yml")

    args = parser.parse_args()

    if args.runs_config:
        runs_data = load_yaml(args.runs_config)
        runs = runs_data["runs"]

        # default multi-run save location
        multi_plot_dir = os.path.join("plots")
        os.makedirs(multi_plot_dir, exist_ok=True)
        save_path = os.path.join(
            multi_plot_dir,
            f"multirun_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        )
        plot_runs(runs, save_path)

    elif args.config:
        config = load_yaml(args.config)
        plot_from_tensorboard(config)

    else:
        print("❌ Please provide either --config or --runs_config")
