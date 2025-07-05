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
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" #this is a patch I hade to make to plot as I have dll conflicts. 

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


def plot_from_tensorboard(config):
    """
    Plot a single run. Accepts either a config dictionary or path to a YAML file.
    """
    if isinstance(config, str):
        with open(config, "r") as f:
            config = yaml.safe_load(f)

    run = {
        "log_dir": config['logging']['log_dir'],
        "label": config['logging'].get('run_label', config['logging']['log_dir'])
    }

    log_dir = config['logging']['log_dir']
    os.makedirs(log_dir, exist_ok=True)
    filename = f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    save_path = os.path.join(log_dir, filename)
    plot_runs([run],save_path)


def plot_runs(run_list: list,save_path):
    """
    Plot multiple runs from a list of dicts with keys 'log_dir' and 'label'
    """
    plt.figure(figsize=(8, 4))


    # --- Loss Plot ---
    plt.subplot(1, 2, 1)
    for run in run_list:
        event_path = find_event_file(run['log_dir'])
        if event_path:
            steps, train_loss = extract_scalar_from_event(event_path, "Loss/Train")
            _, val_loss = extract_scalar_from_event(event_path, "Loss/Val")
            plt.plot(steps, train_loss, label=f"{run['label']} Train")
            if val_loss:
                plt.plot(steps, val_loss, label=f"{run['label']} Val")
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
            steps, train_acc = extract_scalar_from_event(event_path, "Accuracy/Train")
            _, val_acc = extract_scalar_from_event(event_path, "Accuracy/Val")
            plt.plot(steps, train_acc, label=f"{run['label']} Train")
            if val_acc:
                plt.plot(steps, val_acc, label=f"{run['label']} Val")
    plt.title("Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)  # 🔐 Save the plot

    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to config.yml")
    parser.add_argument("--runs_config", type=str, default=None, help="Path to model_runs.yml")
    args = parser.parse_args()

    if args.runs_config:
        runs = load_yaml(args.runs_config)["runs"]
        plot_runs(runs)
    elif args.config:
        config = load_yaml(args.config)
        plot_from_tensorboard(config)
    else:
        print("❌ Please provide either --config or --runs_config")
