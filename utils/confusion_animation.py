import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import imageio
import argparse
import yaml
from datetime import datetime
try:
    from utils.emotion_labels import EMOTION_MAP
except ImportError:
    from emotion_labels import EMOTION_MAP


def animate_confusion(log_dir):
    cm_dict = np.load(os.path.join(log_dir, "confusions_all_epochs.npy"), allow_pickle=True).item()
    frames = []
    emotion_labels = [EMOTION_MAP[i] for i in range(len(EMOTION_MAP))]
    for epoch, cm in cm_dict.items():
        plt.figure(figsize=(8,6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=emotion_labels,
            yticklabels=emotion_labels
        )
        plt.title(f"Confusion Matrix Epoch {epoch+1}")
        plt.xlabel("Predicted")
        plt.ylabel("True")

        frame_file = os.path.join(log_dir, f"cm_frame_{epoch}.png")
        plt.savefig(frame_file)
        plt.close()
        frames.append(imageio.v2.imread(frame_file))

    gif_path = os.path.join(log_dir, "confusion_animation.gif")
    imageio.mimsave(gif_path, frames, fps=2)
    print(f"✅ Confusion matrix animation saved at {gif_path}")

def plot_final_confusion(log_dir):
    cm_dict = np.load(os.path.join(log_dir, "confusions_all_epochs.npy"), allow_pickle=True).item()
    final_epoch = max(cm_dict.keys())
    cm = cm_dict[final_epoch]
    emotion_labels = [EMOTION_MAP[i] for i in range(len(EMOTION_MAP))]

    plt.figure(figsize=(10,8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=emotion_labels,
        yticklabels=emotion_labels
    )
    plt.title(f"Final Confusion Matrix Epoch {final_epoch+1}")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    save_path = os.path.join(log_dir, f"final_confusion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(save_path)
    print(f"✅ Final confusion matrix saved at {save_path}")
    plt.close()

def load_config(path="configs/config.yml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help="Base log directory for the run. If not provided, will read from configs/config.yml"
    )
    parser.add_argument(
        "--run_label",
        type=str,
        default=None,
        help="Optional run label; will use config if not provided"
    )
    args = parser.parse_args()

    if args.log_dir is None:
        config = load_config()
        args.log_dir = config['logging']['log_dir']
        if args.run_label is None:
            args.run_label = config['logging'].get('run_label', 'default')

    # Compose full path
    final_log_dir = os.path.join(args.log_dir, args.run_label or "default")
    if not os.path.exists(final_log_dir):
        raise FileNotFoundError(f"❌ Directory {final_log_dir} does not exist.")

    animate_confusion(final_log_dir)
    plot_final_confusion(final_log_dir)
