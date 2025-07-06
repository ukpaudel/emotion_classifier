import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import imageio
from utils.emotion_labels import EMOTION_MAP

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
    print(f"Confusion matrix animation saved at {gif_path}")



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
    plt.show()


