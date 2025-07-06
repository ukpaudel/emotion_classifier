import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from umap import UMAP
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
import plotly.express as px
from utils.feature_store import feature_store
from utils.analyze_confusion_latent import analyze_confusion_and_latent
from utils.emotion_labels import EMOTION_MAP



def register_hooks(model):
    def cnn_hook(module, input, output):
        features, _ = output
        pooled = features.mean(dim=1)
        for i in range(pooled.shape[0]):
            feature_store["cnn"].append(pooled[i].detach().cpu())

    def mlp_hook(module, input, output):
        for i in range(output.shape[0]):
            feature_store["mlp"].append(output[i].detach().cpu())

    model.encoder.feature_extractor.register_forward_hook(cnn_hook)
    model.classifier.register_forward_hook(mlp_hook)


def extract_features_for_visualization(model, val_loader, device, logger):
    feature_store["cnn"].clear()
    feature_store["encoder"].clear()
    feature_store["mlp"].clear()
    feature_store["labels"].clear()
    feature_store["dataset_ids"].clear()

    model.eval()
    with torch.no_grad():
        for batch_data in val_loader:
            if batch_data is None:
                continue
            waveforms, labels, lengths, dataset_ids = batch_data
            waveforms = waveforms.to(device)
            _ = model(waveforms, lengths)

            for i in range(labels.shape[0]):
                feature_store["labels"].append(labels[i].cpu())
                feature_store["dataset_ids"].append(dataset_ids[i])
    logger.info("Validation pass for feature extraction completed.")


def plot_latent_space(log_dir, logger):
    if len(feature_store["cnn"]) == 0:
        logger.warning("No features found to plot, skipping.")
        return

    cnn_feats = torch.stack(feature_store["cnn"], dim=0)
    encoder_feats = torch.stack(feature_store["encoder"], dim=0)
    mlp_feats = torch.stack(feature_store["mlp"], dim=0)

    labels = torch.tensor([int(x.item()) for x in feature_store["labels"]])
    dataset_names = np.array(feature_store["dataset_ids"])
    N = labels.shape[0]
    assert cnn_feats.shape[0] == N
    assert encoder_feats.shape[0] == N
    assert mlp_feats.shape[0] == N
    assert len(dataset_names) == N

    EMOTION_COLORS = {
        0: "#1f77b4",  # Neutral
        1: "#ff7f0e",  # Calm
        2: "#2ca02c",  # Happy
        3: "#d62728",  # Sad
        4: "#9467bd",  # Angry
        5: "#8c564b",  # Fearful
        6: "#e377c2",  # Disgust
        7: "#7f7f7f",  # Surprised
    }
    label_names = np.array([EMOTION_MAP[int(x)] for x in labels.numpy()])

    tsne = TSNE(n_components=2, random_state=42)
    umap = UMAP(n_components=2, random_state=42)

    tsne_cnn = tsne.fit_transform(cnn_feats.numpy())
    umap_cnn = umap.fit_transform(cnn_feats.numpy())
    tsne_encoder = tsne.fit_transform(encoder_feats.numpy())
    umap_encoder = umap.fit_transform(encoder_feats.numpy())
    tsne_mlp = tsne.fit_transform(mlp_feats.numpy())
    umap_mlp = umap.fit_transform(mlp_feats.numpy())

    pairs = [
        (tsne_cnn, "CNN t-SNE"),
        (umap_cnn, "CNN UMAP"),
        (tsne_encoder, "Encoder t-SNE"),
        (umap_encoder, "Encoder UMAP"),
        (tsne_mlp, "MLP t-SNE"),
        (umap_mlp, "MLP UMAP"),
    ]

    unique_datasets = np.unique(dataset_names)
    marker_styles = ['o', 's', 'D', '^', 'P', 'X', '*', '+']
    marker_map = {ds: marker_styles[i % len(marker_styles)] for i, ds in enumerate(unique_datasets)}

    fig, axes = plt.subplots(3, 2, figsize=(16, 20))
    for ax, (data, name) in zip(axes.flatten(), pairs):
        for emotion in np.unique(labels.numpy()):
            emotion_ix = np.where(labels.numpy() == emotion)[0]
            for ds in unique_datasets:
                ds_ix = np.where(dataset_names == ds)[0]
                final_ix = np.intersect1d(emotion_ix, ds_ix)
                if len(final_ix) > 0:
                    ax.scatter(
                        data[final_ix, 0],
                        data[final_ix, 1],
                        c=[EMOTION_COLORS[emotion]] * len(final_ix),
                        alpha=0.7,
                        marker=marker_map[ds],
                        label=f"{EMOTION_MAP[emotion]}-{ds}",
                        edgecolors='k', linewidths=0.5
                    )
        ax.set_title(name)

        # place text labels colored and larger
        centroids = {}
        for emotion in np.unique(labels.numpy()):
            ix = np.where(labels.numpy() == emotion)[0]
            centroid = data[ix].mean(axis=0)
            centroids[emotion] = centroid
            ax.text(
                centroid[0], centroid[1],
                EMOTION_MAP[int(emotion)],
                fontsize=14,
                color=EMOTION_COLORS[int(emotion)],
                weight="bold",
                bbox=dict(facecolor="white", alpha=0.6, boxstyle="round,pad=0.3")
            )
    #the centroids are from the last year i.e. MLP
    analyze_confusion_and_latent(log_dir, centroids, logger)
    

    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    by_label = dict(zip(legend_labels, handles))
    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc='upper center', ncol=4, fontsize=8
    )

    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "latent_spaces_static.png"))
    logger.info(f"Saved static latent plots to {log_dir}/latent_spaces_static.png")

    # Optional interactive
    px_cnn = px.scatter(
        x=tsne_cnn[:, 0], y=tsne_cnn[:, 1],
        color=label_names,
        hover_data={"dataset": dataset_names},
        title="Interactive CNN t-SNE"
    )
    px_cnn.write_html(os.path.join(log_dir, "tsne_cnn_interactive.html"))
    logger.info("Saved interactive plotly CNN t-SNE.")


