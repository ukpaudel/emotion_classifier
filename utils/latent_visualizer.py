import os
import io
import PIL.Image
import torchvision
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # For heatmap visualization
from sklearn.manifold import TSNE
from umap import UMAP
from scipy.spatial.distance import cdist
import plotly.express as px
import scipy.stats # Not directly used for BC, but good to keep if other stats are needed

from utils.feature_store import feature_store
from utils.analyze_confusion_latent import analyze_confusion_and_latent
from utils.emotion_labels import EMOTION_MAP

# EMOTION_MAP should be defined or imported correctly. Assuming it is.
# EMOTION_MAP = {
#     0: "Neutral", 1: "Calm", 2: "Happy", 3: "Sad",
#     4: "Angry", 5: "Fearful", 6: "Disgust", 7: "Surprised"
# }

def register_hooks(model):
    def cnn_hook(module, input, output):
        features, _ = output
        pooled = features.mean(dim=1)
        for i in range(pooled.shape[0]):
            feature_store["cnn"].append(pooled[i].detach().cpu())

    def mlp_hook(module, input, output):
        # this is the penultimate features before final logits
        for i in range(output.shape[0]):
            feature_store["mlp"].append(output[i].detach().cpu())

    def logits_hook(module, input, output):
        # final classifier logits
        for i in range(output.shape[0]):
            feature_store["logits"].append(output[i].detach().cpu())

    if hasattr(model, 'encoder') and hasattr(model.encoder, 'feature_extractor'):
         model.encoder.feature_extractor.register_forward_hook(cnn_hook)
    else:
        print("Warning: model.encoder.feature_extractor not found. CNN features might not be collected.")

    model.classifier.mlp[0].register_forward_hook(mlp_hook)
    model.classifier.register_forward_hook(logits_hook)


def extract_features_for_visualization(model, val_loader, device, logger):
    feature_store["cnn"].clear()
    feature_store["encoder"].clear()
    feature_store["mlp"].clear()
    feature_store["logits"].clear()
    feature_store["labels"].clear()
    feature_store["dataset_ids"].clear()

    model.eval()
    with torch.no_grad():
        for batch_data in val_loader:
            if batch_data is None:
                continue
            waveforms, labels, lengths, dataset_ids = batch_data
            waveforms = waveforms.to(device)
            _ = model(waveforms, lengths) # This populates feature_store via hooks

            for i in range(labels.shape[0]):
                feature_store["labels"].append(labels[i].cpu())
                feature_store["dataset_ids"].append(dataset_ids[i])
    logger.info("Validation pass for feature extraction completed.")


def bhattacharyya_coefficient_gaussian(mu1, Sigma1, mu2, Sigma2, epsilon=1e-6):
    """
    Calculates the Bhattacharyya coefficient between two multivariate Gaussian distributions.

    Args:
        mu1 (np.array): Mean vector of the first Gaussian.
        Sigma1 (np.array): Covariance matrix of the first Gaussian.
        mu2 (np.array): Mean vector of the second Gaussian.
        Sigma2 (np.array): Covariance matrix of the second Gaussian.
        epsilon (float): Small value for regularization to prevent singular matrices.

    Returns:
        float: Bhattacharyya coefficient (0 to 1).
    """
    d = mu1.shape[0]
    Sigma1_reg = Sigma1 + epsilon * np.eye(d)
    Sigma2_reg = Sigma2 + epsilon * np.eye(d)

    Sigma_avg = (Sigma1_reg + Sigma2_reg) / 2

    try:
        inv_Sigma_avg = np.linalg.inv(Sigma_avg)
    except np.linalg.LinAlgError:
        print("Warning: Sigma_avg is singular even after regularization. Returning 0 (max dissimilarity).")
        return 0.0

    term1 = 0.125 * np.dot(np.dot((mu1 - mu2).T, inv_Sigma_avg), (mu1 - mu2))
    
    det_Sigma1 = np.linalg.det(Sigma1_reg)
    det_Sigma2 = np.linalg.det(Sigma2_reg)
    det_Sigma_avg = np.linalg.det(Sigma_avg)

    if det_Sigma1 <= 0 or det_Sigma2 <= 0 or det_Sigma_avg <= 0:
        # print(f"Warning: Non-positive determinant encountered: det_Sigma1={det_Sigma1}, det_Sigma2={det_Sigma2}, det_Sigma_avg={det_Sigma_avg}. Clamping to small positive.")
        det_Sigma1 = max(det_Sigma1, 1e-10)
        det_Sigma2 = max(det_Sigma2, 1e-10)
        det_Sigma_avg = max(det_Sigma_avg, 1e-10)

    term2 = 0.5 * np.log(det_Sigma_avg / np.sqrt(det_Sigma1 * det_Sigma2))
    
    distance = term1 + term2
    if not np.isfinite(distance):
        print(f"Warning: Bhattacharyya distance is not finite: {distance}. Returning 0.")
        return 0.0

    coefficient = np.exp(-distance)
    
    return np.clip(coefficient, 0.0, 1.0)


def calculate_bhattacharyya_matrix(mlp_feats, labels, logger):
    unique_emotions = sorted(list(np.unique(labels.numpy())))
    num_classes = len(unique_emotions)
    bhattacharyya_matrix = np.zeros((num_classes, num_classes))

    class_means = {}
    class_covs = {}

    for i, emotion_idx in enumerate(unique_emotions):
        class_data = mlp_feats[labels.numpy() == emotion_idx].numpy()
        if class_data.shape[0] < 2:
            logger.warning(f"Class '{EMOTION_MAP[emotion_idx]}' has fewer than 2 samples ({class_data.shape[0]}). Cannot compute proper covariance matrix. Using identity matrix with small value.")
            class_means[emotion_idx] = np.zeros(mlp_feats.shape[1])
            class_covs[emotion_idx] = np.eye(mlp_feats.shape[1]) * 1e-6
            continue

        class_means[emotion_idx] = np.mean(class_data, axis=0)
        cov_matrix = np.cov(class_data, rowvar=False)
        
        if cov_matrix.ndim == 0:
            cov_matrix = np.array([[cov_matrix.item()]])
        elif cov_matrix.ndim == 1:
            cov_matrix = np.diag(cov_matrix)

        class_covs[emotion_idx] = cov_matrix

    for i, e1 in enumerate(unique_emotions):
        for j, e2 in enumerate(unique_emotions):
            if e1 not in class_means or e2 not in class_means:
                bhattacharyya_matrix[i, j] = 0.0
                continue
            
            if i == j:
                # We will set this to NaN later for plotting, but keeping it as 1 for now if needed elsewhere
                bhattacharyya_matrix[i, j] = 1.0
            else:
                bc = bhattacharyya_coefficient_gaussian(
                    class_means[e1], class_covs[e1],
                    class_means[e2], class_covs[e2]
                )
                bhattacharyya_matrix[i, j] = bc
    
    return bhattacharyya_matrix, [EMOTION_MAP[idx] for idx in unique_emotions]


def plot_confusion_and_bhattacharyya(log_dir, mlp_feats, labels, logger):
    # --- Load and prepare Confusion Matrix ---
    cm_path = os.path.join(log_dir, "confusions_all_epochs.npy")
    
    if not os.path.exists(cm_path):
        logger.warning(f"Confusion matrix file not found at {cm_path}. Cannot plot confusion matrix.")
        # Fallback to only plotting Bhattacharyya if CM is missing
        bhattacharyya_matrix, emotion_labels = calculate_bhattacharyya_matrix(mlp_feats, labels, logger)
        
        # Set diagonal to NaN for better visualization
        np.fill_diagonal(bhattacharyya_matrix, np.nan) 

        plt.figure(figsize=(9, 8)) # Adjusted size for single plot
        sns.heatmap(
            bhattacharyya_matrix,
            annot=True,
            fmt=".2f",
            cmap="YlGnBu",
            xticklabels=emotion_labels,
            yticklabels=emotion_labels,
            linewidths=.5,
            linecolor='black',
            vmin=0, vmax=1 # Ensure consistent color range 0-1
        )
        plt.title("Bhattacharyya Coefficient Matrix (MLP Features)\n(1=Correlated, 0=Uncorrelated; Diagonal excluded)")
        plt.xlabel("Class")
        plt.ylabel("Class")
        plt.tight_layout()
        single_bhatt_path = os.path.join(log_dir, "bhattacharyya_coefficient_matrix_single.png")
        plt.savefig(single_bhatt_path)
        logger.info(f"Saved stand-alone Bhattacharyya Coefficient matrix to {single_bhatt_path}")
        plt.close()
        return


    cm_dict = np.load(cm_path, allow_pickle=True).item()
    final_epoch = max(cm_dict.keys())
    cm = cm_dict[final_epoch]
    cm_norm = cm / cm.sum(axis=1, keepdims=True) # Normalize rows to sum to 1

    # --- Calculate Bhattacharyya Coefficient Matrix ---
    bhattacharyya_matrix, emotion_labels = calculate_bhattacharyya_matrix(mlp_feats, labels, logger)

    # --- Plotting side-by-side ---
    fig, axes = plt.subplots(1, 2, figsize=(18, 8)) # 1 row, 2 columns

    # Plot Confusion Matrix (left subplot)
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=emotion_labels, # Use emotion_labels from Bhattacharyya for consistency
        yticklabels=emotion_labels,
        ax=axes[0],
        linewidths=.5,
        linecolor='black',
        vmin=0, vmax=1 # Normalize to 0-1 range for consistent coloring
    )
    axes[0].set_title(f"Confusion Matrix (Normalized Rows)\nEpoch {final_epoch}")
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")

    # Plot Bhattacharyya Coefficient Matrix (right subplot)
    # Set diagonal to NaN to exclude it from colormap scaling
    plot_bhattacharyya_matrix = np.copy(bhattacharyya_matrix)
    np.fill_diagonal(plot_bhattacharyya_matrix, np.nan)

    sns.heatmap(
        plot_bhattacharyya_matrix,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        xticklabels=emotion_labels,
        yticklabels=emotion_labels,
        ax=axes[1],
        linewidths=.5,
        linecolor='black',
        vmin=0, vmax=1 # Ensure consistent color range 0-1 for better resolution of smaller values
    )
    axes[1].set_title("Bhattacharyya Coefficient Matrix (MLP Features)\n(1=Correlated, 0=Uncorrelated; Diagonal excluded)")
    axes[1].set_xlabel("Class")
    axes[1].set_ylabel("Class")

    plt.tight_layout()
    combined_plot_path = os.path.join(log_dir, "confusion_and_bhattacharyya.png")
    plt.savefig(combined_plot_path)
    logger.info(f"Saved combined confusion and Bhattacharyya matrix plot to {combined_plot_path}")
    plt.close(fig)


def plot_latent_space(log_dir, logger):
    if len(feature_store["cnn"]) == 0:
        logger.warning("No features found to plot, skipping.")
        return

    cnn_feats = torch.stack(feature_store["cnn"], dim=0)
    if len(feature_store["encoder"]) > 0:
        encoder_feats = torch.stack(feature_store["encoder"], dim=0)
    else:
        logger.warning("feature_store['encoder'] is empty. Encoder plots will be skipped or may use placeholder.")
        encoder_feats = torch.empty(0)

    mlp_feats = torch.stack(feature_store["mlp"], dim=0)
    logits_feats = torch.stack(feature_store["logits"], dim=0)

    labels = torch.tensor([int(x.item()) for x in feature_store["labels"]])
    dataset_names = np.array(feature_store["dataset_ids"])
    N = labels.shape[0]

    unique_datasets = np.unique(dataset_names)
    marker_styles = ['o', 's', 'D', '^', 'P', 'X', '*', '+']
    marker_map = {ds: marker_styles[i % len(marker_styles)] for i, ds in enumerate(unique_datasets)}

    assert all(f.shape[0] == N for f in [cnn_feats, mlp_feats, logits_feats])
    if len(feature_store["encoder"]) > 0 and encoder_feats.shape[0] == N: # Assert only if encoder features are explicitly collected
        assert encoder_feats.shape[0] == N
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

    tsne_perplexity = min(30, N - 1) if N > 1 else 1
    tsne = TSNE(n_components=2, random_state=42, perplexity=tsne_perplexity)
    umap = UMAP(n_components=2, random_state=42)

    tsne_cnn, umap_cnn = np.zeros((N, 2)), np.zeros((N, 2))
    tsne_encoder, umap_encoder = np.zeros((N, 2)), np.zeros((N, 2))
    tsne_mlp, umap_mlp = np.zeros((N, 2)), np.zeros((N, 2))
    tsne_logits, umap_logits = np.zeros((N, 2)), np.zeros((N, 2))

    if N > 1 and tsne_perplexity > 0:
        tsne_cnn = tsne.fit_transform(cnn_feats.numpy())
        umap_cnn = umap.fit_transform(cnn_feats.numpy())

        if len(feature_store["encoder"]) > 0 and encoder_feats.shape[0] > 0:
            tsne_encoder = tsne.fit_transform(encoder_feats.numpy())
            umap_encoder = umap.fit_transform(encoder_feats.numpy())
        else:
            logger.info("Skipping Encoder t-SNE/UMAP due to missing or empty features.")

        tsne_mlp = tsne.fit_transform(mlp_feats.numpy())
        umap_mlp = umap.fit_transform(mlp_feats.numpy())

        tsne_logits = tsne.fit_transform(logits_feats.numpy())
        umap_logits = umap.fit_transform(logits_feats.numpy())
    else:
        logger.warning(f"Not enough samples (N={N}) for meaningful t-SNE/UMAP, skipping dimensionality reduction plots. Perplexity: {tsne_perplexity}")

    pairs = [
        (tsne_cnn, "CNN t-SNE"),
        (umap_cnn, "CNN UMAP"),
    ]
    if len(feature_store["encoder"]) > 0 and encoder_feats.shape[0] > 0:
        pairs.extend([
            (tsne_encoder, "Encoder t-SNE"),
            (umap_encoder, "Encoder UMAP"),
        ])
    pairs.extend([
        (tsne_mlp, "MLP Hidden t-SNE"),
        (umap_mlp, "MLP Hidden UMAP"),
        (tsne_logits, "Logits t-SNE"),
        (umap_logits, "Logits UMAP"),
    ])

    num_rows_for_plots = len(pairs) // 2 + (len(pairs) % 2 > 0)
    fig, axes = plt.subplots(num_rows_for_plots, 2, figsize=(18, num_rows_for_plots * 6))
    axes = axes.flatten()

    for ax_idx, (data, name) in enumerate(pairs):
        ax = axes[ax_idx]
        if data.shape[0] == 0 or N == 0: # Ensure data is not empty before plotting
            ax.set_title(f"{name} (No data)")
            continue

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

        centroids = {}
        for emotion in np.unique(labels.numpy()):
            ix = np.where(labels.numpy() == emotion)[0]
            if len(ix) > 0:
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
    
    logits_centroids = {}
    for emotion in np.unique(labels.numpy()):
        ix = np.where(labels.numpy() == emotion)[0]
        if len(ix) > 0:
            centroid = logits_feats.numpy()[ix].mean(axis=0)
            logits_centroids[emotion] = centroid
    analyze_confusion_and_latent(log_dir, logits_centroids, logger)

    if len(axes) > 0 and len(axes[0].get_legend_handles_labels()[0]) > 0:
        handles, legend_labels = axes[0].get_legend_handles_labels()
        by_label = dict(zip(legend_labels, handles))
        fig.legend(
            by_label.values(),
            by_label.keys(),
            loc='upper center', ncol=4, fontsize=8
        )
    else:
        logger.warning("No legend handles found, skipping legend for static latent plots.")

    plt.tight_layout()
    static_path = os.path.join(log_dir, "latent_spaces_static.png")
    plt.savefig(static_path)
    logger.info(f"Saved static latent plots to {static_path}")
    plt.close(fig) # Close the figure for static plots


    # Interactive plots
    if N > 1 and tsne_perplexity > 0:
        px_cnn = px.scatter(
            x=tsne_cnn[:, 0], y=tsne_cnn[:, 1],
            color=label_names,
            hover_data={"dataset": dataset_names},
            title="Interactive CNN t-SNE"
        )
        px_cnn.write_html(os.path.join(log_dir, "tsne_cnn_interactive.html"))
        logger.info("Saved interactive Plotly CNN t-SNE.")

        from sklearn.decomposition import PCA
        pca3d = PCA(n_components=3)
        mlp_3d = pca3d.fit_transform(mlp_feats.numpy())

        px_mlp3d = px.scatter_3d(
            x=mlp_3d[:, 0],
            y=mlp_3d[:, 1],
            z=mlp_3d[:, 2],
            color=label_names,
            hover_data={"dataset": dataset_names},
            title="Interactive 3D MLP Latent Space"
        )
        px_mlp3d.update_traces(marker=dict(size=3))
        px_mlp3d.write_html(os.path.join(log_dir, "mlp_latent_3d.html"))
        logger.info("Saved interactive 3D Plotly MLP latent space.")
    else:
        logger.warning("Skipping interactive Plotly plots due to insufficient samples.")

    # --- New Combined Plotting Function Call ---
    # This will handle loading the confusion matrix and plotting both side-by-side
    plot_confusion_and_bhattacharyya(log_dir, mlp_feats, labels, logger)
    # --- End of New Combined Plotting Function Call ---