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
from scipy.stats import pearsonr, spearmanr # For correlation
from sklearn.decomposition import PCA # Import PCA

# --- IMPORTANT: These imports must be present in your actual script ---
from utils.feature_store import feature_store
from utils.analyze_confusion_latent import analyze_confusion_and_latent
from utils.emotion_labels import EMOTION_MAP
# --- End of important imports ---

# --- Constants for PCA ---
# Based on your logs, smallest class has 40 samples.
# PCA components must be less than the number of samples in the smallest class.
# Aim for a dimension that allows robust covariance estimation (e.g., < 40).
PCA_TARGET_DIM = 30 
# Only apply PCA if the original feature dimension is above this threshold.
# Logits are typically 8-dimensional, so this prevents PCA on them.
MIN_DIM_FOR_PCA = 64 
# -------------------------

def register_hooks(model):
    def cnn_hook(module, input, output):
        features, _ = output
        pooled = features.mean(dim=1)
        for i in range(pooled.shape[0]):
            feature_store["cnn"].append(pooled[i].detach().cpu())

    def encoder_general_hook(module, input, output):
        if isinstance(output, tuple):
            encoder_output = output[0]
        else:
            encoder_output = output

        if encoder_output.ndim == 3: # (batch, seq_len, features) like a transformer output
            pooled_encoder = encoder_output.mean(dim=1) # Average over sequence length
        elif encoder_output.ndim == 2: # (batch, features)
            pooled_encoder = encoder_output
        else:
            print(f"Warning: Encoder hook: Unexpected output dimension {encoder_output.ndim} for {module.__class__.__name__}. Skipping collection for this batch.")
            return

        for i in range(pooled_encoder.shape[0]):
            feature_store["encoder"].append(pooled_encoder[i].detach().cpu())


    def mlp_hook(module, input, output):
        for i in range(output.shape[0]):
            feature_store["mlp"].append(output[i].detach().cpu())

    def logits_hook(module, input, output):
        for i in range(output.shape[0]):
            feature_store["logits"].append(output[i].detach().cpu())

    if hasattr(model, 'encoder') and hasattr(model.encoder, 'feature_extractor'):
         model.encoder.feature_extractor.register_forward_hook(cnn_hook)
    else:
        print("Warning: model.encoder.feature_extractor not found. CNN features might not be collected.")

    if hasattr(model, 'encoder'):
        if hasattr(model.encoder, 'transformer'):
            if hasattr(model.encoder.transformer, 'encoder'):
                model.encoder.transformer.encoder.register_forward_hook(encoder_general_hook)
            else:
                 print("Warning: model.encoder.transformer.encoder not found. Consider other layers for encoder hook.")
        else:
            model.encoder.register_forward_hook(encoder_general_hook)
    else:
        print("Warning: model.encoder not found. Encoder features might not be collected.")

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

    if not feature_store["encoder"]:
        logger.warning("feature_store['encoder'] is empty after feature extraction. This means the encoder hook might not be correctly configured or reached during forward pass, or it captured no data.")


def bhattacharyya_coefficient_gaussian(mu1, Sigma1, mu2, Sigma2, epsilon=1e-7, logger=None, class_pair_name=""):
    d = mu1.shape[0]
    
    Sigma1_reg = Sigma1 + epsilon * np.eye(d)
    Sigma2_reg = Sigma2 + epsilon * np.eye(d)

    Sigma_avg = (Sigma1_reg + Sigma2_reg) / 2

    det_Sigma1_reg = np.linalg.det(Sigma1_reg)
    det_Sigma2_reg = np.linalg.det(Sigma2_reg)
    det_Sigma_avg = np.linalg.det(Sigma_avg)

    if logger:
        logger.debug(f"BC for {class_pair_name}: Initial dets: S1_reg={det_Sigma1_reg:.2e}, S2_reg={det_Sigma2_reg:.2e}, S_avg={det_Sigma_avg:.2e}")

    if det_Sigma1_reg <= 0: det_Sigma1_reg = max(det_Sigma1_reg, 1e-30)
    if det_Sigma2_reg <= 0: det_Sigma2_reg = max(det_Sigma2_reg, 1e-30)
    if det_Sigma_avg <= 0: det_Sigma_avg = max(det_Sigma_avg, 1e-30)

    try:
        inv_Sigma_avg = np.linalg.inv(Sigma_avg)
    except np.linalg.LinAlgError:
        if logger:
            logger.warning(f"BC for {class_pair_name}: Sigma_avg is singular even after regularization (epsilon={epsilon}). Returning 0.0.")
        return 0.0

    term1 = 0.125 * np.dot(np.dot((mu1 - mu2).T, inv_Sigma_avg), (mu1 - mu2))
    
    term2 = 0.5 * np.log(det_Sigma_avg / np.sqrt(det_Sigma1_reg * det_Sigma2_reg))
    
    distance = term1 + term2

    if logger:
        logger.debug(f"BC for {class_pair_name}: Term1={term1:.4f}, Term2={term2:.4f}, Distance={distance:.4f}")

    if not np.isfinite(distance):
        if logger:
            logger.warning(f"BC for {class_pair_name}: Bhattacharyya distance is not finite ({distance}). Returning 0.0.")
        return 0.0

    coefficient = np.exp(-distance)
    
    return np.clip(coefficient, 0.0, 1.0)


def calculate_bhattacharyya_matrix(features, labels, logger, feature_name="Features"):
    unique_emotions = sorted(list(np.unique(labels.numpy())))
    num_classes = len(unique_emotions)
    bhattacharyya_matrix = np.zeros((num_classes, num_classes))

    class_means = {}
    class_covs = {}

    feature_dim = features.shape[1]

    for i, emotion_idx in enumerate(unique_emotions):
        class_data = features[labels.numpy() == emotion_idx].numpy()
        num_samples_in_class = class_data.shape[0]

        logger.debug(f"Class '{EMOTION_MAP[emotion_idx]}' ({feature_name}): Samples={num_samples_in_class}, FeatureDim={feature_dim}")

        if num_samples_in_class <= feature_dim:
            logger.warning(f"Class '{EMOTION_MAP[emotion_idx]}' for {feature_name} has {num_samples_in_class} samples, which is <= feature dimension ({feature_dim}). Cannot compute full-rank covariance. Using identity matrix with small value.")
            class_means[emotion_idx] = np.zeros(feature_dim)
            class_covs[emotion_idx] = np.eye(feature_dim) * 1e-6
            continue

        class_means[emotion_idx] = np.mean(class_data, axis=0)
        cov_matrix = np.cov(class_data, rowvar=False)
        
        if cov_matrix.ndim == 0:
            cov_matrix = np.array([[cov_matrix.item()]])
        elif cov_matrix.ndim == 1:
            cov_matrix = np.diag(cov_matrix)
        elif cov_matrix.shape == ():
            cov_matrix = np.array([[cov_matrix.item()]])

        class_covs[emotion_idx] = cov_matrix

    for i, e1 in enumerate(unique_emotions):
        for j, e2 in enumerate(unique_emotions):
            class1_name = EMOTION_MAP[e1]
            class2_name = EMOTION_MAP[e2]

            if i == j:
                bhattacharyya_matrix[i, j] = 1.0
            else:
                if e1 not in class_means or e2 not in class_means:
                    logger.warning(f"Skipping BC for {class1_name}-{class2_name} ({feature_name}) due to missing means/covs (likely insufficient samples for one/both classes). Setting to 0.0.")
                    bhattacharyya_matrix[i, j] = 0.0
                    continue

                bc = bhattacharyya_coefficient_gaussian(
                    class_means[e1], class_covs[e1],
                    class_means[e2], class_covs[e2],
                    epsilon=1e-7,
                    logger=logger,
                    class_pair_name=f"{class1_name}-{class2_name} ({feature_name})"
                )
                bhattacharyya_matrix[i, j] = bc
    
    return bhattacharyya_matrix, [EMOTION_MAP[idx] for idx in unique_emotions]


def plot_combined_bhattacharyya_matrices(log_dir, feature_sets, labels, logger):
    num_plots = len(feature_sets)
    rows = int(np.ceil(num_plots / 2))
    cols = 2

    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 6 * rows))
    axes = axes.flatten()

    for i, (original_features, original_title_suffix) in enumerate(feature_sets):
        ax = axes[i]
        
        if original_features.shape[0] == 0:
            ax.set_title(f"Bhattacharyya Coeff. ({original_title_suffix})\n(No data available)")
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        current_features = original_features
        current_title_suffix = original_title_suffix
        
        # Apply PCA for high-dimensional features
        if original_features.shape[1] >= MIN_DIM_FOR_PCA:
            # Calculate actual number of samples in the smallest class
            unique_labels = np.unique(labels.numpy())
            min_samples_per_class = min([
                (labels.numpy() == l).sum() for l in unique_labels
            ])
            
            # Ensure PCA components are less than the smallest class size
            n_components_for_pca = min(PCA_TARGET_DIM, min_samples_per_class - 1)
            
            if n_components_for_pca < 1: # If only one sample or less per class after subsetting
                logger.warning(f"Not enough samples in any class ({min_samples_per_class}) to perform PCA for {original_title_suffix}. Skipping PCA.")
            else:
                try:
                    pca = PCA(n_components=n_components_for_pca, random_state=42)
                    current_features = torch.from_numpy(pca.fit_transform(original_features.numpy()))
                    current_title_suffix = f"{original_title_suffix} (PCA {n_components_for_pca}D)"
                    logger.info(f"Reduced {original_title_suffix} from {original_features.shape[1]}D to {n_components_for_pca}D for BC calculation.")
                except ValueError as e:
                    logger.error(f"Error applying PCA to {original_title_suffix}: {e}. Using original features.")
        else:
            logger.info(f"Skipping PCA for {original_title_suffix} as its dimension ({original_features.shape[1]}) is below threshold ({MIN_DIM_FOR_PCA}).")


        bhattacharyya_matrix, emotion_labels = calculate_bhattacharyya_matrix(current_features, labels, logger, feature_name=current_title_suffix)
        
        plot_bhattacharyya_matrix = np.copy(bhattacharyya_matrix)
        np.fill_diagonal(plot_bhattacharyya_matrix, np.nan)

        sns.heatmap(
            plot_bhattacharyya_matrix,
            annot=True,
            fmt=".2f",
            cmap="YlGnBu",
            xticklabels=emotion_labels,
            yticklabels=emotion_labels,
            ax=ax,
            linewidths=.5,
            linecolor='black',
            vmin=0
        )
        ax.set_title(f"Bhattacharyya Coeff. ({current_title_suffix})\n(1=Correlated, 0=Uncorrelated; Diagonal excluded)")
        ax.set_xlabel("Class")
        ax.set_ylabel("Class")

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    combined_bhatt_path = os.path.join(log_dir, "all_bhattacharyya_matrices.png")
    plt.savefig(combined_bhatt_path)
    logger.info(f"Saved all Bhattacharyya Coefficient matrices to {combined_bhatt_path}")
    plt.close(fig)


def plot_confusion_and_bhattacharyya(log_dir, mlp_feats, labels, logger):
    cm_path = os.path.join(log_dir, "confusions_all_epochs.npy")
    
    cm_exists = os.path.exists(cm_path)

    if not cm_exists:
        logger.warning(f"Confusion matrix file not found at {cm_path}. Cannot plot confusion matrix.")
        # Apply PCA to MLP features here as well if it's high dim
        current_mlp_feats = mlp_feats
        current_mlp_name = "MLP Features"
        if mlp_feats.shape[1] >= MIN_DIM_FOR_PCA:
             unique_labels = np.unique(labels.numpy())
             min_samples_per_class = min([(labels.numpy() == l).sum() for l in unique_labels])
             n_components_for_pca = min(PCA_TARGET_DIM, min_samples_per_class - 1)
             if n_components_for_pca >=1:
                try:
                    pca = PCA(n_components=n_components_for_pca, random_state=42)
                    current_mlp_feats = torch.from_numpy(pca.fit_transform(mlp_feats.numpy()))
                    current_mlp_name = f"MLP Features (PCA {n_components_for_pca}D)"
                    logger.info(f"Reduced MLP Features from {mlp_feats.shape[1]}D to {n_components_for_pca}D for BC calculation (standalone plot).")
                except ValueError as e:
                    logger.error(f"Error applying PCA to MLP Features (standalone plot): {e}. Using original features.")
             else:
                logger.warning(f"Not enough samples in any class ({min_samples_per_class}) to perform PCA for MLP Features (standalone plot). Skipping PCA.")

        bhattacharyya_matrix, emotion_labels = calculate_bhattacharyya_matrix(current_mlp_feats, labels, logger, feature_name=current_mlp_name)
        np.fill_diagonal(bhattacharyya_matrix, np.nan)

        plt.figure(figsize=(9, 8))
        sns.heatmap(
            bhattacharyya_matrix,
            annot=True,
            fmt=".2f",
            cmap="YlGnBu",
            xticklabels=emotion_labels,
            yticklabels=emotion_labels,
            linewidths=.5,
            linecolor='black',
            vmin=0
        )
        plt.title(f"Bhattacharyya Coefficient Matrix ({current_mlp_name})\n(1=Correlated, 0=Uncorrelated; Diagonal excluded)")
        plt.xlabel("Class")
        plt.ylabel("Class")
        plt.tight_layout()
        single_bhatt_path = os.path.join(log_dir, "bhattacharyya_coefficient_matrix_mlp_standalone.png")
        plt.savefig(single_bhatt_path)
        logger.info(f"Saved stand-alone MLP Bhattacharyya Coefficient matrix to {single_bhatt_path}")
        plt.close()
        return

    cm_dict = np.load(cm_path, allow_pickle=True).item()
    final_epoch = max(cm_dict.keys())
    cm = cm_dict[final_epoch]
    cm_norm = cm / cm.sum(axis=1, keepdims=True)

    # Apply PCA to MLP features for the combined plot
    current_mlp_feats = mlp_feats
    current_mlp_name = "MLP Features"
    if mlp_feats.shape[1] >= MIN_DIM_FOR_PCA:
        unique_labels = np.unique(labels.numpy())
        min_samples_per_class = min([(labels.numpy() == l).sum() for l in unique_labels])
        n_components_for_pca = min(PCA_TARGET_DIM, min_samples_per_class - 1)
        if n_components_for_pca >= 1:
            try:
                pca = PCA(n_components=n_components_for_pca, random_state=42)
                current_mlp_feats = torch.from_numpy(pca.fit_transform(mlp_feats.numpy()))
                current_mlp_name = f"MLP Features (PCA {n_components_for_pca}D)"
                logger.info(f"Reduced MLP Features from {mlp_feats.shape[1]}D to {n_components_for_pca}D for BC calculation (combined plot).")
            except ValueError as e:
                logger.error(f"Error applying PCA to MLP Features (combined plot): {e}. Using original features.")
        else:
            logger.warning(f"Not enough samples in any class ({min_samples_per_class}) to perform PCA for MLP Features (combined plot). Skipping PCA.")
    else:
        logger.info(f"Skipping PCA for MLP Features (combined plot) as its dimension ({mlp_feats.shape[1]}) is below threshold ({MIN_DIM_FOR_PCA}).")


    bhattacharyya_matrix_mlp, emotion_labels = calculate_bhattacharyya_matrix(current_mlp_feats, labels, logger, feature_name=current_mlp_name)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=emotion_labels,
        yticklabels=emotion_labels,
        ax=axes[0],
        linewidths=.5,
        linecolor='black',
        vmin=0, vmax=1.0
    )
    axes[0].set_title(f"Confusion Matrix (Normalized Rows)\nEpoch {final_epoch}")
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")

    plot_bhattacharyya_matrix = np.copy(bhattacharyya_matrix_mlp)
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
        vmin=0
    )
    axes[1].set_title(f"Bhattacharyya Coefficient Matrix ({current_mlp_name})\n(1=Correlated, 0=Uncorrelated; Diagonal excluded)")
    axes[1].set_xlabel("Class")
    axes[1].set_ylabel("Class")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    off_diag_cm_mask = ~np.eye(cm_norm.shape[0], dtype=bool)
    off_diag_cm = cm_norm[off_diag_cm_mask]

    off_diag_bhatt_mask = ~np.isnan(plot_bhattacharyya_matrix)
    off_diag_bhatt = plot_bhattacharyya_matrix[off_diag_bhatt_mask]

    min_len = min(len(off_diag_cm), len(off_diag_bhatt))
    off_diag_cm = off_diag_cm[:min_len]
    off_diag_bhatt = off_diag_bhatt[:min_len]

    correlation_text = ""
    if min_len > 1:
        try:
            pearson_corr, _ = pearsonr(off_diag_cm, off_diag_bhatt)
            spearman_corr, _ = spearmanr(off_diag_cm, off_diag_bhatt)
            
            correlation_text = (
                f"Correlation (Off-Diagonal):\n"
                f"  Pearson: {pearson_corr:.3f}\n"
                f"  Spearman: {spearman_corr:.3f}"
            )
            logger.info(f"Correlation between off-diagonal Normalized Confusion and Bhattacharyya Matrices:")
            logger.info(f"  Pearson Correlation: {pearson_corr:.4f}")
            logger.info(f"  Spearman's Rank Correlation: {spearman_corr:.4f}")

        except ValueError as e:
            correlation_text = f"Correlation: Error ({e})"
            logger.warning(f"Could not compute correlation: {e}. This might happen if all values are identical or constant in one of the arrays.")
    else:
        correlation_text = "Correlation: N/A (Insufficient data)"
        logger.warning("Not enough off-diagonal elements to compute correlation between matrices (min_len <= 1).")

    fig.text(0.5, 0.01, correlation_text, ha='center', va='bottom', fontsize=12,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=0.5, alpha=0.8))

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
        logger.warning("feature_store['encoder'] is empty. Encoder plots will not show data.")
        encoder_feats = torch.empty(0, cnn_feats.shape[1] if cnn_feats.shape[0] > 0 else 1) 

    mlp_feats = torch.stack(feature_store["mlp"], dim=0)
    logits_feats = torch.stack(feature_store["logits"], dim=0)

    labels = torch.tensor([int(x.item()) for x in feature_store["labels"]])
    dataset_names = np.array(feature_store["dataset_ids"])
    N = labels.shape[0]

    unique_datasets = np.unique(dataset_names)
    marker_styles = ['o', 's', 'D', '^', 'P', 'X', '*', '+']
    marker_map = {ds: marker_styles[i % len(marker_styles)] for i, ds in enumerate(unique_datasets)}

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
    tsne = TSNE(n_components=2, random_state=42, perplexity=tsne_perplexity, learning_rate='auto', init='pca', max_iter=1000)
    umap = UMAP(n_components=2, random_state=42)

    tsne_cnn, umap_cnn = np.empty((0, 2)), np.empty((0, 2))
    tsne_encoder, umap_encoder = np.empty((0, 2)), np.empty((0, 2))
    tsne_mlp, umap_mlp = np.empty((0, 2)), np.empty((0, 2))
    tsne_logits, umap_logits = np.empty((0, 2)), np.empty((0, 2))

    if N > 1 and tsne_perplexity > 0:
        if cnn_feats.shape[0] > 1:
            tsne_cnn = tsne.fit_transform(cnn_feats.numpy())
            umap_cnn = umap.fit_transform(cnn_feats.numpy())
        else:
            logger.warning("CNN features have insufficient samples for t-SNE/UMAP.")

        if encoder_feats.shape[0] > 1:
            tsne_encoder = tsne.fit_transform(encoder_feats.numpy())
            umap_encoder = umap.fit_transform(encoder_feats.numpy())
        else:
            logger.warning("Encoder features have insufficient samples for t-SNE/UMAP or are empty. This plot will be skipped or show 'No data'.")

        if mlp_feats.shape[0] > 1:
            tsne_mlp = tsne.fit_transform(mlp_feats.numpy())
            umap_mlp = umap.fit_transform(mlp_feats.numpy())
        else:
            logger.warning("MLP features have insufficient samples for t-SNE/UMAP.")

        if logits_feats.shape[0] > 1:
            tsne_logits = tsne.fit_transform(logits_feats.numpy())
            umap_logits = umap.fit_transform(logits_feats.numpy())
        else:
            logger.warning("Logits features have insufficient samples for t-SNE/UMAP.")
    else:
        logger.warning(f"Not enough samples (N={N}) for meaningful t-SNE/UMAP, skipping dimensionality reduction plots. Perplexity: {tsne_perplexity}")


    pairs = [
        (tsne_cnn, "CNN t-SNE"),
        (umap_cnn, "CNN UMAP"),
    ]
    if tsne_encoder.shape[0] > 0:
        pairs.extend([
            (tsne_encoder, "Encoder t-SNE"),
            (umap_encoder, "Encoder UMAP"),
        ])
    else:
        logger.warning("Encoder t-SNE/UMAP plots will be skipped because data is empty or insufficient for transformation.")

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
        if data.shape[0] == 0 or N == 0:
            ax.set_title(f"{name} (No data / Insufficient samples)")
            ax.set_xticks([])
            ax.set_yticks([])
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
        if data.shape[0] > 0:
            for emotion in np.unique(labels.numpy()):
                ix = np.where(labels.numpy() == emotion)[0]
                if len(ix) > 0 and data[ix].shape[0] > 0:
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
    if logits_feats.shape[0] > 0:
        for emotion in np.unique(labels.numpy()):
            ix = np.where(labels.numpy() == emotion)[0]
            if len(ix) > 0:
                centroid = logits_feats.numpy()[ix].mean(axis=0)
                logits_centroids[emotion] = centroid
        analyze_confusion_and_latent(log_dir, logits_centroids, logger)
    else:
        logger.warning("Logits features are empty, skipping analyze_confusion_and_latent.")

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
    plt.close(fig)


    # Interactive plots
    if N > 1 and tsne_perplexity > 0:
        if tsne_cnn.shape[0] > 0:
            px_cnn = px.scatter(
                x=tsne_cnn[:, 0], y=tsne_cnn[:, 1],
                color=label_names,
                hover_data={"dataset": dataset_names},
                title="Interactive CNN t-SNE"
            )
            px_cnn.write_html(os.path.join(log_dir, "tsne_cnn_interactive.html"))
            logger.info("Saved interactive Plotly CNN t-SNE.")

        if mlp_feats.shape[0] > 0:
            # PCA for 3D plot needs to be on original MLP features before any other reduction
            # This is specifically for the 3D interactive plot, not the BC calculation.
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

    if mlp_feats.shape[0] > 0:
        # Call for combined plot (which also handles PCA internally now)
        plot_confusion_and_bhattacharyya(log_dir, mlp_feats, labels, logger)
    else:
        logger.warning("Skipping combined confusion and Bhattacharyya plot due to empty MLP features.")

    all_feature_sets = []
    if cnn_feats.shape[0] > 0:
        all_feature_sets.append((cnn_feats, "CNN Features"))
    
    if encoder_feats.shape[0] > 0:
        all_feature_sets.append((encoder_feats, "Encoder Features"))
    else:
        logger.warning("Encoder Features will not be included in Bhattacharyya plots as they are empty.")
        
    if mlp_feats.shape[0] > 0:
        all_feature_sets.append((mlp_feats, "MLP Features"))
    if logits_feats.shape[0] > 0:
        all_feature_sets.append((logits_feats, "Logits Features"))

    if all_feature_sets:
        plot_combined_bhattacharyya_matrices(log_dir, all_feature_sets, labels, logger)
    else:
        logger.warning("No feature sets available to plot combined Bhattacharyya matrices.")