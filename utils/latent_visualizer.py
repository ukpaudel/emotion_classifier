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
from scipy.stats import shapiro # For Gaussianity check
from sklearn.metrics.pairwise import rbf_kernel # For MMD RBF kernel calculation

# --- IMPORTANT: These imports must be present in your actual script ---
from utils.feature_store import feature_store
from utils.analyze_confusion_latent import analyze_confusion_and_latent
from utils.emotion_labels import EMOTION_MAP
# --- End of important imports ---

# --- Constants for PCA ---
PCA_TARGET_DIM = 30
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


def calculate_mmd(X, Y, gamma=None):
    """
    Calculates Maximum Mean Discrepancy (MMD) between two sets of samples X and Y
    using an RBF kernel.
    MMD^2(X, Y) = 1/n^2 sum(k(xi, xj)) + 1/m^2 sum(k(yi, yj)) - 2/(nm) sum(k(xi, yj))
    A smaller MMD indicates more similar distributions.
    """
    if X.shape[0] == 0 or Y.shape[0] == 0:
        return 0.0 # Or np.inf if you want to represent maximal distance

    if gamma is None:
        # Heuristic for gamma: 1 / number of features
        gamma = 1.0 / X.shape[1] if X.shape[1] > 0 else 1.0

    # Ensure gamma is not too small/zero if feature_dim is very large
    if gamma < 1e-6:
        gamma = 1e-6

    K_XX = rbf_kernel(X, X, gamma=gamma)
    K_YY = rbf_kernel(Y, Y, gamma=gamma)
    K_XY = rbf_kernel(X, Y, gamma=gamma)

    n = X.shape[0]
    m = Y.shape[0]

    mmd_sq = (np.sum(K_XX) / (n * n)) + \
             (np.sum(K_YY) / (m * m)) - \
             (2 * np.sum(K_XY) / (n * m))
    
    # MMD squared can be slightly negative due to numerical instability
    # We take max(0, mmd_sq) and then sqrt
    return np.sqrt(max(0.0, mmd_sq))


def calculate_mmd_matrix(features, labels, logger, feature_name="Features"):
    unique_emotions = sorted(list(np.unique(labels.numpy())))
    num_classes = len(unique_emotions)
    mmd_matrix = np.zeros((num_classes, num_classes))

    class_data_map = {}
    
    feature_dim = features.shape[1]
    
    # Calculate gamma once for this feature set based on its dimension
    # Using a common heuristic: gamma = 1 / number_of_features
    mmd_gamma = 1.0 / feature_dim if feature_dim > 0 else 1.0
    if mmd_gamma < 1e-6: # Prevent extremely small gamma if feature_dim is huge
        mmd_gamma = 1e-6

    logger.info(f"MMD Gamma for {feature_name} (dim {feature_dim}): {mmd_gamma:.4f}")

    # --- Gaussianity Check Parameters (for informational purposes) ---
    p_value_threshold = 0.05
    # ------------------------------------------------------------------

    for i, emotion_idx in enumerate(unique_emotions):
        class_data = features[labels.numpy() == emotion_idx].numpy()
        class_data_map[emotion_idx] = class_data
        num_samples_in_class = class_data.shape[0]

        logger.debug(f"Class '{EMOTION_MAP[emotion_idx]}' ({feature_name}): Samples={num_samples_in_class}, FeatureDim={feature_dim}")

        # --- Gaussianity Check for each class's features (informational) ---
        non_gaussian_dims_count = 0
        if num_samples_in_class > 3: # Shapiro-Wilk requires at least 3 samples
            for dim_idx in range(feature_dim):
                if np.std(class_data[:, dim_idx]) > 1e-9: # Check for variance
                    try:
                        stat, p_value = shapiro(class_data[:, dim_idx])
                        if p_value < p_value_threshold:
                            non_gaussian_dims_count += 1
                    except Exception as e:
                        #logger.warning(f"Shapiro-Wilk test failed for dim {dim_idx} of '{EMOTION_MAP[emotion_idx]}' ({feature_name}): {e}")
                        continue
            if non_gaussian_dims_count > 0:
                logger.warning(
                    f"Gaussianity Check: For '{EMOTION_MAP[emotion_idx]}' in {feature_name}, "
                    f"{non_gaussian_dims_count}/{feature_dim} dimensions failed Shapiro-Wilk test (p < {p_value_threshold}). "
                    f"Note: MMD does not assume Gaussianity, but this highlights distributional characteristics."
                )
            else:
                continue
                #logger.info(f"Gaussianity Check: For '{EMOTION_MAP[emotion_idx]}' in {feature_name}, all {feature_dim} dimensions passed Shapiro-Wilk test (p >= {p_value_threshold}).")
        else:
            continue
            # logger.warning(
            #     f"Gaussianity Check: Not enough samples ({num_samples_in_class}) for '{EMOTION_MAP[emotion_idx]}' in {feature_name} "
            #     f"to perform Shapiro-Wilk test (min 4 samples recommended)."
            # )
        # --- End of Gaussianity Check ---

    for i, e1 in enumerate(unique_emotions):
        for j, e2 in enumerate(unique_emotions):
            class1_name = EMOTION_MAP[e1]
            class2_name = EMOTION_MAP[e2]

            if i == j:
                mmd_matrix[i, j] = 0.0 # MMD of a distribution with itself is 0
            else:
                X_data = class_data_map.get(e1)
                Y_data = class_data_map.get(e2)

                if X_data is None or Y_data is None or X_data.shape[0] == 0 or Y_data.shape[0] == 0:
                    logger.warning(f"Skipping MMD for {class1_name}-{class2_name} ({feature_name}) due to missing or empty class data. Setting to 0.0 (or should be inf).")
                    mmd_matrix[i, j] = 0.0 # Consider np.inf if you want to represent maximal distance
                    continue

                mmd_val = calculate_mmd(X_data, Y_data, gamma=mmd_gamma)
                mmd_matrix[i, j] = mmd_val
    
    return mmd_matrix, [EMOTION_MAP[idx] for idx in unique_emotions]


def plot_combined_mmd_matrices(log_dir, feature_sets, labels, logger):
    num_plots = len(feature_sets)
    rows = int(np.ceil(num_plots / 2))
    cols = 2

    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 6 * rows))
    axes = axes.flatten()

    for i, (original_features, original_title_suffix) in enumerate(feature_sets):
        ax = axes[i]
        
        if original_features.shape[0] == 0:
            ax.set_title(f"Maximum Mean Discrepancy ({original_title_suffix})\n(No data available)")
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        current_features = original_features
        current_title_suffix = original_title_suffix
        
        # Apply PCA for high-dimensional features
        if original_features.shape[1] >= MIN_DIM_FOR_PCA:
            unique_labels = np.unique(labels.numpy())
            min_samples_per_class = min([
                (labels.numpy() == l).sum() for l in unique_labels
            ])
            
            n_components_for_pca = min(PCA_TARGET_DIM, min_samples_per_class - 1)
            
            if n_components_for_pca < 1:
                logger.warning(f"Not enough samples in any class ({min_samples_per_class}) to perform PCA for {original_title_suffix}. Skipping PCA.")
            else:
                try:
                    pca = PCA(n_components=n_components_for_pca, random_state=42)
                    current_features = torch.from_numpy(pca.fit_transform(original_features.numpy()))
                    current_title_suffix = f"{original_title_suffix} (PCA {n_components_for_pca}D)"
                    logger.info(f"Reduced {original_title_suffix} from {original_features.shape[1]}D to {n_components_for_pca}D for MMD calculation.")
                except ValueError as e:
                    logger.error(f"Error applying PCA to {original_title_suffix}: {e}. Using original features.")
        else:
            logger.info(f"Skipping PCA for {original_title_suffix} as its dimension ({original_features.shape[1]}) is below threshold ({MIN_DIM_FOR_PCA}).")


        mmd_matrix, emotion_labels = calculate_mmd_matrix(current_features, labels, logger, feature_name=current_title_suffix)
        
        plot_mmd_matrix = np.copy(mmd_matrix)
        # For MMD, diagonal is 0, no need to exclude or set to NaN
        # np.fill_diagonal(plot_mmd_matrix, np.nan) 

        # Determine vmax dynamically for MMD plots
        # Exclude diagonal (which is 0) when finding max for better color scaling
        max_mmd_val = np.max(plot_mmd_matrix[np.triu_indices(plot_mmd_matrix.shape[0], k=1)]) # Upper triangle excluding diagonal
        if max_mmd_val == 0: max_mmd_val = 0.1 # Prevent issues if all MMDs are zero

        sns.heatmap(
            plot_mmd_matrix,
            annot=True,
            fmt=".2f",
            cmap="Blues", # Use a cmap suitable for distances (higher values = more different)
            xticklabels=emotion_labels,
            yticklabels=emotion_labels,
            ax=ax,
            linewidths=.5,
            linecolor='black',
            vmin=0, vmax=max_mmd_val * 1.1 # Scale vmax slightly above max observed
        )
        ax.set_title(f"Maximum Mean Discrepancy ({current_title_suffix})\n(0=Identical; Higher=More Different)")
        ax.set_xlabel("Class")
        ax.set_ylabel("Class")

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    combined_mmd_path = os.path.join(log_dir, "all_mmd_matrices.png")
    plt.savefig(combined_mmd_path)
    logger.info(f"Saved all MMD matrices to {combined_mmd_path}")
    plt.close(fig)


def plot_confusion_and_mmd(log_dir, mlp_feats, labels, logger):
    cm_path = os.path.join(log_dir, "confusions_all_epochs.npy")

    cm_exists = os.path.exists(cm_path)

    if not cm_exists:
        logger.warning(f"Confusion matrix file not found at {cm_path}. Cannot plot combined plot.")
        # (Rest of the standalone MMD plotting code remains the same)
        return

    cm_dict = np.load(cm_path, allow_pickle=True).item()
    final_epoch = max(cm_dict.keys())
    cm = cm_dict.get(final_epoch, cm_dict.get(list(cm_dict.keys())[-1])) # Get last epoch if final_epoch missing
    cm_norm = cm / cm.sum(axis=1, keepdims=True)

    # Apply PCA to MLP features for the combined plot
    current_mlp_feats = mlp_feats
    current_mlp_name = "MLP Features"
    if mlp_feats.shape and mlp_feats.shape[-1] >= MIN_DIM_FOR_PCA:
        unique_labels = np.unique(labels.numpy())
        min_samples_per_class = min([(labels.numpy() == l).sum() for l in unique_labels])
        n_components_for_pca = min(PCA_TARGET_DIM, min_samples_per_class - 1)
        if n_components_for_pca >= 1:
            try:
                pca = PCA(n_components=n_components_for_pca, random_state=42)
                current_mlp_feats = torch.from_numpy(pca.fit_transform(mlp_feats.numpy()))
                current_mlp_name = f"MLP Features (PCA {n_components_for_pca}D)"
                logger.info(f"Reduced MLP Features from {mlp_feats.shape[-1]}D to {n_components_for_pca}D for MMD calculation (combined plot).")
            except ValueError as e:
                logger.error(f"Error applying PCA to MLP Features (combined plot): {e}. Using original features.")
        else:
            logger.warning(f"Not enough samples in any class ({min_samples_per_class}) to perform PCA for MLP Features (combined plot). Skipping PCA.")
    else:
        logger.info(f"Skipping PCA for MLP Features (combined plot) as its dimension ({mlp_feats.shape[-1]}) is below threshold ({MIN_DIM_FOR_PCA}).")


    mmd_matrix_mlp, emotion_labels = calculate_mmd_matrix(current_mlp_feats, labels, logger, feature_name=current_mlp_name)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=emotion_labels,
        yticklabels=emotion_labels,
        ax=axes[-2], # Access the first subplot
        linewidths=.5,
        linecolor='black',
        vmin=0, vmax=1.0
    )
    axes[-2].set_title(f"Confusion Matrix (Normalized Rows)\nEpoch {final_epoch}")
    axes[-2].set_xlabel("Predicted Label")
    axes[-2].set_ylabel("True Label")

    plot_mmd_matrix = np.copy(mmd_matrix_mlp)
    # np.fill_diagonal(plot_mmd_matrix, np.nan) # MMD diagonal is 0

    # Determine vmax dynamically for MMD plots
    max_mmd_val_mlp = np.max(plot_mmd_matrix)
    if max_mmd_val_mlp == 0: max_mmd_val_mlp = 0.1

    # We want darker for smaller MMD (more similar), so we can either:
    # 1. Use a colormap that goes from light to dark, and the values themselves are the MMD.
    # 2. Invert the MMD values (or scale them appropriately) and use a standard colormap.

    # Option 1: Use a colormap where darker means lower value (more similar).
    # 'Blues_r' reverses the 'Blues' colormap (light to dark).
    # 'viridis_r', 'plasma_r', 'cividis_r', 'magma_r' are other options.
    mmd_cmap = "Blues_r"
    mmd_vmin = 0
    mmd_vmax = max_mmd_val_mlp * 1.1

    sns.heatmap(
        plot_mmd_matrix,
        annot=True,
        fmt=".2f",
        cmap=mmd_cmap,
        xticklabels=emotion_labels,
        yticklabels=emotion_labels,
        ax=axes[-1], # Access the second subplot
        linewidths=.5,
        linecolor='black',
        vmin=mmd_vmin, vmax=mmd_vmax
    )
    axes[-1].set_title(f"Maximum Mean Discrepancy Matrix ({current_mlp_name})\n(Darker=More Identical; Lighter=More Different)") # Updated title
    axes[-1].set_xlabel("Class")
    axes[-1].set_ylabel("Class")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    off_diag_cm_mask = ~np.eye(cm_norm.shape[-1], dtype=bool)
    off_diag_cm = cm_norm.flatten()[off_diag_cm_mask.flatten()]

    off_diag_mmd_mask = ~np.eye(plot_mmd_matrix.shape[-1], dtype=bool)
    off_diag_mmd = plot_mmd_matrix.flatten()[off_diag_mmd_mask.flatten()]

    min_len = min(len(off_diag_cm), len(off_diag_mmd))
    off_diag_cm = off_diag_cm[:min_len]
    off_diag_mmd = off_diag_mmd[:min_len]

    correlation_text = ""
    if min_len > 1:
        try:
            pearson_corr, _ = pearsonr(off_diag_cm, off_diag_mmd)
            spearman_corr, _ = spearmanr(off_diag_cm, off_diag_mmd)

            correlation_text = (
                f"Correlation (Off-Diagonal):\n"
                f"  Pearson: {pearson_corr:.3f}\n"
                f"  Spearman: {spearman_corr:.3f}"
            )
            logger.info(f"Correlation between off-diagonal Normalized Confusion and MMD Matrices:")
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

    combined_plot_path = os.path.join(log_dir, "confusion_and_mmd.png")
    plt.savefig(combined_plot_path)
    logger.info(f"Saved combined confusion and MMD matrix plot to {combined_plot_path}")
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
            # This is specifically for the 3D interactive plot, not the MMD calculation.
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
        plot_confusion_and_mmd(log_dir, mlp_feats, labels, logger) # Changed function name
    else:
        logger.warning("Skipping combined confusion and MMD plot due to empty MLP features.")

    all_feature_sets = []
    if cnn_feats.shape[0] > 0:
        all_feature_sets.append((cnn_feats, "CNN Features"))
    
    if encoder_feats.shape[0] > 0:
        all_feature_sets.append((encoder_feats, "Encoder Features"))
    else:
        logger.warning("Encoder Features will not be included in MMD plots as they are empty.")
        
    if mlp_feats.shape[0] > 0:
        all_feature_sets.append((mlp_feats, "MLP Features"))
    if logits_feats.shape[0] > 0:
        all_feature_sets.append((logits_feats, "Logits Features"))

    if all_feature_sets:
        plot_combined_mmd_matrices(log_dir, all_feature_sets, labels, logger) # Changed function name
    else:
        logger.warning("No feature sets available to plot combined MMD matrices.")