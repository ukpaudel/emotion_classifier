# 📊 Latent Space Visualization

This module allows you to visualize the latent representations of your emotion classifier across different layers (CNN, Encoder, MLP) using t-SNE and UMAP projections. You can explore the internal representations of the model after training, and relate them to confusion patterns.

---

## 🔧 How to Run

After you have trained a model, you can run:

```bash
python visualize_latents.py --config configs/config.yml --checkpoint path/to/checkpoint.pt
```

* `--config` : points to your YAML configuration
* `--checkpoint` : path to your saved checkpoint `.pt` file

This script will:

✅ load the model from checkpoint
✅ extract features from the **validation set**
✅ run t-SNE and UMAP on the features collected from:

* CNN layers
* Encoder outputs
* MLP classifier head
  ✅ save static latent plots and interactive HTML plots in the log directory.

---

## 🔎 How the Data is Processed

1. During a *validation* pass, the script hooks into the intermediate layers (CNN, Encoder, MLP) and saves pooled features for each sample, along with its true emotion label and dataset ID.
2. After feature extraction, it applies:

   * **t-SNE** for local cluster visualization
   * **UMAP** for global neighborhood visualization
3. Features are colored by emotion, and shaped by dataset, so you can see how well the emotions cluster across multiple datasets.
4. Centroids of each emotion cluster are marked and labeled, to summarize where the network places its “prototypes” of each class.
5. Finally, a confusion-vs-distance plot compares the classifier’s confusion matrix with the distances between these centroids, providing insight into which clusters are closer or more easily confused.

---

## 📈 How to Interpret the Results

* **t-SNE plots**: emphasize local similarity; good for seeing tight emotion clusters
* **UMAP plots**: preserve global topology; good for seeing relationships or transitions between emotions
* **Centroids**: show the *average* representation for each emotion in 2D — you can think of them as the “ideal” embedding
* **Confusion vs. Cluster Distance**: plots confusion rates against cluster distances, so you can see which classes are confused even though their latent representations are close or far apart.

---

## 💾 Outputs

* `runs/emotion_classifier_exp2/hubert_2MLP_0Enc_wnoisedata_cosinewrmst_D0p3/latent_spaces_static.png` — side-by-side t-SNE and UMAP static plots
* `runs/emotion_classifier_exp2/hubert_2MLP_0Enc_wnoisedata_cosinewrmst_D0p3/tsne_cnn_interactive.html` — interactive plotly HTML you can zoom and explore
* `runs/emotion_classifier_exp2/hubert_2MLP_0Enc_wnoisedata_cosinewrmst_D0p3/confusion_vs_distance.png` — confusion vs. cluster distance scatter plot
* All saved in the log directory of your experiment.

---

## 📌 Notes

* Visualization **does not require retraining**. It just loads your checkpoint and a validation loader.
* If you want to change color mappings or add new plots, edit `utils/latent_visualizer.py`.

Happy exploring!
