# Executive Summary: Emotion Classification Experiments

This document summarizes a series of experiments aimed at improving audio-based emotion classification using latent space diagnostics, augmentation, domain adaptation, and metric learning techniques. The mclassifer was built using an open source transformer (Wav2Vec2, HuBERT, WavLM) with frozen or partially unfrozen layers along with a classifer head.

---

## 1. **Establishing the Baseline & Feature Bias Analysis**

**Goal:**  
Evaluate initial performance and identify inherent dataset biases.

**Methodology:**
- Training on RAVDESS + CREMA-D without augmentation or noise.
- Latent space visualization across CNN, encoder, MLP, and logits using t-SNE/UMAP.

**Findings:**
- **Dataset Bias:** CNN features clustered by dataset (CREMA-D vs. RAVDESS) instead of emotion (squares vs. circles).![[Pasted image 20250707210825.png]]
- **Classifier Performance:** Slow convergence, ~55% validation accuracy.
- **Limitation:** Poor separability, strong dataset dependence.
  ![[metrics_20250705_032406.png]]
---

## 2. **Robustness Through Augmentation & Domain Adaptation**
**Goal:**  
Improve generalization and reduce dataset-specific learning.
**Methodology:**
- **Data Augmentation:** 30% MUSAN noise, 30% time stretch, 50% gain, and center-cropped 2–3s random slices.
- **HuBERT Fine-Tuning:** Unfroze last 2 layers.
- **Domain-Adversarial Training (DANN):** GRL (λ = 1) added to enforce domain-invariant emotion features.
    
**Findings:**
- **Prediction Boost:** Accuracy jumped to ~70% in fewer epochs, driven mainly by HuBERT fine-tuning.![[Pasted image 20250706005100.png]]
- **Latent Spread & Domain Independence:** UMAP plots showed better emotion clustering and dataset mixing (TSNE at MLP Internal Layer).![[Pasted image 20250707173949.png]]
- **Plateau Effect:** Accuracy did not exceed 70% even with hyperparameter tuning.
  
**Limitations:**
- Domain-adversarial training showed visual but not performance improvements.
- Overfitting observed; mitigated in next stage using larger dropout.
    
## 3. **Latent Space Diagnostics with MMD**

**Goal:**  
Quantify separability and assess feature distribution overlaps.

**Methodology:**
- **Projection Tools:** t-SNE, UMAP for visualization; PCA (30D) to reduce feature dimensions.
- **Metric:** Maximum Mean Discrepancy (MMD) with RBF kernel.
- **Validation:** Shapiro-Wilk test confirmed non-Gaussianity → MMD valid.
![[Pasted image 20250707093212.png]]
**Findings:**
- **Strong Correlation:**
    - Spearman ρ = -0.7 between MMD and confusion rates:  
	    - A value of **0.00 (diagonal)** indicates that the distribution of a class's features is identical to itself, as expected. **Higher values** indicate that the two class distributions are **more different** (less similar or less overlapping) in the feature space. **Lower values** (closer to 0) indicate that the two class distributions are **more similar** (more overlapping).![[confusion_and_mmd.png]]
    
    - Centroid distance also inversely correlated with classification accuracy.
    -         → Closer distributions = higher confusion.
        ![[confusion_vs_latent_distance.png]]
- **Key Insight:** Latent feature separability might be a fundamental bottleneck?

**Limitations:**
- MMD is diagnostic, not directly optimizable.

---

## 4. **Improving Latent Separation via Triplet Loss**

**Goal:**  
Directly shape latent space by minimizing intra-class and maximizing inter-class distances.

**Motivation:**  
Previous stages revealed persistent latent overlap (especially in subtle emotion pairs). Black-box emergence of latent clusters was a bottleneck.

**Implementation:**
- Applied triplet loss to pooled HuBERT features.
- Also tested reversed audio as robustness probe.
![[Pasted image 20250707180152.png]]

**Findings:**
- **Improved Visual Clustering** (HuBERT encoder output):  
    → Better latent alignment but **no gain** in classification accuracy.
- **Accuracy Ceiling Persisted:**  
    Despite tighter clusters, accuracy remained ≤ 70%.
    ![[Pasted image 20250707211735.png]]
**Limitations & Future Directions:**
- Triplet loss alone insufficient; future work to investigate:
    - Better balance of multi-loss training (cross-entropy, adversarial, triplet).
    - Smarter triplet mining strategies.

---

## 5. **The 70% Accuracy Plateau: Root Causes & Next Steps**

### 🔍 Key Problem
Despite improvements, validation accuracy consistently plateaus at ~70%.  Explored 3 different models (Wav2Vec2, HuBERT, and WavLM), all exhibit bottleneck, with HuBERT resulting in the best accuracy. 
### 🔎 Likely Causes

- **Ambiguous Labels:** Emotion distinction (e.g., Sad vs. Fearful) is fuzzy, even for humans.
- **HuBERT Bottleneck:** Pretraining may not capture subtle emotional cues needed for fine-grained classification.
- **Dataset Limitations:** Insufficient clean, varied samples in some emotion classes.
- **Loss Conflicts:** Gradient interference between Cross-Entropy, DANN, and Triplet Loss may hamper optimization.

---

## 6. **Recommendations & Research Directions**

### 🧪 Data-Centric Improvements
- **Targeted Data Collection:**  
    Prioritize ambiguous pairs (Sad–Fearful, Happy–Neutral) via clearer protocols.
- **Advanced Augmentation:**  
    Use generative or noise-invariant augmentations to mimic real-world conditions.

### 🧠 Latent Space Optimization
- **Better Triplet Mining:**  
    Use batch-hard or semi-hard online mining for stronger training signals.
- **Hybrid Metric Losses:**  
    Combine Triplet (inter-class) and Center Loss (intra-class compactness) for better clustering.

### 🎧 Encoder Adaptation

- **Deeper HuBERT Fine-tuning:**  
    Unfreeze earlier layers for full encoder adaptation, though didn't see any improvement with more unfrozen layers. Try finetuning with LoRA?
- **Try Alternatives:**
    - **WavLM** or **data2vec** may offer better emotional feature disentanglement?
    - WavLM’s denoising and prediction objectives could help in noisy real-world conditions though preliminary data showed it also exhibited bottleneck.
    - Explore hybrid model (what architectures??)
        

---

## Final Takeaway

Although 70% accuracy represents respectival, latent space diagnostics clearly show that limitations in emotional feature separability—particularly for subtle or ambiguous pairs—are the key bottleneck. To go beyond this threshold, future efforts must focus on better encoding, smarter metric learning, and improved data curation.