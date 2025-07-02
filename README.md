# Summary

**emotion_classifier** is a modular, production-ready Python framework for speech emotion classification. It leverages powerful self-supervised speech encoders (like Wav2Vec2 or HuBERT) with an attention-based classifier on top, allowing easy transfer learning and fine-tuning for emotion recognition tasks.

It is designed to be configuration-driven, making training, evaluation, and experimentation easy and reproducible.

---

## 🚀 Features

- 🔌 Plug-and-play support for self-supervised encoders (tested with HuBERT, Wav2Vec2)
- ❄️ Encoder freezing or selective fine-tuning of the last *N* layers (`unfreeze_last_n_layers`)
- 🧩 Dynamic masking support for variable-length audio
- 🗂️ Extensible to new datasets — just add a new dataset loader like `ravdess_dataset.py`
- 🗂️ Noise augmentation datasets to train the audio with real world noise (MUSAN)  
- 🗂️ Training data augmentation with Gain (volume adjustement), Gaussian Noise, Musan Noise 
- 💾 Pretrained weight loading with `pretrained_weights.enabled: true`
- 📝 TensorBoard integration and logging for visualization
- 🔄 Checkpoint-based resume functionality
- 🎯 Supports single-file inference and real-time emotion detection with ASR
- 🐛 Automatic saving of misclassified audio samples for inspection 
- 🛠️ Configurable entirely through `configs/config.yml`

-----

# Data Handling and Transformation

This document outlines how audio data from various sources (e.g., RAVDESS, CREMA-D) is loaded, processed, split, and augmented within this project.

## 1\. Data Sources

This project supports loading data from multiple emotion speech datasets. Currently, the primary datasets integrated are:

  * **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song):** Contains emotional speech and song from 24 professional actors.
  * **CREMA-D (Crowd-sourced Emotional Multimodal Actors Dataset):** Features emotional speech from 91 actors across various demographic groups.

## 2\. Noise Augmentation Data

For robust model training, audio augmentation with background noise is applied. The noise source used is:

  * **MUSAN (Acoustic Noise, Speech, and Music):** This dataset comprises a large collection of diverse audio. For noise augmentation, specific categories from MUSAN are utilized:

      * **Noise:** Environmental and everyday sounds.
      * **Speech:** Recordings of spoken dialogue.
      * **Music:** Various genres of music.

    **Reference:** F. S. Lim, R. K. M. Ko, J. S. P. Chen, Y. C. Pang, and P. S. Lee, "MUSAN: An open source dataset for music, speech, and noise," in *Proceedings of the 23rd ACM international conference on Multimedia*, 2015, pp. 1159-1160.
    **Dataset Link:** [Kaggle MUSAN Dataset](https://www.kaggle.com/code/kerneler/starter-musan-noise-b2c57001-3/input) (assuming this link refers to a source from which the MUSAN data can be downloaded or accessed).

    The `noise_dir` specified in the configuration (e.g., `config['data']['noise_dir']`) should point to the root directory where the MUSAN dataset (or at least its 'noise', 'speech', 'music' subdirectories) is located.
    user needs to manually download the file and store it in the directory data/noise_musan/noise.
    
## 3\. Data Loading and Preprocessing (`Dataset` Classes)

Each raw audio dataset (e.g., RAVDESS, CREMA-D) is managed by its own dedicated PyTorch `Dataset` class (e.g., `RAVDESSDataset`, `CREMADataset`). These classes handle:

  * **File Collection:** Scanning the specified `data_dir` to identify and list all valid audio files.
  * **Metadata Extraction:** Parsing filenames to extract relevant metadata such as actor IDs, emotional labels, etc.
  * **Per-Sample Actor IDs:** Crucially, each dataset class maintains a list of `actor_ids` where each ID corresponds to a specific audio file at the same index. This per-sample mapping is vital for ensuring speaker-independent data splits.
  * **Conditional Augmentations:** The `Dataset` class constructors accept an `is_train` boolean flag and a `noise_dir` path.
      * If `is_train` is `True`, augmentations (like adding noise from `noise_dir`, shifting, pitching, etc.) are applied to the audio on-the-fly during data loading (`__getitem__`).
      * If `is_train` is `False`, no augmentations are applied, ensuring a clean and consistent validation set.
  * **Audio Loading and Transformation:** When `__getitem__` is called, the audio file is loaded, resampled (if necessary), and converted into a suitable format (e.g., PyTorch tensor). Any specified augmentations are then applied.

## 4\. Data Augmentation Strategy

This dataset implementation employs **on-the-fly data augmentation** during training. This means that for each audio file retrieved during a training epoch, random transformations are applied to the waveform.

**Key Principles:**

  * **Speaker Independence:** Augmentations like Pitch Shift, Time Stretch, and Gain help to make the model robust to speaker-specific vocal characteristics (e.g., natural pitch, speaking rate, loudness), encouraging it to learn emotion-specific features that generalize across different voices.
  * **Increased Data Diversity:** By randomly transforming samples, the effective size and variability of the training data are significantly increased without requiring additional storage.
  * **Regularization:** This acts as a powerful regularization technique, preventing the model from overfitting to the exact characteristics of the original, unaugmented training samples.
  * **On-the-Fly Processing:** Transformations are applied within the `__getitem__` method of the PyTorch Dataset. This is crucial because:
      * It ensures a new, random set of augmentation parameters is applied each time a sample is requested (e.g., across different epochs or batches).
      * It leverages PyTorch's `DataLoader` `num_workers` to perform these computationally intensive operations on the CPU in parallel, preventing the GPU from becoming a bottleneck and speeding up training.
  * **Training Only:** Augmentations are strictly applied **ONLY** when the `is_train` flag is set to `True` (i.e., for the training dataset). The validation and test datasets remain untouched and reflect the original, unaugmented data. This ensures an unbiased and realistic evaluation of the model's generalization performance on unseen data.

**Applied Augmentations:**

  * **Random Noise:** Adds a tiny amount of noise, typically sourced from the MUSAN dataset as described in Section 2, to the waveform.
  * **Pitch Shift:** Randomly shifts the pitch up or down by a few semitones.
  * **Time Stretch:** Randomly changes the playback speed of the audio.
  * **Gain:** Randomly adjusts the overall volume (loudness) of the audio.

Each of these augmentations is applied probabilistically (e.g., typically with a 50% chance) to further diversify the training experience for the model.

## 5\. Data Splitting and Augmentation Strategy (`create_dataloaders` function)

The `create_dataloaders` function orchestrates the loading, splitting, and merging of multiple datasets to produce the final training and validation data loaders.

### Speaker-Independent Splitting (Split-Then-Merge Approach)

To ensure that the model does not train and validate on the same speaker's voice, a **speaker-independent split** is performed. The strategy employed is a "split-then-merge" approach:

1.  **Individual Dataset Splitting:** For each configured dataset (e.g., RAVDESS, CREMA-D):
      * The full dataset is first instantiated (temporarily, without train-specific augmentations).
      * `sklearn.model_selection.GroupShuffleSplit` is used to split the dataset into training and validation indices based on its `actor_ids`. This guarantees that each actor's recordings are entirely in either the training set or the validation set for that specific dataset.
2.  **Train/Validation Dataset Instantiation:**
      * For the training portion of the split, a *new* instance of the dataset class is created with `is_train=True` and the `global_noise_dir` provided from the configuration. A `Subset` is then created using the training indices.
      * Similarly, for the validation portion, a *new* instance is created with `is_train=False` and `noise_dir=None` (no augmentations for validation), and a `Subset` is created using the validation indices.
3.  **Merging Splits:** All individual training `Subset`s (e.g., RAVDESS training subset, CREMA-D training subset) are combined using `torch.utils.data.ConcatDataset` to form the final `train_dataset`. The same process is applied to create the `val_dataset`.

**Implication of "Split-Then-Merge":** This method ensures speaker independence **within each original dataset**. For example, a RAVDESS actor will not appear in both the RAVDESS train split and the RAVDESS validation split. Given that actors in RAVDESS and CREMA-D are distinct individuals, this approach effectively provides global speaker independence for the combined dataset in this context.

## 6\. DataLoaders

Finally, `torch.utils.data.DataLoader` instances are created for both the `train_dataset` and `val_dataset`:

  * **`train_loader`:** `shuffle=True` to randomize sample order in each epoch. Uses a custom `collate_fn` to handle variable-length audio sequences (e.g., padding).
  * **`val_loader`:** `shuffle=False` as shuffling is not needed for validation. Also uses the custom `collate_fn`.

-----

## 7\. Install Datasets

To use this project, you need to download and correctly place the raw audio datasets:

  * **RAVDESS:**

      * **Download from:** [Kaggle RAVDESS Emotional Speech Audio](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)
      * **Extraction Path:** After downloading and extracting, ensure the directory structure looks like this relative to your project root (adjusting `data/ravdess/data` based on your `config.yml`):
        ```
        your_project/
        ├── data/
        │   └── ravdess/
        │       └── data/
        │           ├── Actor_01/
        │           │   ├── 03-01-01-01-01-01-01.wav
        │           │   └── ...
        │           ├── Actor_02/
        │           │   ├── ...
        │           └── ...
        └── ...
        ```
        (i.e., `data/ravdess/data` should contain folders named `Actor_01`, `Actor_02`, ..., `Actor_24`).

  * **CREMA-D:**

      * **Download from:**  [CREMA-D Dataset](https://github.com/CheyneyComputerScience/CREMA-D/tree/master/AudioWAV) (or an equivalent source)
      * **Extraction Path:** Ensure the `data_dir` specified in your `config.yml` points to the correct location of its audio files.

  * **MUSAN (Noise):**

      * **Download from:** [Kaggle MUSAN Dataset](https://www.kaggle.com/code/kerneler/starter-musan-noise-b2c57001-3/input) (or an equivalent source)
      * **Extraction Path:** Extract the dataset such that the `noise_dir` in your `config.yml` points to the folder containing the 'noise', 'speech', and 'music' subdirectories from MUSAN. For example, if your `noise_dir` is `data/audio_noise_samples`, then:
        ```
        your_project/
        ├── data/
        │   └── audio_noise_samples/
        │       ├── noise/
        │       │   ├── ... .wav
        │       ├── speech/
        │       │   ├── ... .wav
        │       └── music/
        │           ├── ... .wav
        └── ...
        ```

## \8. Folder Structure

```text
emotion_classifier/
├── configs/
│   ├── config.yml
│   ├── models_runs.yml
│   └── inference_config.yml
│
├── data/
│   ├── ravdess_dataset.py
│   ├── cremad_dataset.py
│   ├── dataloader.py
│   ├── collate.py
│   ├── ravdess/
│   │   └── data/
│   └── CREMA-D/
│       └── data/
│
├── models/
│   ├── attention_classifier.py
│   └── emotion_model.py
│
├── runs/
│   ├── emotion_classifier_v#/
│   │   └── run_label/  # stored based on config file logging.run_label
│   │       ├── checkpoint.pt  # trained weights
│   │       └── events.out.tfevents  # tensorboard logs
│   ├── example_comparisons.png
│   ├── encoder_loader.py
│   ├── run_tracker.py
│   ├── model_utils.py
│   └── tensorboard_plot_utils.py
│
├── train/
│   └── trainer.py
│
├── utils/
│   ├── config.py
│   ├── logger.py
│   ├── encoder_loader.py
│   ├── run_tracker.py
│   ├── model_utils.py
│   └── tensorboard_plot_utils.py
│
├── inference/
│   └── realtime_inference_withASR.py
│   └── inference.py
│
├── run_experiment.py
├── setup.py
├── README.md
└── ...
```


## \9. Architecture
```
               ┌───────────────────────────────┐
               │         Waveform Input        │
               └───────────────────────────────┘
                              │
                              ▼
               ┌───────────────────────────────┐
               │  Self-Supervised Encoder      │
               │   (Wav2Vec2, HuBERT, WavLM)   │
               └───────────────────────────────┘
                              │
                              ▼
               ┌───────────────────────────────┐
               │    Variable-Length Features   │
               │  (sequence of 768-dim tokens) │
               └───────────────────────────────┘
                              │
                       (with optional mask)
                              │
                              ▼
               ┌───────────────────────────────┐
               │     Attention Pooling         │
               │   (weighted average pooling)  │
               └───────────────────────────────┘
                              │
                              ▼
               ┌───────────────────────────────┐
               │  Feedforward Classifier (MLP) │
               │    (with Dropout + GELU)      │
               └───────────────────────────────┘
                              │
                              ▼
               ┌───────────────────────────────┐
               │       Emotion Logits          │
               │      (one of 8 classes)       │
               └───────────────────────────────┘


```
## \10. ⚙️ Installation

```bash
git clone https://github.com/ukpaudel/emotion_classifier.git
cd emotion_classifier
pip install -e .

```

## 11\. Usage

### Training

```bash
cd emotion_classifier
python .\run_experiment.py
```

This command will execute the training process, reading parameters such as encoder type, dataset paths, hyperparameters, output directories, and resume/pretrained settings from `configs/config.yml`.


### Visualize Comparisons Between Different Experiments

To visualize TensorBoard logs, navigate to your project root in the bash terminal and run:

```bash
tensorboard --logdir=runs
```

Then, open your web browser and go to `http://localhost:6006/`.

### Inference

For real-time emotion and ASR detection from a live microphone feed:
Navigate to the project root:

```bash
cd emotion_classifier
```

Then run:

```bash
python .\inference\realtime_inference_withASR.py
```

**Note:** This real-time inference script processes audio in **5-second windows** for both ASR transcription and emotion classification. Emotion predictions are based on the primary emotion detected within this window, providing aggregated results every 5 seconds.

For single audio file inference (uses preloaded example files in the root directory):

```bash
python .\inference\inference.py
```

## \12. Results

Comparison of Different Trainings (see ```runs/example_comparisons.png``` for details):

- Achieved up to ~80% classification accuracy using the RAVDESS dataset alone (though there are concern about speaker independence in this dataset, i.e. it is learning features from the speaker+emotion).

- Adding the CREMA-D dataset and addressing speaker independence reduced classification accuracy to around 67%.

- Unfreezing the encoder layers gave inconsistent results. For the RAVDESS dataset, unfreezing improved performance, but I suspect the small validation dataset may have caused overfitting.

- Wav2Vec2 showed poorer performance overall the HuBERT.

- HuBERT achieved higher classification accuracy and converged in fewer epochs when trained with frozen encoder layers on the RAVDESS dataset.

![confusion_animation](https://github.com/user-attachments/assets/4adf6350-f210-4f7b-bf4b-563de67e32de)

- Real-time inference has shown mixed results. Human classification of emotions from short audio segments can be inherently challenging. The current approach focuses on classifying emotion from a few-second-long audio window. Future work could explore using much longer time windows (e.g., 20-30 seconds) to classify the overall tone of a conversation for more contextual emotion analysis.


With frozen encoder layers and only RAVDEV audio dataset. ![image](https://github.com/user-attachments/assets/a06e94a5-a9e8-4f12-ae0d-a7df148ec078)
Example of Tensorboard: <img width="1000" alt="example_comparisons" src="https://github.com/user-attachments/assets/9764f031-abff-4670-a537-e574633b5f28" />

Example of Inference: Both ASR text and emotions are labeled. The performance on real-time audio is not yet satisfactory despite high validation accuracy, indicating a potential domain gap.

![image](https://github.com/user-attachments/assets/da4f1505-5996-4acc-bc7e-9d1218fd73a4)

## 13\. Limitations and Issues

### Problem State: Discrepancy Between Offline Evaluation and Real-time Inference

**Observation:**
The current emotion classification model, trained on the RAVDESS and CREMA-D datasets with added MUSAN noise augmentation, achieves a decent classification accuracy of approximately 68% on its 8 emotion classes during offline validation. However, when the trained model is deployed for real-time inference on live audio streams, its performance degrades significantly, failing to accurately classify emotions.

**Hypothesized Causes for Performance Degradation:**

The core issue appears to stem from a domain mismatch between the controlled, augmented training data and the unpredictable nature of real-time audio. Several factors are believed to contribute to this discrepancy:

**1. Lack of Real-time Augmentation Coverage (Feature Robustness):**

While MUSAN noise is applied during training, other crucial augmentations like audio masking (Time Masking, Frequency Masking / SpecAugment) are not currently utilized. These techniques are vital for making the model robust to corrupted or incomplete audio segments. Without them, the model may struggle with real-world scenarios where parts of the signal might be obscured or missing.

Furthermore, even the applied MUSAN noise might not fully capture the diversity and characteristics of real-world environmental noise encountered in live scenarios.

**2. Temporal Distribution Mismatch:**

Training data often consists of pre-segmented audio clips with relatively consistent durations. Real-time audio, however, features variable speech tempos, pauses, and overall duration. While the model is now trained on 5-second windows, the original utterances within these windows might still have temporal characteristics (e.g., specific start/end times, internal silences) that are not fully represented by the training augmentations (e.g., insufficient variability in sample lengths or positioning within the fixed window). This can lead to a model that is overfit to specific temporal patterns of the original training data.

**3. Real-world Audio Imperfections and Variability:**

Live audio inherently contains imperfections not fully present or accounted for in curated datasets:

  * **Unforeseen Noise Types:** Background conversations, sudden loud noises, or specific ambient sounds that differ from the MUSAN categories.
  * **Microphone Variability:** Differences in microphone quality, distance, and placement (e.g., built-in laptop mic vs. headset vs. external mic) can introduce spectral and amplitude distortions.
  * **Acoustic Environments:** Variations in room acoustics, reverberation, and echoes in real-world settings that are absent in the anechoic or controlled recording conditions of the datasets.
  * **Non-standard Speech Characteristics:** Subtle variations in speaking style, volume, or voice quality that fall outside the learned distribution of the training data.

## Author
Uttam Paudel
paudeluttam@gmail.com

## License

[MIT](https://choosealicense.com/licenses/mit/)
