# Summary

**emotion_classifier** is a modular, production-ready Python framework for speech emotion classification. It leverages powerful self-supervised speech encoders (like Wav2Vec2 or HuBERT) with an attention-based classifier on top, allowing easy transfer learning and fine-tuning for emotion recognition tasks.

It is designed to be configuration-driven, making training, evaluation, and experimentation easy and reproducible.

---

## 🚀 Features

- 🔌 Plug-and-play support for self-supervised encoders (tested with HuBERT, Wav2Vec2)
- ❄️ Encoder freezing or selective fine-tuning of the last *N* layers (`unfreeze_last_n_layers`)
- 🧩 Dynamic masking support for variable-length audio
- 🗂️ Extensible to new datasets — just add a new dataset loader like `ravdess_dataset.py`
- 💾 Pretrained weight loading with `pretrained_weights.enabled: true`
- 📝 TensorBoard integration and logging for visualization
- 🔄 Checkpoint-based resume functionality
- 🎯 Supports single-file inference and real-time emotion detection with ASR
- 🐛 Automatic saving of misclassified audio samples for inspection (Need to update)
- 🛠️ Configurable entirely through `configs/config.yml`

---
Here's the updated `README.md` with the new sections integrated. I've placed the "Data Augmentation Strategy" after the "Data Loading and Preprocessing" and added a new "Install Datasets" section at the end.

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

## Install Datasets

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

      * (Add specific download instructions for CREMA-D here, similar to RAVDESS, if available.)
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

## Folder Structure

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
│
├── run_experiment.py
├── setup.py
├── README.md
└── ...
```


## Architecture
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
## ⚙️ Installation

```bash
git clone https://github.com/ukpaudel/emotion_classifier.git
cd emotion_classifier
pip install -e .

```

## Usage

### Training

```
>cd emotion_classifier
>python .\run_experiment.py 

This will read parameters from configs/config.yml, including:
encoder type
dataset path
hyperparameters
output directories
resume or pretrained settings

```
### Visualize Comparisons Between Different Experiments
To Visualize Tensboard Weights go to bash and run ```  > tensorboard --logdir=runs and open http://localhost:6006/ ```

### Inference 
For realtime
To run go to the root folder ```cd emotion_classifier\emotion_classifier```  
``` > python .\inference\realtime_inference_withASR.py```

for single audio
``` > python .\inference\realtime_inference_withASR.py```

## Results/Verdicts.

Comparison of Different Trainings (see ```runs/example_comparisons.png``` for details):

- Achieved up to ~80% classification accuracy using the RAVDESS dataset alone.

- Adding the CREMA-D dataset reduced classification accuracy to around 70%.

- Unfreezing the encoder layers gave inconsistent results. For the RAVDESS dataset, unfreezing improved performance, but I suspect the small validation dataset may have caused overfitting.

- Wav2Vec2 showed poorer performance overall.

- HuBERT achieved higher classification accuracy and converged in fewer epochs when trained with frozen encoder layers on the RAVDESS dataset.
  
- Realtime inference  has a mixed results. Even I have trouble classifying emotions from audio. Perhaps the idea of classifying emotion from a few second long audio is an unrealistic goals? Once could train to use much longer time window, 20-30 seconds to classify the overall tone of the conversation? 


With frozen encoder layers and only RAVDEV audio dataset. ![image](https://github.com/user-attachments/assets/a06e94a5-a9e8-4f12-ae0d-a7df148ec078)
Example of Tensorboard: <img width="1000" alt="example_comparisons" src="https://github.com/user-attachments/assets/9764f031-abff-4670-a537-e574633b5f28" />

Example of Inference: Both ASR text and emotions are labeled. I am not happy with the performance on the realtime audio despite the validation accuracy being high. Something ain't right...
![image](https://github.com/user-attachments/assets/da4f1505-5996-4acc-bc7e-9d1218fd73a4)

## Author
Uttam Paudel
paudeluttam@gmail.com

## License

[MIT](https://choosealicense.com/licenses/mit/)
