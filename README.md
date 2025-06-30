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
## Install Dataset
RAVDESS: https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio
Download and extract it into data/ravdess/data/actor_1, actor_2, etc. 
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

## Results.
see comparison between different trainings in 
```runs\example_comparisons.png``` file. 
Verdict: mixed results. Saw as high as closer to 80% classification accuracy with RAVDAV dataset. Adding CREMA-D dataset made the classification accuracy go hover around 70%. Unfreezing the encoder layers gave inconsistent results, for RAVDAV dataset, it increased the efficiency significantly, though I am concerned that the validation dataset was too small and the model was overfitting.
Wav2Vec2 had poorer performance. HuBert results in increase classification accuracy with model converging with fewer epoches.
With frozen encoder layers and only RAVDEV audio dataset. ![image](https://github.com/user-attachments/assets/a06e94a5-a9e8-4f12-ae0d-a7df148ec078)
Example of Tensorboard: <img width="1000" alt="example_comparisons" src="https://github.com/user-attachments/assets/9764f031-abff-4670-a537-e574633b5f28" />

## Author
Uttam Paudel
paudeluttam@gmail.com

## License

[MIT](https://choosealicense.com/licenses/mit/)
