emotion_classifier/
│
├── config/
│   └── config.yaml                # All configurable parameters: model, training, data, paths
│
├── data/
│   ├── __init__.py
│   ├── downloader.py             # Download/extract data if needed
│   ├── ravdess_dataset.py        # Dataset class for RAVDESS
│   └── collate.py                # Collate function for dynamic padding + masking
│
├── models/
│   ├── __init__.py
│   ├── base_model.py             # Generic wrapper model class (e.g. encoder + classifier)
│   ├── attention_classifier.py   # Attention + MLP classifier
│   └── hubert_model.py           # HuBERT encoder integration
│
├── train/
│   ├── __init__.py
│   ├── trainer.py                # Full training + validation loop with checkpointing
│   └── evaluate.py               # Inference, accuracy, confusion matrix, misclassification logging
│
├── utils/
│   ├── __init__.py
│   ├── logger.py                 # Bad file logger
│   ├── saver.py                  # Model saving, loading, checkpoint handling
│   ├── tb_writer.py              # TensorBoard writer initialization and logging
│   ├── plot_utils.py             # Plot accuracy/loss/confusion matrix from logs
│   └── misc.py                   # helper functions (e.g., timestamping runs, label decoding)
│
├── runs/                         # TensorBoard logs
│
├── checkpoints/                 # Model checkpoints
│
├── misclassified/               # Audio files that were misclassified
│
├── notebooks/
│   └── main_experiments.ipynb   # For rapid testing and visualization
│
├── main.py                      # Entry point for training/testing from command line
├── README.md
└── requirements.txt
