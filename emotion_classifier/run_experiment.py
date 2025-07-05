import os
import yaml
import torch
import logging
from utils.confusion_animation import animate_confusion, plot_final_confusion
from utils.config import load_config
from utils.logger import setup_logger
from utils.run_tracker import update_model_runs_yaml
from train.trainer import train_model
from utils.tensorboard_plot_utils import plot_from_tensorboard
from data.dataloader import create_dataloaders
from models.emotion_model import EmotionModel
from utils.encoder_loader import load_ssl_encoder

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def run_experiment(config_path: str = "configs/config.yml"):
    # Load YAML config
    config = load_config(config_path)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.cfg['device'] = device  # Inject device into config

    # Set up logger
    log_dir = config['logging']['log_dir']
    run_name = config['logging'].get('run_label', log_dir)
    logger = setup_logger(log_dir, run_name)
    logger.info("===== Running Experiment =====")

    # Track run in YAML registry. For example different experiments will get stored as a separate file you need to specify in yml file
    #  log_dir: "runs/emotion_classifier_v2" #this is a unique id for the training. when it reruns it overrides.
    #  run_label: "Wav2Vec2-unfreez1-layer+Dropout0.3"
    if config['logging'].get('track_run', True):
        run_config_path = config['logging'].get('runs_config_path', 'configs/model_runs.yml')
        update_model_runs_yaml(run_config_path, log_dir, run_name)

    # Load encoder and model
    model_name = config['model']['encoder_name']
    #encoder = load_ssl_encoder(model_name)
    logger.info(f"===== Successfully loaded {model_name} model =====")

    model = EmotionModel(
        encoder_name=config.get("model", "encoder_name"),
        dropout=config.get("model", "dropout"),
        hidden_dim=config.get("model", "hidden_dim"),
        num_classes=config.get("model", "num_classes"),
        freeze_encoder=config.get("model", "freeze_encoder"),
        unfreeze_last_n_layers=config.get("model", "unfreeze_last_n_layers", default=None),
        logger=logger
)

    #print trainable layers.
    for name, param in model.encoder.encoder.transformer.named_parameters():
        print(f"{name}: requires_grad={param.requires_grad}")

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(config)

    # Train model
    resume_training = config['training'].get('resume_training', False)
    train_model(model, train_loader, val_loader, config, run_name, resume_training=resume_training)

    # Plot results
    plot_from_tensorboard(config_path)

    # plot confusion
    log_dir_confusion = os.path.join(config['logging']['log_dir'], run_name)
    animate_confusion(log_dir_confusion)
    plot_final_confusion(log_dir_confusion)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run experiment from config")
    parser.add_argument("--config", type=str, default="configs/config.yml", help="Path to YAML config file")
    args = parser.parse_args()

    run_experiment(args.config)
    print("To Visualize Tensboard Weights go to bash and run > tensorboard --logdir=runs and open http://localhost:6006/ ")