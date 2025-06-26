import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from utils.logger import setup_logger
from utils.model_utils import save_checkpoint, load_checkpoint, save_misclassified_audio
from utils.run_tracker import update_model_runs_yaml
"""
trainer.py

Handles training and validation loop with support for:
- Resume training from checkpoint
- TensorBoard logging
- Misclassified audio saving
- Flexible optimizer/loss/scheduler

Assumes: model implements forward(x, lengths)
"""

def train_model(model, train_loader, val_loader, config, run_name, resume_training=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'{device} will be used for the training')
    model = model.to(device)

    log_dir = os.path.join(config['logging']['log_dir'], run_name)
    checkpoint_path = os.path.join(log_dir, "checkpoint.pt")
    
    # === Stores the training information to a YML file so that we can compare metrics between different models ===
    # if config['logging'].get('track_run', True):
    #     run_config_path = config['logging'].get('runs_config_path', 'configs/model_runs.yml')
    #     run_label = config['logging'].get('run_label', run_name)
    #     update_model_runs_yaml(run_config_path, log_dir, run_label)
        
    # === CLEAN LOG DIRECTORY IF NOT RESUMING ===
    if not resume_training and os.path.exists(log_dir):
        print(f"[INFO] Removing old log directory: {log_dir}")
        shutil.rmtree(log_dir, ignore_errors=True)

    logger = setup_logger(log_dir=log_dir, run_name=run_name)
    writer = SummaryWriter(log_dir=log_dir)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['lr'])
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config['training']['max_lr'],
        steps_per_epoch=len(train_loader),
        epochs=config['training']['epochs'],
        pct_start=0.1
    )

    # === RESUME TRAINING IF CHECKPOINT EXISTS ===
    start_epoch = 0
    best_val_acc = 0.0
    if resume_training and os.path.exists(checkpoint_path):
        start_epoch, best_val_acc = load_checkpoint(
            model, optimizer, scheduler, checkpoint_path, device, logger
        )
    else:
        logger.info("[INFO] Starting fresh training")


    for epoch in range(start_epoch, config['training']['epochs']):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for i, (waveforms, labels, lengths) in enumerate(train_loader):
          try:
            waveforms, labels, lengths = waveforms.to(device), labels.to(device), lengths.to(device)

            outputs = model(waveforms, lengths)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
          except Exception as e:
            msg = f"[WARN] Failed training batch {i}: {e}"
            print(msg)
            if logger:
              logger.war(msg)

            continue
            
        train_acc = 100 * correct / total
        avg_loss = total_loss / len(train_loader)

        writer.add_scalar("Accuracy/Train", train_acc, epoch)
        writer.add_scalar("Loss/Train", avg_loss, epoch)
        writer.add_scalar("LearningRate", scheduler.get_last_lr()[0], epoch)

        logger.info(f"Epoch {epoch+1}/{config['training']['epochs']} | Train Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}%")

        # === VALIDATION ===
        model.eval()
        val_correct, val_total = 0, 0

        with torch.no_grad():
            for waveforms, labels, lengths in val_loader:
                waveforms, labels, lengths = waveforms.to(device), labels.to(device), lengths.to(device)
                outputs = model(waveforms, lengths)
                _, predicted = torch.max(outputs, 1)

                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

                # Optional: save misclassified examples
                if config['training'].get("save_misclassified"):
                    save_misclassified_audio(waveforms, labels, predicted, lengths, config, epoch, logger)

        val_acc = 100 * val_correct / val_total
        writer.add_scalar("Accuracy/Val", val_acc, epoch)

        logger.info(f"Validation Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch, best_val_acc, checkpoint_path, logger, scheduler)
            logger.info("New best model saved.")

    writer.close()
    logger.info("Training complete.")
