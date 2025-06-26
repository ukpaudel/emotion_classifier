import torch
import os
import yaml

def save_checkpoint(model, optimizer, epoch, best_val_acc, checkpoint_path, logger=None, scheduler=None):
    checkpoint = {
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'epoch': epoch,
        'best_val_acc': best_val_acc
    }
    if scheduler:
        checkpoint['scheduler_state'] = scheduler.state_dict()

    torch.save(checkpoint, checkpoint_path)
    if logger:
        logger.info(f"Checkpoint saved at epoch {epoch} to {checkpoint_path}")


def load_checkpoint(model, optimizer, scheduler, checkpoint_path, device, logger=None):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    if scheduler and 'scheduler_state' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state'])
    start_epoch = checkpoint['epoch'] + 1
    best_val_acc = checkpoint['best_val_acc']

    if logger:
        logger.info(f"Loaded checkpoint from {checkpoint_path} at epoch {start_epoch}")

    return start_epoch, best_val_acc


def save_misclassified_audio(waveforms, labels, predicted, lengths, config, epoch, logger=None, file_paths=None):
    from torchaudio import save
    os.makedirs("misclassified", exist_ok=True)
    for i in range(len(labels)):
        if labels[i] != predicted[i]:
            file_info = os.path.basename(file_paths[i]) if file_paths else f"sample{i}"
            filename = f"misclassified/epoch{epoch}_{file_info}_true{labels[i].item()}_pred{predicted[i].item()}.wav"
            try:
                save(filename, waveforms[i].cpu(), config['dataset']['sample_rate'])
            except Exception as e:
                if logger:
                    logger.warning(f"Failed to save misclassified audio: {e}")
