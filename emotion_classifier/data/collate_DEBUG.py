# Placeholder for collate.py
import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    '''
    collate_fn prepares padded waveforms and lengths for downstream encoder-aware masking.
    The actual masking will be handled inside the model after seeing encoder output shape.
    '''
    # Remove any None entries due to file load errors
    batch = [sample for sample in batch if sample is not None]
    if len(batch) == 0:
        return None

    waveforms, labels = zip(*batch)
    assert all(waveform.shape[0] == 1 for waveform in waveforms), "Non mono waveform detected"


    # Pad waveforms to max length in batch
    lengths = [w.shape[-1] for w in waveforms]  # assuming waveform shape is [1, T] or [T]
    padded_waveforms = pad_sequence(waveforms, batch_first=True)  # shape: [B, 1, T_max]
    for w in padded_waveforms:
        print(w.shape)  # All should be [1, T]
    return padded_waveforms, torch.tensor(labels), torch.tensor(lengths)
