# Placeholder for collate.py
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

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
    #convert to 1D flatten mono channel, if needed
    waveforms = [w.squeeze(0) for w in waveforms]
    #find max length of the time from the batch
    lengths = [w.shape[0] for w in waveforms]
    max_len = max(lengths)

    #pad all waveform to max_len
    padded = [F.pad(w, (0,max_len-w.shape[0])) for w in waveforms]

    #stack into (BxL)
    batch_waveforms = torch.stack(padded)
    #stack labels
    batch_labels = torch.tensor(labels)

    # NOTE: Don't try to use this mask on the encoder output!
    # Just return the original waveform lengths.
    return batch_waveforms, batch_labels, torch.tensor(lengths)
