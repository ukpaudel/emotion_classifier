import torch
import torchaudio
import random

def _add_noise_with_snr(clean_audio: torch.Tensor, noise_audio: torch.Tensor, 
                        snr_db: float, sample_rate: int) -> torch.Tensor:
    """
    Mixes noise_audio with clean_audio at a specified Signal-to-Noise Ratio (SNR).

    Args:
        clean_audio (torch.Tensor): The clean speech waveform (e.g., shape (1, seq_len)).
        noise_audio (torch.Tensor): The noise waveform (e.g., shape (1, noise_len)).
        snr_db (float): Desired SNR in decibels.
        sample_rate (int): The sample rate of the audio.

    Returns:
        torch.Tensor: The mixed waveform.
    """
    if clean_audio.dim() == 1:
        clean_audio = clean_audio.unsqueeze(0) # Ensure (1, seq_len)

    if noise_audio.dim() == 1:
        noise_audio = noise_audio.unsqueeze(0) # Ensure (1, seq_len)

    clean_len = clean_audio.shape[-1]
    noise_len = noise_audio.shape[-1]

    # Ensure noise segment is at least as long as clean audio
    if noise_len < clean_len:
        # Repeat noise if too short
        repeat_factor = (clean_len + noise_len - 1) // noise_len
        noise_audio = noise_audio.repeat(1, repeat_factor)[:, :clean_len]
    else:
        # Take a random segment of noise if it's longer
        start_idx = random.randint(0, noise_len - clean_len)
        noise_audio = noise_audio[:, start_idx : start_idx + clean_len]

    # Calculate signal power and noise power
    P_signal = torch.mean(clean_audio**2)
    P_noise = torch.mean(noise_audio**2)

    # Calculate desired noise power based on SNR_dB
    # SNR_dB = 10 * log10(P_signal / P_noise_desired)
    # log10(P_signal / P_noise_desired) = SNR_dB / 10
    # P_signal / P_noise_desired = 10^(SNR_dB / 10)
    # P_noise_desired = P_signal / (10^(SNR_dB / 10))
    # P_noise_desired = P_signal / (10**(snr_db / 10))
    
    # Handle potential division by zero or very small numbers
    epsilon = 1e-10
    P_signal = max(P_signal, epsilon)
    P_noise = max(P_noise, epsilon)

    target_noise_power = P_signal / (10**(snr_db / 10))

    # Scale noise to achieve desired noise power
    scale_factor = torch.sqrt(target_noise_power / P_noise)
    scaled_noise = noise_audio * scale_factor

    return clean_audio + scaled_noise