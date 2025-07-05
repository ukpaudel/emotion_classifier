# data_augmentations.py
import torch
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F # For functional operations if needed
import random
import os

# Helper function for adding noise with SNR, moved here
def _add_noise_with_snr(clean_audio: torch.Tensor, noise_audio: torch.Tensor, 
                        snr_db: float, sample_rate: int) -> torch.Tensor:
    """
    Mixes noise_audio with clean_audio at a specified Signal-to-Noise Ratio (SNR).
    Ensures noise segment matches clean_audio length.

    Args:
        clean_audio (torch.Tensor): The clean speech waveform (e.g., shape (1, seq_len)).
        noise_audio (torch.Tensor): The noise waveform (e.g., shape (1, noise_len)).
        snr_db (float): Desired SNR in decibels.
        sample_rate (int): The sample rate of the audio.

    Returns:
        torch.Tensor: The mixed waveform.
    """
    # Ensure tensors are 2D (channels, samples)
    if clean_audio.dim() == 1:
        clean_audio = clean_audio.unsqueeze(0)
    if noise_audio.dim() == 1:
        noise_audio = noise_audio.unsqueeze(0)

    # Ensure same number of channels
    if clean_audio.shape[0] != noise_audio.shape[0]:
        if clean_audio.shape[0] == 1 and noise_audio.shape[0] > 1:
            # If clean is mono, noise is stereo, convert noise to mono
            noise_audio = torch.mean(noise_audio, dim=0, keepdim=True)
        elif clean_audio.shape[0] > 1 and noise_audio.shape[0] == 1:
            # If clean is stereo, noise is mono, duplicate mono noise
            noise_audio = noise_audio.repeat(clean_audio.shape[0], 1)
        else:
            # Mismatched multi-channel, this case is complex, might need more specific handling
            # For now, let's assume mono or matching channels for simplicity
            print(f"Warning: Channel mismatch during noise addition. Clean: {clean_audio.shape[0]}, Noise: {noise_audio.shape[0]}. Skipping noise.")
            return clean_audio


    clean_len = clean_audio.shape[-1]
    noise_len = noise_audio.shape[-1]

    # Match length of noise to clean audio
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
    epsilon = 1e-10 # Small value to prevent division by zero
    P_signal = max(P_signal, epsilon)
    P_noise = max(P_noise, epsilon)

    target_noise_power = P_signal / (10**(snr_db / 10))

    # Scale noise to achieve desired noise power
    scale_factor = torch.sqrt(target_noise_power / P_noise)
    scaled_noise = noise_audio * scale_factor

    return clean_audio + scaled_noise


# Main augmentation function to be called from dataset classes
def apply_augmentations(waveform: torch.Tensor, sample_rate: int, noise_file_paths_map: dict = None, device: torch.device = None) -> torch.Tensor: # New parameter

    """
    Applies random data augmentations to the waveform.

    Args:
        waveform (torch.Tensor): The input audio waveform (e.g., shape (1, seq_len)).
        sample_rate (int): The sample rate of the waveform.
        noise_file_paths_map (dict, optional): A dictionary where keys are noise categories 
                    (e.g., 'noise', 'speech', 'music') and values 
                    are lists of full file paths for that category.
                    This should be pre-collected.
        evice (torch.device, optional): The device (e.g., 'cuda', 'cpu') to perform operations on.
                                        If None, defaults to waveform's current device.

    Returns:
        torch.Tensor: The augmented waveform.
    """
    if device is None:
        device = waveform.device # Fallback if device isn't explicitly passed

    # Ensure waveform is 2D (channels, samples) for consistent transform application
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)


# --- 1. Apply waveform-based augmentations ---

    # --- Gain (Volume) ---
    if random.random() < 0.5: # 50% chance to apply gain
        gain_db = random.uniform(-6, 6) # Random gain between -6dB and +6dB
        vol_transform = T.Vol(gain=gain_db, gain_type="db").to(device)
        waveform = vol_transform(waveform)

    # # --- Pitch Shift ---
    if random.random() < 0.0: # 10% chance to apply pitch shift
        try:
            n_steps = random.uniform(-4, 4) # Shift pitch by -4 to +4 semitones
            pitch_shifter = T.PitchShift(sample_rate, n_steps).to(device)
            waveform = pitch_shifter(waveform)
        except Exception as e:
            # Handle cases where PitchShift might not be fully supported or cause issues
            print(f"Warning: Could not apply PitchShift: {e}. Skipping.")
            pass # Continue without this augmentation

    # --- Noise from MUSAN (Crackle/Microphone Noise) ---
    if noise_file_paths_map and any(noise_file_paths_map.values()) and random.random() < 0.3:
        try:
            noise_categories = ['noise', 'speech', 'music']
            #selected_category = random.choice(noise_categories)
            #TODO hardcoded noise folder only for now. 
            selected_category = 'noise'
            # Retrieve pre-collected file paths
            current_category_noise_files = noise_file_paths_map.get(selected_category)

            if current_category_noise_files: # Check if there are files in this category
                random_noise_file = random.choice(current_category_noise_files) # Pick from pre-collected list
                # Noise waveform is loaded on CPU first
                noise_waveform, noise_sr = torchaudio.load(random_noise_file)
                # Resample noise if its sample rate doesn't match
                if noise_sr != sample_rate:  
                    resampler = T.Resample(noise_sr, sample_rate)
                    noise_waveform = resampler(noise_waveform)

                # ---  Move loaded noise to GPU for mixing ---

                noise_waveform_gpu = noise_waveform.to(device)
                # Choose a random SNR for mixing (e.g., between 5 dB and 20 dB)
                snr_db = random.uniform(5.0, 20.0) 
                # _add_noise_with_snr will now operate on GPU tensors
                waveform = _add_noise_with_snr(waveform, noise_waveform_gpu, snr_db, sample_rate)
            else:
                print(f"Warning: Noise category path '{noise_category_path}' does not exist. Skipping noise augmentation.")
        except Exception as e:
            print(f"Warning: Error applying MUSAN noise from {noise_dir}: {e}. Skipping.")
            pass # Continue without this noise augmentation if there's an error


    # --- 2. Apply Spectrogram-based augmentations (e.g., Time Stretch) in GPU ---
    # TimeStretch requires a complex-valued spectrogram input.
    # We'll convert to spectrogram, apply stretch, then convert back to waveform.
    if random.random() < 0.3: # 5% chance to apply time stretch
        stretch_rate = random.uniform(0.8, 1.2) 
        n_fft = 1024
        hop_length = n_fft // 4 
        n_freq = n_fft // 2 + 1 # This calculates 1025 for n_fft=2048
        
        # Initialize Spectrogram WITHOUT return_complex=True (deprecated warning)
        spectrogram_transform = T.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            power=None, 
        ).to(device) 
        
        # Initialize TimeStretch with explicit n_freq and hop_length for robustness
        # This is the main fix for the "size of tensor a (1025) must match b (201)" error
        time_stretcher = T.TimeStretch(n_freq=n_freq, hop_length=hop_length).to(device) 
        
        inverse_spectrogram_transform = T.InverseSpectrogram(
            n_fft=n_fft,
            hop_length=hop_length
        ).to(device) 

        try:
            complex_spectrogram = spectrogram_transform(waveform) # Input (waveform) is already on GPU
            stretched_complex_spectrogram = time_stretcher(complex_spectrogram, stretch_rate)
            waveform = inverse_spectrogram_transform(stretched_complex_spectrogram)
            #print('Waveform time stretched')
            
        except Exception as e:
            print(f"Warning: Error applying TimeStretch: {e}. Skipping.")
            pass 

    return waveform # The augmented is now on the GPU
