import torchaudio
import torch

"""
Utility for loading self-supervised audio encoder bundles from torchaudio or other sources.
Supports flexible model switching for EmotionModel and other downstream tasks.
Checks if the models have consistent output parameter to be fed to emotion_model.py
Returns validated encoder + metadata.
"""

def load_ssl_encoder(name: str, logger=None):
    """
    Load and validate an SSL encoder.

    Args:
        name (str): encoder name, one of ["wav2vec2", "hubert", "wavlm"]

    Returns:
        dict: {
            'model': encoder model,
            'sample_rate': int,
            'feature_dim': int,
            'name': str
        }

    Raises:
        ValueError: if encoder is unsupported or missing required API
    """
    name = name.lower()

    if logger:
        logger.info(f"Loading encoder: {name}")

    if name == "wav2vec2":
        bundle = torchaudio.pipelines.WAV2VEC2_BASE
    elif name == "hubert":
        bundle = torchaudio.pipelines.HUBERT_BASE
    elif name == "wavlm":
        bundle = torchaudio.pipelines.WAVLM_BASE
    else:
        raise ValueError(f"Unsupported encoder name: {name}")

    model = bundle.get_model()
    sample_rate = bundle.sample_rate
    
    # Validate extract_features
    if not hasattr(model, "extract_features"):
        msg = f"Encoder '{name}' must implement extract_features()."
        if logger: logger.error(msg)
        raise ValueError(msg)

    # Dummy forward to infer feature dimension
    try:
        model.eval()
        with torch.no_grad():
            dummy_waveform = torch.randn(1, 16000)
            features = model.extract_features(dummy_waveform)
            if isinstance(features, tuple):
                features = features[0]
            if isinstance(features, list):
                last_layer_feat = features[-1]
            else:
                last_layer_feat = features

            if not isinstance(last_layer_feat, torch.Tensor):
                msg = "extract_features() must return a list/tuple of Tensors."
                if logger: logger.error(msg)
                raise TypeError(msg)

            feature_dim = last_layer_feat.shape[-1]
    except Exception as e:
        msg = f"Encoder '{name}' failed extract_features() check. Error: {e}"
        if logger: logger.error(msg)
        raise ValueError(msg)

    return {
        "model": model,
        "sample_rate": sample_rate,
        "feature_dim": feature_dim,
        "name": name
    }