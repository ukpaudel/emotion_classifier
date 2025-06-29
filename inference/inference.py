import torch
import torchaudio
import importlib
import yaml
import torch.nn.functional as F

def resolve_class(class_path_or_callable):
    if callable(class_path_or_callable):
        return class_path_or_callable
    module_path, class_name = class_path_or_callable.rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

def load_model_from_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Build classifier
    Classifier = resolve_class(config['model']['class_path'])

    model = Classifier(
        **config['model']['args']
    )
    
    # Load weights
    print('TEST TEST TEST')
    checkpoint = torch.load(config['model']['checkpoint_path'], map_location='cpu')
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    return model, config['model']['sample_rate']

def preprocess_audio(wav_path, sample_rate):
    waveform, sr = torchaudio.load(wav_path)
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    return waveform

def predict(model, waveform):
    # Compute lengths (needed if using attention masks, etc.)
    lengths = torch.tensor([waveform.shape[1]])

    # Wav2Vec2 expects [batch, time], mono channel
    if waveform.ndim == 2 and waveform.shape[0] == 1:
        waveform = waveform.squeeze(0)
    waveform = waveform.unsqueeze(0)  # [1, time]

    with torch.no_grad():
        output = model(waveform, lengths)
        probs = F.softmax(output, dim=-1)
        pred_idx = torch.argmax(probs, dim=-1).item()
    return pred_idx, probs.squeeze().tolist()

if __name__ == "__main__":
    model, target_sr = load_model_from_config("configs/inference_config.yml")
    audio_path = "1001_IEO_ANG_HI.wav"

    waveform = preprocess_audio(audio_path, target_sr)
    label, confidences = predict(model, waveform)

    print(f"Predicted label index: {label}")
    print(f"Confidence scores: {confidences}")

