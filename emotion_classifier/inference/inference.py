import torch
import torchaudio
import importlib
import yaml
import torch.nn.functional as F
import os
import numpy as np
'''
This file passes example data recorded at home and use it for inference.
run for test
\emotion_classifier\emotion_classifier> python inference\inference.py
'''

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
    # Convert stereo → mono by averaging channels
    if waveform.shape[0] == 2:
        waveform = waveform.mean(dim=0, keepdim=True)  # shape: (1, num_samples)

    with torch.no_grad():
        output = model(waveform, lengths)
        probs = F.softmax(output, dim=-1)
        pred_idx = torch.argmax(probs, dim=-1).item()
    return pred_idx, probs.squeeze().tolist()

if __name__ == "__main__":
    model, target_sr = load_model_from_config("configs/inference_config.yml")
    test_dir = r"./test_data"
    for audio_file in ['angry.wav','happy.wav','sad.wav','surprise.wav']:
        audio_path = os.path.join(test_dir,audio_file)
        print('Inference on ',audio_path)
        waveform = preprocess_audio(audio_path, target_sr)
        label, confidences = predict(model, waveform)
        emotion_map = {
                0: 'Neutral', 1: 'Calm', 2: 'Happy', 3: 'Sad',
                4: 'Angry', 5: 'Fearful', 6: 'Disgust', 7: 'Surprised'
            }
        print(f"\n\n\n\nPredicted emotion: {emotion_map[label]} vs True Emotion: {audio_file.split('.')[0]}\n")
        print(f"Confidence scores: {list(zip(emotion_map.values(),np.round(np.array(confidences),1)))}\n\n\n\n\n")

