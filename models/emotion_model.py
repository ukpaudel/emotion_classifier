import torch
import torch.nn as nn
from emotion_classifier.models.attention_classifier import AttentionClassifier
from emotion_classifier.utils.encoder_loader import load_ssl_encoder

'''
EmotionModel is a modular audio classification model designed to wrap a frozen or partially trainable
self-supervised speech encoder (e.g., wav2vec2, HuBERT) with a downstream attention-based classifier.

Features:
- Plug-and-play support for different SSL encoders via config
- Optional encoder freezing or selective fine-tuning of last N layers
- Dynamic masking support for variable-length audio input
'''

class EmotionModel(nn.Module):
    def __init__(self, encoder_name="wav2vec2", dropout=0.3, hidden_dim=256, num_classes=8,
                 freeze_encoder=True, unfreeze_last_n_layers=None, logger=None):
        super().__init__()
        self.encoder_name = encoder_name
        self.num_classes = num_classes
        # Load encoder bundle components and validate API
        print("Inside EmotionModel")
        encoder_bundle = load_ssl_encoder(self.encoder_name)
        msg = f"[Info] Encoder Bundle Extracted!!! Model Information {encoder_bundle}."
        if logger:
            logger.info(msg)
        self.encoder = encoder_bundle["model"]
        self.sample_rate = encoder_bundle["sample_rate"]
        self.feature_dim = encoder_bundle["feature_dim"]

        # Optionally freeze encoder
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            #msg = f"Encoder '{self.encoder_name}' is frozen (no gradient updates) unless we ."
            #if logger:
            #    logger.info(msg)

        if unfreeze_last_n_layers:
            # Unfreeze the last N transformer layers (if supported)
            try:
                transformer_layers = self.encoder.encoder.transformer.layers
                for layer in transformer_layers[-unfreeze_last_n_layers:]:
                    for param in layer.parameters():
                        param.requires_grad = True

            except AttributeError:
                msg = f"[Warning] Encoder '{self.encoder_name}' does not expose transformer layers. Cannot unfreeze selectively."
                if logger:
                    logger.warn(msg)
                print(msg)

        # #print model info in the log file
        for name, param in self.encoder.encoder.transformer.named_parameters():
            msg = f"{name}: requires_grad={param.requires_grad}"
            if logger:
                logger.info(msg)

        # # check what is trainable inside the transformer layers
        # for name, param in self.encoder.encoder.transformer.named_parameters():
        #     print(f"{name}: requires_grad={param.requires_grad}")


        self.classifier = AttentionClassifier(
            input_dim=self.feature_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout
        )

        if logger:
            logger.info(f"Initialized EmotionModel with encoder='{self.encoder_name}' | Feature dim: {self.feature_dim} | Classes: {self.num_classes}")

    def forward(self, waveforms, lengths):
        """
        waveforms: Tensor [B, 1, T]
        lengths: Tensor [B]  (original waveform lengths)
        """
        x = waveforms.squeeze(1)

        with torch.no_grad():
            features, _ = self.encoder.extract_features(x)
            features = features[-1].detach()  # shape [B, T_out, F]

        B, T_out, _ = features.shape
        T_in = waveforms.shape[-1]
        downsampled_lengths = (lengths.float() * T_out / T_in).long()

        mask = torch.zeros(B, T_out, dtype=torch.bool, device=features.device)
        for i, l in enumerate(downsampled_lengths):
            mask[i, :l] = 1

        logits = self.classifier(features, mask)
        return logits
