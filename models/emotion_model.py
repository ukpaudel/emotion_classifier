import torch
import torch.nn as nn
import random
from models.attention_classifier import AttentionClassifier
from utils.encoder_loader import load_ssl_encoder
from utils.feature_store import feature_store #for mutable dictionary that both EmotionModel and run_experiments will see
from models.attention_classifier import AttentionClassifier
from models.domain_classifier import DomainClassifier
from models.grl import grad_reverse

'''
EmotionModel is a modular audio classification model designed to wrap a frozen or partially trainable
self-supervised speech encoder (e.g., wav2vec2, HuBERT) with a downstream attention-based classifier.

Features:
- Plug-and-play support for different SSL encoders via config
- Optional encoder freezing or selective fine-tuning of last N layers
- Dynamic masking support for variable-length audio input
'''

class EmotionModel(nn.Module):
    def __init__(self, encoder_name="hubert", dropout=0.3, hidden_dim=256, num_classes=8,
                 freeze_encoder=True, unfreeze_last_n_layers=None,  num_domains=2,
                 grl_lambda=1.0, logger=None):
        super().__init__()
        self.encoder_name = encoder_name
        self.num_classes = num_classes
        self.num_domains = num_domains
        self.grl_lambda = grl_lambda

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

        # new domain classifier
        self.domain_classifier = DomainClassifier(
            input_dim=self.feature_dim,
            hidden_dim=hidden_dim // 2,
            num_domains=num_domains,
            dropout=dropout
        )
        if logger:
            logger.info(f"Initialized EmotionModel with domain-adversarial head (domains={num_domains})")

    
    def apply_feature_masking(self, features, time_mask_width=10, feature_mask_width=20, p=0.5):
        '''
        Applies SpecAugment-style feature masking directly on HuBERT features.
        '''
        if random.random() > p:
            return features
        
        B, T, F = features.shape
        
        for b in range(B):
            t = random.randint(0, T - time_mask_width)
            width = random.randint(1, time_mask_width)
            features[b, t:t+width, :] = 0.0

        for b in range(B):
            f = random.randint(0, F - feature_mask_width)
            width = random.randint(1, feature_mask_width)
            features[b, :, f:f+width] = 0.0
        
        return features



    def forward(self, waveforms, lengths):
        """
        waveforms: Tensor [B, 1, T]
        lengths: Tensor [B]  (original waveform lengths)
        """
        x = waveforms.squeeze(1)

        features, _ = self.encoder.extract_features(x)
        features = features[-1] # shape [B, T_out, F]
        #Applies SpecAugment-style feature masking directly on HuBERT features. nn.module knows if it is training
        if self.training:
            features = self.apply_feature_masking(features)
        # add this line to store pooled encoder features
        pooled_encoder_features = features.mean(dim=1)  # [B, F], F is self.feature_dim
        for i in range(pooled_encoder_features.shape[0]):
            feature_store["encoder"].append(pooled_encoder_features[i].cpu())

        B, T_out, _ = features.shape
        T_in = waveforms.shape[-1]
        downsampled_lengths = (lengths.float() * T_out / T_in).long()

        mask = torch.zeros(B, T_out, dtype=torch.bool, device=features.device)
        for i, l in enumerate(downsampled_lengths):
            mask[i, :l] = 1
        
        # Call the AttentionClassifier, which now returns both logits and latent_features
        logits_emotion, latent_features = self.classifier(features, mask)

        # Domain classifier with GRL
        grl_features = grad_reverse(pooled_encoder_features, lambda_=self.grl_lambda)
        logits_domain = self.domain_classifier(grl_features)

        return logits_emotion, logits_domain, pooled_encoder_features #output of the transformer encoder
