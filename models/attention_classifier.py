import torch
import torch.nn as nn

"""
AttentionClassifier module that takes a sequence of hidden features (e.g., from a transformer encoder)
with optional attention masking and returns class logits after attention pooling and MLP.

Input shape: [B, T, H]
Output shape: [B, num_classes]
"""

class AttentionClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_classes=8, dropout=0.3):
        super().__init__()
        self.attn = nn.Linear(input_dim, 1)  # [B, T, H] -> [B, T, 1]
        self.dropout = nn.Dropout(dropout)
        
        # Split the MLP into two parts:
        # Part 1: Produces the latent features (embeddings)
        self.mlp_latent = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        # Part 2: Produces the final logits from the latent features
        self.mlp_output_layer = nn.Linear(hidden_dim, num_classes) # Changed this to be separate

    def forward(self, x, mask=None):
        """
        Args:
            x (Tensor): [B, T, H] features
            mask (Tensor or None): [B, T] boolean mask (True for valid positions)

        Returns:
            logits: [B, num_classes]
            latent_features: [B, hidden_dim] - New return value
        """
        attn_logits = self.attn(x).squeeze(-1)  # [B, T]
        if mask is not None:
            attn_logits[~mask] = -1e9  # mask out padded tokens

        attn_weights = torch.softmax(attn_logits, dim=1).unsqueeze(-1)  # [B, T, 1]
        pooled = torch.sum(attn_weights * x, dim=1)  # [B, H]
        pooled = self.dropout(pooled)
        
        # Pass through the first part of the MLP to get latent features
        latent_features = self.mlp_latent(pooled) # This is your MLP Hidden output

        # Pass latent features through the final layer to get logits
        logits = self.mlp_output_layer(latent_features) # [B, num_classes]
        
        return logits, latent_features # Now return both!