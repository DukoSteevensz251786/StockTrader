"""
CNN model for AAPL 15-minute direction prediction
--------------------------------------------------
Input  : (batch, 13, 30)  — 13 features × 30 time steps
Output : (batch, 2)       — logits for [down, up]

Architecture:
    Three conv layers extract local temporal patterns at different scales,
    each followed by batch norm and dropout for regularisation.
    Global average pooling collapses the time dimension.
    Two fully connected layers produce the final classification.

    Conv1 (kernel=3) : short patterns  — 2-3 bar moves
    Conv2 (kernel=5) : medium patterns — 4-6 bar moves  
    Conv3 (kernel=7) : longer patterns — 6-8 bar moves

Usage:
    from src.models.cnn import CNN
    model = CNN()
"""

import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(
        self,
        n_features : int = 21,
        n_classes  : int = 2,
        dropout    : float = 0.5,
    ):
        super().__init__()

        # ── Convolutional blocks ───────────────────────────────────────────────
        # Each block: Conv1d → BatchNorm → ReLU → Dropout
        # in_channels grows as we extract more abstract features

        self.conv1 = nn.Sequential(
            nn.Conv1d(n_features, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ── Global average pooling ─────────────────────────────────────────────
        # Collapses (batch, 128, 30) → (batch, 128)
        # More robust than flattening — handles variable length input too
        self.gap = nn.AdaptiveAvgPool1d(1)

        # ── Classifier head ────────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        # x: (batch, 13, 30)
        x = self.conv1(x)   # (batch, 32, 30)
        x = self.conv2(x)   # (batch, 64, 30)
        x = self.conv3(x)   # (batch, 128, 30)
        x = self.gap(x)     # (batch, 128, 1)
        x = x.squeeze(-1)   # (batch, 128)
        x = self.classifier(x)  # (batch, 2)
        return x

    def extract_features(self, x):
        """
        Returns the 128-dim embedding before the classifier head.
        Used later to feed into XGBoost alongside sentiment features.
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.gap(x)
        x = x.squeeze(-1)
        return x  # (batch, 128)


# ── Sanity check ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = CNN()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model architecture:\n{model}")
    print(f"\nTotal parameters    : {total_params:,}")
    print(f"Trainable parameters: {trainable:,}")

    # Forward pass with dummy data
    batch = torch.randn(32, 21, 60)   # batch of 32 samples
    out   = model(batch)
    emb   = model.extract_features(batch)

    print(f"\nForward pass check:")
    print(f"  Input  : {batch.shape}")
    print(f"  Output : {out.shape}   (logits for [down, up])")
    print(f"  Embed  : {emb.shape}  (features for XGBoost)")
    print(f"\nOutput sample (logits): {out[0].detach().numpy()}")