import math

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class PatchEmbedding(nn.Module):
    """
    Divide time series into patches and embed them via a linear projection.
    """

    def __init__(self, patch_size: int, d_model: int, in_channels: int = 1):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Linear(patch_size * in_channels, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, channels) - raw time series data
        Returns:
            (batch, num_patches, d_model) - embedded patches
        """
        batch, seq_len = x.shape
        channels = 1

        # Reshape into patches: (batch, num_patches, patch_size * channels)
        # We ignore any excess time steps that don't fit into a full patch (for now)
        num_patches = seq_len // self.patch_size
        x = x[:, : num_patches * self.patch_size]
        x = x.reshape(batch, num_patches, self.patch_size * channels)

        return self.projection(x)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class ForecastingHead(nn.Module):
    """
    Forecasting head that maps transformer outputs to forecasted values.
    """

    def __init__(
        self,
        d_model: int,
        forecast_horizon: int,
        out_channels: int = 1,
    ):
        super().__init__()
        self.forecast_horizon = forecast_horizon
        self.out_channels = out_channels

        # Project the mean of patches to forecast horizon
        self.head = nn.Linear(d_model, forecast_horizon * out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, num_patches, d_model)
        Returns:
            (batch, forecast_horizon, out_channels)
        """
        batch = x.size(0)
        x = x.mean(dim=1)  # (batch, d_model)
        x = self.head(x)  # (batch, forecast_horizon * out_channels)
        return x.reshape(batch, self.forecast_horizon, self.out_channels)


class TSFMModel(nn.Module):
    """
    Simple Transformer-based model for time series forecasting.
    """

    def __init__(
        self,
        forecast_horizon: int,
        patch_size: int = 16,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        in_channels: int = 1,
        out_channels: int = 1,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.patch_embedding = PatchEmbedding(patch_size, d_model, in_channels)

        self.pos_encoder = PositionalEncoding(d_model, max_len=500, dropout=dropout)

        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.forecasting_head = ForecastingHead(
            d_model=d_model,
            forecast_horizon=forecast_horizon,
            out_channels=out_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, channels) - raw time series
        Returns:
            (batch, forecast_horizon, out_channels) - forecasted values
        """
        x = self.patch_embedding(x)  # (batch, num_patches, d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)  # (batch, num_patches, d_model)
        out = self.forecasting_head(x)  # (batch, forecast_horizon, out_channels)

        return out


if __name__ == "__main__":
    # Model config
    seq_len = 512
    forecast_horizon = 96
    batch_size = 32
    in_channels = 1

    model = TSFMModel(
        forecast_horizon=forecast_horizon,
        patch_size=16,
        d_model=128,
        nhead=8,
        num_encoder_layers=4,
    )

    x = torch.randn(batch_size, seq_len, in_channels)

    out = model(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
