"""LSTM model and the on-disk model bundle shared by training and serving."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .features import Preprocessor

# Files that make up a model bundle (packed as model.tar.gz on SageMaker)
WEIGHTS_FILE      = 'model.pt'
CONFIG_FILE       = 'model_config.json'
PREPROCESSOR_FILE = 'preprocessor.json'
BACKGROUND_FILE   = 'shap_background.npy'


class SepsisLSTM(nn.Module):
    """
    Stacked LSTM over a window of hourly features.

    Returns raw logits so training can use BCEWithLogitsLoss (numerically
    stable); call predict_proba() for probabilities.
    """

    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            dropout     = dropout if num_layers > 1 else 0.0,
            batch_first = True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)                    # (batch, time, hidden)
        out    = self.dropout(out[:, -1, :])     # last timestep
        return self.fc(out).squeeze(1)           # (batch,) logits

    @torch.no_grad()
    def predict_proba(self, x: np.ndarray, batch_size: int = 8192) -> np.ndarray:
        self.eval()
        device = next(self.parameters()).device
        probs = [
            torch.sigmoid(self(torch.from_numpy(x[i:i + batch_size]).to(device))).cpu()
            for i in range(0, len(x), batch_size)
        ]
        return torch.cat(probs).numpy()


def save_bundle(model_dir: str | Path, model: SepsisLSTM, config: dict,
                preprocessor: Preprocessor, background: np.ndarray) -> None:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_dir / WEIGHTS_FILE)
    (model_dir / CONFIG_FILE).write_text(json.dumps(config, indent=2))
    preprocessor.to_json(model_dir / PREPROCESSOR_FILE)
    np.save(model_dir / BACKGROUND_FILE, background.astype(np.float32))


def load_bundle(model_dir: str | Path, device: str = 'cpu'):
    """Return (model, config, preprocessor, background or None)."""
    model_dir = Path(model_dir)
    config = json.loads((model_dir / CONFIG_FILE).read_text())
    arch = config['architecture']
    model = SepsisLSTM(
        input_size  = len(config['feature_cols']),
        hidden_size = arch['hidden_size'],
        num_layers  = arch['num_layers'],
        dropout     = arch['dropout'],
    )
    model.load_state_dict(torch.load(model_dir / WEIGHTS_FILE,
                                     map_location=device, weights_only=True))
    model.to(device).eval()

    preprocessor = Preprocessor.from_json(model_dir / PREPROCESSOR_FILE)
    bg_path = model_dir / BACKGROUND_FILE
    background = np.load(bg_path) if bg_path.exists() else None
    return model, config, preprocessor, background
