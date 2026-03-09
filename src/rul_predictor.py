from __future__ import annotations
import json
from pathlib import Path
import numpy as np

class RULPredictorWrapper:
    """
    Tries to load a trained GRU artifact (from your earlier notebook).
    Falls back to a simple linear mapping if artifact not found.
    """
    def __init__(self, artifact_dir: str | None, window: int, features: list[str]):
        self.window = int(window)
        self.features = features
        self.use_artifact = False

        if artifact_dir and Path(artifact_dir).exists():
            try:
                from rul_model_artifact.example_usage import RULPredictor as ArtifactPredictor
                self.p = ArtifactPredictor(artifact_dir=artifact_dir)
                # sanity: align window/features
                self.window = int(self.p.window)
                self.features = self.p.features
                self.use_artifact = True
            except Exception:
                self.use_artifact = False

    def predict(self, data_no: int, window_array: np.ndarray, t_idx: int, lifespan: int) -> float:
        """
        Returns RUL_norm in [0,1].
        - If GRU artifact available: use it.
        - Else: use linear true RUL from index/lifespan (for debugging).
        """
        if self.use_artifact:
            return float(self.p.predict(data_no=int(data_no), window_array=window_array))
        # fallback: true linear RUL
        denom = max(lifespan - 1, 1)
        return float(np.clip((denom - t_idx) / denom, 0.0, 1.0))
