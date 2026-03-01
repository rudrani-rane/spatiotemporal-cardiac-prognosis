"""
Ensemble Predictor
==================
Combines predictions from MotionDifferenceModel and HybridEFModel.

Weighting strategy:
    Optimal weights are found by grid-searching the val split (split=1)
    to minimise MAE, then applied at test time.  Equal weights (0.5/0.5)
    are used as the default fallback.

Both models are loaded from their saved checkpoints and run on
matching dataset splits.  Since EchoDataset and HybridEchoDataset
share the same underlying CSV and ordering, predictions can be
paired by index and averaged directly.

Usage (standalone):
    from training.models.ensemble.ensemble_predictor import EnsemblePredictor
    predictor = EnsemblePredictor()
    predictor.find_optimal_weights()        # tune on val split
    preds, true, _, _ = predictor.predict(split=2)
"""

import os
import torch
import numpy as np
from torch.utils.data import DataLoader

# ── Paths ──────────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../..")
)

MOTION_MODEL_PATH = os.path.join(
    PROJECT_ROOT, "training", "models", "motion_diff", "motion_diff_model.pt"
)
HYBRID_MODEL_PATH = os.path.join(
    PROJECT_ROOT, "training", "models", "hybrid", "hybrid_model.pth"
)


# ── Predictor ──────────────────────────────────────────────────────────────

class EnsemblePredictor:
    """
    Loads both trained models and produces weighted-average EF predictions.

    Args:
        device        : torch.device or None (auto-detect)
        motion_weight : initial weight for MotionDifferenceModel (default 0.5)
        hybrid_weight : initial weight for HybridEFModel         (default 0.5)
    """

    def __init__(self, device=None, motion_weight=0.5, hybrid_weight=0.5):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device        = device
        self.motion_weight = motion_weight
        self.hybrid_weight = hybrid_weight

        self._load_models()

    def _load_models(self):
        from training.models.motion_diff.motion_difference_model import MotionDifferenceModel
        from training.models.hybrid.hybrid_model import HybridEFModel

        self.motion_model = MotionDifferenceModel()
        self.motion_model.load_state_dict(
            torch.load(MOTION_MODEL_PATH, map_location=self.device)
        )
        self.motion_model.to(self.device).eval()

        self.hybrid_model = HybridEFModel()
        self.hybrid_model.load_state_dict(
            torch.load(HYBRID_MODEL_PATH, map_location=self.device)
        )
        self.hybrid_model.to(self.device).eval()

    def find_optimal_weights(self, val_steps=200):
        """
        Grid-search the optimal motion_weight on the val split (split=1)
        by minimising MAE.  Updates self.motion_weight / self.hybrid_weight
        in place and returns the best weight pair.

        Args:
            val_steps : number of grid points between 0.0 and 1.0
                        (default 200 → resolution of 0.005 per step)

        Returns:
            (best_motion_weight, best_hybrid_weight, best_val_mae)
        """
        _, true_val, motion_val, hybrid_val = self.predict(split=1)

        best_mae    = float("inf")
        best_alpha  = 0.5

        alphas = np.linspace(0.0, 1.0, val_steps + 1)
        for alpha in alphas:
            blended = alpha * motion_val + (1.0 - alpha) * hybrid_val
            mae     = float(np.mean(np.abs(blended - true_val)))
            if mae < best_mae:
                best_mae   = mae
                best_alpha = float(alpha)

        self.motion_weight = best_alpha
        self.hybrid_weight = 1.0 - best_alpha
        return self.motion_weight, self.hybrid_weight, best_mae

    def predict(self, split=2, batch_size=32):
        """
        Run both models on the given split and return averaged predictions.

        Args:
            split: 0 = train, 1 = val, 2 = test
            batch_size: DataLoader batch size

        Returns:
            ensemble_preds : np.ndarray [N]  – averaged EF predictions
            true_labels    : np.ndarray [N]  – ground-truth EF values
            motion_preds   : np.ndarray [N]  – raw MotionDifferenceModel preds
            hybrid_preds   : np.ndarray [N]  – raw HybridEFModel preds
        """
        from training.dataset_loader import EchoDataset
        from training.hybrid_dataset_loader import HybridEchoDataset

        motion_loader = DataLoader(
            EchoDataset(split=split),
            batch_size=batch_size, shuffle=False, num_workers=0
        )
        hybrid_loader = DataLoader(
            HybridEchoDataset(split=split),
            batch_size=batch_size, shuffle=False, num_workers=0
        )

        motion_preds_list = []
        hybrid_preds_list = []
        true_list         = []

        with torch.no_grad():
            for (mx, my), (hx, _) in zip(motion_loader, hybrid_loader):
                m_pred = self.motion_model(mx.to(self.device)).squeeze(-1)
                h_pred = self.hybrid_model(hx.to(self.device)).squeeze(-1)

                motion_preds_list.append(m_pred.cpu().numpy())
                hybrid_preds_list.append(h_pred.cpu().numpy())
                true_list.append(my.numpy())

        motion_preds   = np.concatenate(motion_preds_list)
        hybrid_preds   = np.concatenate(hybrid_preds_list)
        true_labels    = np.concatenate(true_list)

        ensemble_preds = (
            self.motion_weight * motion_preds +
            self.hybrid_weight * hybrid_preds
        )

        return ensemble_preds, true_labels, motion_preds, hybrid_preds


if __name__ == "__main__":
    predictor = EnsemblePredictor()
    preds, true, _, _ = predictor.predict(split=2)
    print(f"Ensemble predictions on test split: {len(preds)} samples")
    mae = float(np.mean(np.abs(preds - true)))
    print(f"Quick MAE: {mae:.3f}")
