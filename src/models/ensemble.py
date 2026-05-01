from __future__ import annotations
import numpy as np


class TieredEnsemble:
    def __init__(
        self, det_weight: float = 0.4, ridge_weight: float = 0.1, bu_weight: float = 0.1
    ):
        self.det_weight = det_weight
        self.ridge_weight = ridge_weight
        self.bu_weight = bu_weight

    def blend_tier1(
        self, specialist_pred: np.ndarray, base_pred: np.ndarray, alpha: float = 0.6
    ) -> np.ndarray:
        return alpha * specialist_pred + (1 - alpha) * base_pred

    def blend_tier2(self, tier1: np.ndarray, ridge_pred: np.ndarray) -> np.ndarray:
        w = self.ridge_weight
        return w * ridge_pred + (1 - w) * tier1

    def blend_final(self, det_pred: np.ndarray, tier2: np.ndarray) -> np.ndarray:
        w = self.det_weight
        return w * det_pred + (1 - w) * tier2

    def blend_bu(self, final_pred: np.ndarray, bu_pred: np.ndarray) -> np.ndarray:
        bu_scale = final_pred.mean() / bu_pred.mean()
        w = self.bu_weight
        return (1 - w) * final_pred + w * bu_pred * bu_scale

    def blend_all(
        self,
        det_pred: np.ndarray,
        base_pred: np.ndarray,
        specialist_pred: np.ndarray | None = None,
        ridge_pred: np.ndarray | None = None,
        bu_pred: np.ndarray | None = None,
        alpha: float = 0.6,
    ) -> np.ndarray:
        if specialist_pred is not None:
            tier1 = self.blend_tier1(specialist_pred, base_pred, alpha)
        else:
            tier1 = base_pred
        if ridge_pred is not None:
            tier2 = self.blend_tier2(tier1, ridge_pred)
        else:
            tier2 = tier1
        final = self.blend_final(det_pred, tier2)
        if bu_pred is not None:
            final = self.blend_bu(final, bu_pred)
        return final
