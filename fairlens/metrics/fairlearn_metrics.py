from __future__ import annotations
from typing import Dict
import numpy as np
from fairlens.metrics.common import REGISTRY

def compute_fairlearn_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_features: np.ndarray,
) -> Dict[str, float]:
   
    from fairlearn.metrics import (
        MetricFrame,
        demographic_parity_difference,
        demographic_parity_ratio,
        equalized_odds_difference,
    )
    from sklearn.metrics import recall_score

    results: Dict[str, float] = {}

    results["statistical_parity_difference"] = float(
        demographic_parity_difference(
            y_true, y_pred, sensitive_features=sensitive_features
        )
    )
    results["disparate_impact"] = float(
        demographic_parity_ratio(
            y_true, y_pred, sensitive_features=sensitive_features
        )
    )
    results["average_odds_difference"] = float(
        equalized_odds_difference(
            y_true, y_pred, sensitive_features=sensitive_features
        )
    )

    mf = MetricFrame(
        metrics=recall_score,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features,
    )
    groups = list(mf.by_group.index)
    if len(groups) == 2:
        g0, g1 = groups
        tpr_unpriv = mf.by_group.loc[0] if 0 in groups else mf.by_group.iloc[0]
        tpr_priv = mf.by_group.loc[1] if 1 in groups else mf.by_group.iloc[1]
        results["equal_opportunity_difference"] = float(tpr_unpriv - tpr_priv)
    else:
        results["equal_opportunity_difference"] = float("nan")

    return results
