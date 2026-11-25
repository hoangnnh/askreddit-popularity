"""Spark-free inference for the exported stage-2 score regression (ridge on log(1 + score)).

Reuses the title preprocessing of ``binary_inference.BinaryPopularityModel``; the
linear margin is the predicted log(1 + score). The export stores validation residual
quantiles (for an 80% interval) and validation prediction percentiles (for a
relative rank among posts with score >= 2).
"""

from __future__ import annotations

import bisect
import math

from askreddit.binary_inference import BinaryPopularityModel


class ScoreModel(BinaryPopularityModel):
    def predict_score(self, title: str) -> dict:
        log_prediction = self.margin(title)
        q = self.metadata["residual_quantiles"]
        percentiles = self.metadata["prediction_percentiles"]
        rank = bisect.bisect_left(percentiles, log_prediction)
        return {"log_prediction": log_prediction, "score": math.expm1(log_prediction),
                "interval": (max(0.0, math.expm1(log_prediction + q["q10"])), math.expm1(log_prediction + q["q90"])),
                "relative_percentile": min(100, max(0, rank - 1)),
                "population": self.metadata["population"]}
