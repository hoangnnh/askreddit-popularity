"""Spark-free inference for an exported binary (high vs low popularity) logistic-regression model.

Reuses the NumPy preprocessing of ``inference.py`` (NFKC, tokenizer, uni/bigrams over
the frozen vocabulary, train IDF, L2 normalization, train-std scaling).
``export_binary_model`` verifies parity against Spark before writing the export.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from askreddit.inference import normalize_unicode, structured_values, tokenize

SUPPORTED_FEATURES = ("text_features", "title_structured_features", "combined_title_features",
                      "time_features", "structured_features", "combined_features")
TITLE_WIDTH = 5  # title_characters, title_words, question_count, question_present, uppercase_ratio
NEEDS_TIME = ("time_features", "structured_features", "combined_features")


class BinaryPopularityModel:
    def __init__(self, metadata: dict, arrays: dict):
        if metadata["features"] not in SUPPORTED_FEATURES:
            raise ValueError(f"Unsupported feature set for Spark-free inference: {metadata['features']}")
        self.metadata = metadata
        self.features = metadata["features"]
        self.vocabulary = {term: index for index, term in enumerate(metadata["vocabulary"])}
        self.coefficients = np.asarray(arrays["coefficients"], dtype=np.float64)
        self.intercept = float(np.asarray(arrays["intercept"]).reshape(()))
        self.idf = np.asarray(arrays["idf"], dtype=np.float64)
        std = np.asarray(arrays["std"], dtype=np.float64)
        self.inverse_std = np.divide(1.0, std, out=np.zeros_like(std), where=std != 0)
        self.threshold = float(metadata["threshold"])
        if self.coefficients.shape != (metadata["feature_dimension"],) or len(self.feature_names()) != len(self.coefficients):
            raise ValueError("Coefficient shape does not match the feature layout")
        if len(self.idf) != len(self.vocabulary):
            raise ValueError("IDF length must equal the vocabulary size")

    @classmethod
    def load(cls, directory: Path | str) -> "BinaryPopularityModel":
        directory = Path(directory)
        metadata = json.loads((directory / "model.json").read_text(encoding="utf-8"))
        with np.load(directory / "model.npz") as arrays:
            return cls(metadata, {key: arrays[key] for key in arrays.files})

    @property
    def uses_time(self) -> bool:
        return self.features in NEEDS_TIME

    def text_vector(self, title: str) -> np.ndarray:
        tokens = tokenize(normalize_unicode(title).lower())
        vector = np.zeros(len(self.vocabulary))
        for term in [f"u:{t}" for t in tokens] + [f"b:{a} {b}" for a, b in zip(tokens, tokens[1:])]:
            index = self.vocabulary.get(term)
            if index is not None:
                vector[index] += 1.0
        vector *= self.idf
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector

    def feature_vector(self, title: str, created_utc: int | None = None) -> np.ndarray:
        if self.uses_time and created_utc is None:
            raise ValueError(f"{self.features} needs the posting time")
        raw = np.asarray(structured_values(title, 0 if created_utc is None else created_utc))
        scaled = raw * self.inverse_std
        if self.features == "time_features":
            return raw[TITLE_WIDTH:]
        if self.features == "structured_features":
            return scaled
        if self.features == "title_structured_features":
            return scaled[:TITLE_WIDTH]
        text = self.text_vector(title)
        if self.features == "text_features":
            return text
        return np.concatenate([text, scaled if self.features == "combined_features" else scaled[:TITLE_WIDTH]])

    def margin(self, title: str, created_utc: int | None = None) -> float:
        return float(self.coefficients @ self.feature_vector(title, created_utc) + self.intercept)

    def predict_proba(self, title: str, created_utc: int | None = None) -> float:
        """Probability of the HIGH class (Spark's probability[1])."""
        m = self.margin(title, created_utc)
        return 1.0 / (1.0 + math.exp(-m)) if m >= 0 else math.exp(m) / (1.0 + math.exp(m))

    def calibrated_probability(self, title: str, created_utc: int | None = None) -> float:
        """P(high) with the balanced-class-weight bias removed: logit + log(w_low / w_high).

        Weighted training scales the odds by w_high / w_low; the export stores the inverse
        as ``prior_logit_offset`` (0 when absent).
        """
        m = self.margin(title, created_utc) + float(self.metadata.get("prior_logit_offset", 0.0))
        return 1.0 / (1.0 + math.exp(-m)) if m >= 0 else math.exp(m) / (1.0 + math.exp(m))

    def predict(self, title: str, created_utc: int | None = None) -> dict:
        p = self.predict_proba(title, created_utc)
        return {"probability_high": p, "label": "high" if p >= self.threshold else "low", "threshold": self.threshold}

    def feature_names(self) -> list[str]:
        structured = self.metadata["structured_columns"]
        vocabulary = self.metadata["vocabulary"]
        return {"time_features": structured[TITLE_WIDTH:], "structured_features": structured,
                "title_structured_features": structured[:TITLE_WIDTH], "text_features": vocabulary,
                "combined_title_features": vocabulary + structured[:TITLE_WIDTH],
                "combined_features": vocabulary + structured}[self.features]

    def contributions(self, title: str, created_utc: int | None = None, top: int = 10) -> list[dict]:
        """Largest |coefficient × feature| terms of the high-vs-low margin (descriptive, not causal)."""
        vector = self.feature_vector(title, created_utc)
        product = self.coefficients * vector
        names = self.feature_names()
        order = [i for i in np.argsort(-np.abs(product)) if vector[i] != 0][:top]
        return [{"feature": names[i], "value": float(vector[i]), "contribution": float(product[i])} for i in order]

    def top_terms(self, top: int = 20) -> dict:
        if self.features not in ("text_features", "combined_title_features", "combined_features"):
            return {"high": [], "low": []}
        width = len(self.vocabulary)
        text = self.coefficients[:width]
        names = self.metadata["vocabulary"]
        order = np.argsort(text)
        return {"high": [{"term": names[i], "coefficient": float(text[i])} for i in order[::-1][:top]],
                "low": [{"term": names[i], "coefficient": float(text[i])} for i in order[:top]]}
