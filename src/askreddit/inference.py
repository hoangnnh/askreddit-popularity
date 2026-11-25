"""Spark-free inference for an exported multinomial logistic-regression popularity model.

Reimplements, in NumPy, exactly the fitted transformations of ``features.py``:
NFKC + whitespace collapse, lowercase, the Unicode word tokenizer, unigram and
bigram counts over the frozen training vocabulary, training IDF, per-document L2
normalization, nine structured features scaled by training standard deviations,
and UTC cyclic hour/weekday. ``export_model`` verifies parity against Spark.
Only ``title`` and ``created_utc`` are inputs; the score is never used.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import unicodedata

import numpy as np

# Python equivalent of the Java pattern [\p{L}\p{N}]+(?:['’][\p{L}\p{N}]+)*:
# [^\W_] is a Unicode letter or number (\w without underscore).
TOKEN_PATTERN = re.compile(r"[^\W_]+(?:['’][^\W_]+)*")
TWO_PI_HOUR = 2 * 3.141592653589793 / 24
TWO_PI_WEEKDAY = 2 * 3.141592653589793 / 7
SUPPORTED_FEATURES = ("time_features", "structured_features", "text_features", "combined_features")


def normalize_unicode(text: str) -> str:
    """Identical to askreddit.features.normalize_unicode (tested for equality)."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


def tokenize(normalized_lower: str) -> list[str]:
    return TOKEN_PATTERN.findall(normalized_lower)


def structured_values(title: str, created_utc: int) -> list[float]:
    """Raw (unscaled) STRUCTURED_COLUMNS in feature order."""
    unicode_title = normalize_unicode(title)
    tokens = tokenize(unicode_title.lower())
    letters = sum(1 for ch in unicode_title if unicodedata.category(ch).startswith("L"))
    upper = sum(1 for ch in unicode_title if unicodedata.category(ch) == "Lu")
    moment = datetime.fromtimestamp(int(created_utc), timezone.utc)
    hour, weekday = moment.hour, moment.weekday()  # Monday=0, as in features.base_features
    return [float(len(unicode_title)), float(len(tokens)), float(unicode_title.count("?")),
            1.0 if "?" in unicode_title else 0.0, upper / letters if letters else 0.0,
            math.sin(hour * TWO_PI_HOUR), math.cos(hour * TWO_PI_HOUR),
            math.sin(weekday * TWO_PI_WEEKDAY), math.cos(weekday * TWO_PI_WEEKDAY)]


class PopularityModel:
    def __init__(self, metadata: dict, arrays: dict):
        if metadata["features"] not in SUPPORTED_FEATURES:
            raise ValueError(f"Unsupported feature set for Spark-free inference: {metadata['features']}")
        self.metadata = metadata
        self.vocabulary = {term: index for index, term in enumerate(metadata["vocabulary"])}
        self.coefficients = np.asarray(arrays["coefficients"], dtype=np.float64)
        self.intercepts = np.asarray(arrays["intercepts"], dtype=np.float64)
        self.idf = np.asarray(arrays["idf"], dtype=np.float64)
        std = np.asarray(arrays["std"], dtype=np.float64)
        self.inverse_std = np.divide(1.0, std, out=np.zeros_like(std), where=std != 0)
        if self.coefficients.shape != (8, metadata["feature_dimension"]) or self.intercepts.shape != (8,):
            raise ValueError("Coefficient shapes do not match the eight-class feature layout")
        if len(self.idf) != len(self.vocabulary):
            raise ValueError("IDF length must equal the vocabulary size")

    @classmethod
    def load(cls, directory: Path | str) -> "PopularityModel":
        directory = Path(directory)
        metadata = json.loads((directory / "model.json").read_text(encoding="utf-8"))
        with np.load(directory / "model.npz") as arrays:
            return cls(metadata, {key: arrays[key] for key in arrays.files})

    def text_vector(self, title: str) -> tuple[np.ndarray, list[str]]:
        tokens = tokenize(normalize_unicode(title).lower())
        terms = [f"u:{t}" for t in tokens] + [f"b:{a} {b}" for a, b in zip(tokens, tokens[1:])]
        vector = np.zeros(len(self.vocabulary))
        for term in terms:
            index = self.vocabulary.get(term)
            if index is not None:
                vector[index] += 1.0
        vector *= self.idf
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        return vector, terms

    def feature_vector(self, title: str, created_utc: int) -> np.ndarray:
        raw = np.asarray(structured_values(title, created_utc))
        kind = self.metadata["features"]
        if kind == "time_features":
            return raw[5:]
        structured = raw * self.inverse_std
        if kind == "structured_features":
            return structured
        text, _ = self.text_vector(title)
        return text if kind == "text_features" else np.concatenate([text, structured])

    def predict_proba(self, title: str, created_utc: int) -> np.ndarray:
        margins = self.coefficients @ self.feature_vector(title, created_utc) + self.intercepts
        exp = np.exp(margins - margins.max())
        return exp / exp.sum()

    def predict(self, title: str, created_utc: int) -> dict:
        probabilities = self.predict_proba(title, created_utc)
        label = int(np.argmax(probabilities))
        return {"label": label, "probabilities": probabilities.tolist(),
                "score_range": self.metadata["label_mapping"][label]["score_range"]}

    def feature_names(self) -> list[str]:
        structured = self.metadata["structured_columns"]
        kind = self.metadata["features"]
        if kind == "time_features":
            return structured[5:]
        if kind == "structured_features":
            return structured
        return self.metadata["vocabulary"] + (structured if kind == "combined_features" else [])

    def contributions(self, title: str, created_utc: int, label: int, top: int = 8) -> list[dict]:
        """Largest |coefficient × feature| terms for one class margin (descriptive, not causal)."""
        vector = self.feature_vector(title, created_utc)
        product = self.coefficients[label] * vector
        names = self.feature_names()
        order = [i for i in np.argsort(-np.abs(product)) if vector[i] != 0][:top]
        return [{"feature": names[i], "value": float(vector[i]), "contribution": float(product[i])} for i in order]

    def top_terms(self, label: int, top: int = 15) -> list[dict]:
        """Text terms with the largest coefficient for ``label`` relative to the mean over classes."""
        if self.metadata["features"] not in ("text_features", "combined_features"):
            return []
        width = len(self.vocabulary)
        relative = self.coefficients[label, :width] - self.coefficients[:, :width].mean(axis=0)
        order = np.argsort(-relative)[:top]
        return [{"term": self.metadata["vocabulary"][i], "relative_coefficient": float(relative[i])} for i in order]
