"""Train-fitted word TF-IDF and publication-time features; no target-derived inputs."""

import re
import unicodedata

from pyspark.ml import PipelineModel
from pyspark.ml.feature import (CountVectorizer, IDF, NGram, Normalizer, RegexTokenizer,
                                StandardScaler, VectorAssembler, VectorSlicer)
from pyspark.sql import functions as F, types as T

FEATURE_SOURCES = ("title", "created_utc")
TIME_COLUMNS = ["hour_sin", "hour_cos", "weekday_sin", "weekday_cos"]
STRUCTURED_COLUMNS = ["title_characters", "title_words", "question_count", "question_present", "uppercase_ratio", *TIME_COLUMNS]
FEATURE_COLUMNS = ("time_features", "text_features", "combined_features", "rf_features")


def normalize_unicode(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


def with_normalized_title(frame):
    normalized = F.udf(normalize_unicode, T.StringType(), useArrow=False)
    return frame.withColumn("normalized_title", F.lower(normalized("title")))


def validate_feature_sources(sources):
    if set(sources) != set(FEATURE_SOURCES):
        raise ValueError("Only title and created_utc may supply predictive features")


def base_features(frame, config):
    validate_feature_sources(config["feature_sources"])
    # Drop every outcome/audit column BEFORE deriving features or fitting vocabulary.
    frame = frame.select("id", "title", "created_utc")
    normalize = F.udf(normalize_unicode, T.StringType(), useArrow=False)
    frame = frame.withColumn("_unicode_title", normalize("title")).withColumn("normalized_title", F.lower("_unicode_title"))
    frame = RegexTokenizer(inputCol="normalized_title", outputCol="tokens", gaps=False,
                           pattern=config["token_pattern"], minTokenLength=1, toLowercase=False).transform(frame)
    frame = NGram(n=2, inputCol="tokens", outputCol="bigrams").transform(frame)
    letters = F.length(F.regexp_replace("_unicode_title", r"[^\p{L}]", ""))
    uppercase = F.length(F.regexp_replace("_unicode_title", r"[^\p{Lu}]", ""))
    hour = F.hour(F.timestamp_seconds("created_utc"))
    # Monday=0, Sunday=6, explicitly independent of locale.
    weekday = F.pmod(F.dayofweek(F.timestamp_seconds("created_utc")) + 5, F.lit(7))
    return frame.withColumns({
        "terms": F.concat(F.transform("tokens", lambda t: F.concat(F.lit("u:"), t)),
                          F.transform("bigrams", lambda t: F.concat(F.lit("b:"), t))),
        "title_characters": F.length("_unicode_title").cast("double"),
        "title_words": F.size("tokens").cast("double"),
        "question_count": F.length(F.regexp_replace("_unicode_title", r"[^?]", "")).cast("double"),
        "question_present": F.col("_unicode_title").contains("?").cast("double"),
        "uppercase_ratio": F.when(letters > 0, uppercase / letters).otherwise(F.lit(0.0)),
        "hour_sin": F.sin(hour * (2 * 3.141592653589793 / 24)),
        "hour_cos": F.cos(hour * (2 * 3.141592653589793 / 24)),
        "weekday_sin": F.sin(weekday * (2 * 3.141592653589793 / 7)),
        "weekday_cos": F.cos(weekday * (2 * 3.141592653589793 / 7)),
    })


def fit_preprocessing(train, config):
    from pyspark import StorageLevel
    if train.filter(F.col("split").isNull() | (F.col("split") != "train")).limit(1).count():
        raise ValueError("Preprocessing must be fitted on TRAIN only")
    prepared = base_features(train, config).persist(StorageLevel.DISK_ONLY)
    try:
        vocabulary = CountVectorizer(inputCol="terms", outputCol="term_counts", minDF=float(config["min_df"]),
                                     vocabSize=config["vocab_size"], binary=False).fit(prepared)
        width = len(vocabulary.vocabulary)
        if width == 0:
            raise ValueError("Training vocabulary is empty")
        idf = IDF(inputCol="term_counts", outputCol="raw_tfidf", minDocFreq=0).fit(vocabulary.transform(prepared))
        normalize = Normalizer(inputCol="raw_tfidf", outputCol="text_features", p=2.0)
        structured = VectorAssembler(inputCols=STRUCTURED_COLUMNS, outputCol="raw_structured")
        scaler = StandardScaler(inputCol="raw_structured", outputCol="structured_features",
                                withMean=False, withStd=True).fit(structured.transform(prepared))
        time = VectorAssembler(inputCols=TIME_COLUMNS, outputCol="time_features")
        rf_width = min(width, config["rf_text_features"])
        compact = VectorSlicer(inputCol="text_features", outputCol="compact_text", indices=list(range(rf_width)))
        forest = VectorAssembler(inputCols=["compact_text", "structured_features"], outputCol="rf_features")
        combined = VectorAssembler(inputCols=["text_features", "structured_features"], outputCol="combined_features")
        model = PipelineModel(stages=[vocabulary, idf, normalize, structured, scaler, time, compact, forest, combined])
        metadata = {
            "fitted_on": "train", "feature_sources": list(FEATURE_SOURCES), "config": config,
            "dimensions": {"time_features": 4, "text_features": width,
                           "combined_features": width + len(STRUCTURED_COLUMNS),
                           "rf_features": rf_width + len(STRUCTURED_COLUMNS)},
            "structured_columns": STRUCTURED_COLUMNS,
            "rf_selection": "First training-vocabulary entries (most frequent train terms); no test/validation statistics or labels used.",
            "text_weighting": "CountVectorizer term counts, train IDF, per-document L2 normalization",
            "scaling": "Structured features scaled by train standard deviation without mean centering; cyclic time-only inputs are bounded sin/cos.",
            "inference": "Call base_features(raw, saved_config) before the saved PipelineModel; source title is preserved.",
        }
        return model, metadata
    finally:
        prepared.unpersist()


def transform_features(frame, model, config):
    predictors = model.transform(base_features(frame, config)).select("id", *FEATURE_COLUMNS)
    # Join outcomes only after all feature transformations, for supervised fitting/evaluation.
    return predictors.join(frame.select("id", "label", "split", "title_seen_in_train"), "id")
