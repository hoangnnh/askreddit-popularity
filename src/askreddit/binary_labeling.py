"""Train-quantile binary popularity labels from the registered proposal.

Quantiles are fitted on the training split only, optionally restricted to a score
population (``population_min_score``). Two tie conventions are supported because
AskReddit scores are heavily tied:

- v3: all posts, ``low = score <= median`` and ``high = score > p75``.
- v4: posts with ``score >= 2`` only (the upper remainder on which Fang et al. and
  Chua compute their medians), ``low = score < median`` and ``high = score >= p75``,
  the literal "below the median" / "top quartile" wording.

- Fixed cuts (``le_value`` / ``ge_value`` with ``low_value`` / ``high_value``), e.g. the
  engagement stage ``low = score <= 1``, ``high = score >= 2``. Quantiles are still
  fitted and reported.

Scores outside the population are ``excluded``; scores between the groups are the
excluded ``middle``. Neither is used for training or evaluation.
"""

from fractions import Fraction

from pyspark.sql import DataFrame, functions as F

from askreddit.labeling import SPLITS

VERSION = "binary-train-quantile-v1"
GROUPS = ("excluded", "low", "middle", "high")
QUANTILE_DEFINITION = "Inverse empirical CDF: the smallest train score s with F(s) >= q, from exact integer counts."
DEFAULT_RULES = {"population_min_score": None, "low_rule": "le_median", "high_rule": "gt_p75",
                 "low_value": None, "high_value": None}
LOW_RULES = {"le_median": "<=", "lt_median": "<", "le_value": "<="}
HIGH_RULES = {"gt_p75": ">", "ge_p75": ">=", "ge_value": ">="}
OPERATORS = {"<=": lambda a, b: a <= b, "<": lambda a, b: a < b, ">": lambda a, b: a > b, ">=": lambda a, b: a >= b}


def label_rules(config: dict | None) -> dict:
    rules = {**DEFAULT_RULES, **{k: v for k, v in (config or {}).items() if k in DEFAULT_RULES}}
    if rules["low_rule"] not in LOW_RULES or rules["high_rule"] not in HIGH_RULES:
        raise ValueError(f"Unsupported label rules: {rules}")
    for key in ("population_min_score", "low_value", "high_value"):
        if rules[key] is not None and not isinstance(rules[key], int):
            raise ValueError(f"{key} must be an integer or null")
    if (rules["low_rule"] == "le_value") != (rules["low_value"] is not None) or \
            (rules["high_rule"] == "ge_value") != (rules["high_value"] is not None):
        raise ValueError("Fixed rules need their value; quantile rules must not set one")
    return rules


def cuts(rules: dict, median: int, upper: int) -> tuple[int, int]:
    low = rules["low_value"] if rules["low_rule"] == "le_value" else median
    high = rules["high_value"] if rules["high_rule"] == "ge_value" else upper
    return int(low), int(high)


def quantile_from_histogram(histogram, q: Fraction) -> int:
    """Exact inverse-ECDF quantile from (score, count) pairs."""
    pairs = sorted((int(score), int(count)) for score, count in histogram)
    total = sum(count for _, count in pairs)
    if not pairs or total <= 0 or any(count <= 0 for _, count in pairs) or not 0 < q <= 1:
        raise ValueError("A nonempty positive histogram and 0 < q <= 1 are required")
    cumulative = 0
    for score, count in pairs:
        cumulative += count
        if cumulative * q.denominator >= q.numerator * total:
            return score
    raise AssertionError("Unreachable: cumulative count reaches the total")


def is_low(score, cut, rule):
    return OPERATORS[LOW_RULES[rule]](score, cut)


def is_high(score, cut, rule):
    return OPERATORS[HIGH_RULES[rule]](score, cut)


def thresholds_from_histogram(histogram, rules: dict | None = None) -> dict:
    rules = label_rules(rules)
    minimum = rules["population_min_score"]
    all_pairs = sorted((int(s), int(c)) for s, c in histogram)
    pairs = [(s, c) for s, c in all_pairs if minimum is None or s >= minimum]
    median = quantile_from_histogram(pairs, Fraction(1, 2))
    upper = quantile_from_histogram(pairs, Fraction(3, 4))
    low_cut, high_cut = cuts(rules, median, upper)
    if any(is_low(s, low_cut, rules["low_rule"]) and is_high(s, high_cut, rules["high_rule"]) for s in (low_cut, high_cut)):
        raise ValueError(f"Low and high groups overlap: low cut={low_cut}, high cut={high_cut}, rules={rules}")
    population = sum(c for _, c in pairs)
    low = sum(c for s, c in pairs if is_low(s, low_cut, rules["low_rule"]))
    high = sum(c for s, c in pairs if is_high(s, high_cut, rules["high_rule"]))
    excluded = sum(c for _, c in all_pairs) - population
    if low == 0 or high == 0:
        raise ValueError(f"Degenerate binary labels: median={median}, p75={upper}, low={low}, high={high}")
    population_text = "all train posts" if minimum is None else f"train posts with score >= {minimum}"
    counts = {"excluded": excluded, "low": low, "middle": population - low - high, "high": high}
    low_source = "fixed" if rules["low_rule"] == "le_value" else f"median of {population_text}"
    high_source = "fixed" if rules["high_rule"] == "ge_value" else f"p75 of {population_text}"
    return {
        "algorithm_version": VERSION, "fitted_on": "train", "quantile_definition": QUANTILE_DEFINITION,
        "rules": rules, "population": population_text, "train_records": sum(c for _, c in all_pairs),
        "population_records": population, "median": median, "p75": upper, "low_cut": low_cut, "high_cut": high_cut,
        "low_rule": f"score {LOW_RULES[rules['low_rule']]} {low_cut} ({low_source})",
        "high_rule": f"score {HIGH_RULES[rules['high_rule']]} {high_cut} ({high_source})",
        "middle_rule": "population scores in neither group; excluded from training and evaluation",
        "train_counts": counts,
        "train_percentages_of_population": {g: 100 * counts[g] / population for g in ("low", "middle", "high")},
    }


def fit_binary_thresholds(train: DataFrame, rules: dict | None = None) -> dict:
    if not {"score", "split"}.issubset(train.columns):
        raise ValueError("Threshold fitting requires score and an explicit train split")
    if train.filter(F.col("split").isNull() | (F.col("split") != "train") | F.col("score").isNull()).limit(1).count():
        raise ValueError("Thresholds must be fitted on valid TRAIN scores only")
    if dict(train.dtypes)["score"] not in {"bigint", "int"}:
        raise ValueError("Exact quantiles require integer scores")
    # Only distinct scores and their counts reach the driver (a few thousand rows).
    return thresholds_from_histogram(((r.score, r["count"]) for r in train.groupBy("score").count().collect()), rules)


def model_cuts(model: dict) -> tuple[dict, int, int]:
    rules = label_rules(model.get("rules"))
    return rules, int(model.get("low_cut", model["median"])), int(model.get("high_cut", model["p75"]))


def group_expression(model: dict):
    rules, low_cut, high_cut = model_cuts(model)
    score = F.col("score")
    low = OPERATORS[LOW_RULES[rules["low_rule"]]](score, low_cut)
    high = OPERATORS[HIGH_RULES[rules["high_rule"]]](score, high_cut)
    expression = F.when(low, F.lit("low")).when(high, F.lit("high")).when(score.isNotNull(), F.lit("middle"))
    if rules["population_min_score"] is not None:
        expression = F.when(score < rules["population_min_score"], F.lit("excluded")).otherwise(expression)
    return expression


def group_sql(model: dict) -> str:
    """The same rule as a DuckDB/Spark SQL CASE expression, for independent verification."""
    rules, low_cut, high_cut = model_cuts(model)
    excluded = "" if rules["population_min_score"] is None else f"WHEN score < {int(rules['population_min_score'])} THEN 'excluded' "
    return (f"CASE {excluded}WHEN score {LOW_RULES[rules['low_rule']]} {low_cut} THEN 'low' "
            f"WHEN score {HIGH_RULES[rules['high_rule']]} {high_cut} THEN 'high' ELSE 'middle' END")


def apply_binary_labels(frame: DataFrame, model: dict) -> DataFrame:
    group = group_expression(model)
    return frame.withColumn("popularity_group", group).withColumn(
        "label", F.when(group == "high", 1.0).when(group == "low", 0.0).cast("double"))


def group_distributions(frame: DataFrame) -> dict:
    counts = {(r.split, r.popularity_group): r["count"] for r in frame.groupBy("split", "popularity_group").count().collect()}
    if any(group not in GROUPS for _, group in counts):
        raise ValueError("Unexpected/null popularity group")
    result = {}
    for split in SPLITS:
        total = sum(counts.get((split, g), 0) for g in GROUPS)
        kept = counts.get((split, "low"), 0) + counts.get((split, "high"), 0)
        result[split] = {
            "total": total, "kept": kept,
            "groups": {g: {"count": counts.get((split, g), 0),
                           "percentage_of_split": 100 * counts.get((split, g), 0) / total if total else None} for g in GROUPS},
            "positive_rate_among_kept": counts.get((split, "high"), 0) / kept if kept else None,
        }
    return result


def binary_class_weights(counts: list[int]) -> list[float]:
    """Balanced weights n / (2 n_k); their training-row mean is exactly 1."""
    if len(counts) != 2 or min(counts) <= 0:
        raise ValueError("Weights require two nonempty training classes")
    n = sum(counts)
    return [n / (2 * c) for c in counts]
