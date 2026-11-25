"""How much of a post's score can its title explain? Evidence from identical repeated titles.

Uses the cleaned dataset *before* title deduplication. For titles posted several
times in 2025 (same normalized text), the spread of scores across postings measures
everything that is not the title: timing, early votes, who saw it, removal. Reports:

- for repeated titles that produced at least one post with score >= 271, how often
  the typical (median) posting scored <= 2;
- the share of log(1 + score) variance explained by exact title identity within
  groups of at least ``min_group`` postings (an in-sample, optimistic upper bound
  for any title-only model on these titles);
- the largest groups with their score spread.

DuckDB only; writes one JSON file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb

KEY = "translate(lower(regexp_replace(trim(title), '\\s+', ' ', 'g')), '’‘“”', '''''\"\"')"
PLACEHOLDER = r"^\[\s*(removed|deleted)\b[^\]]*\]$"


def analyse(files: list[str], min_group: int = 5) -> dict:
    con = duckdb.connect(config={"threads": 2, "memory_limit": "1GB"})
    try:
        con.execute(f"""CREATE TEMP TABLE t AS SELECT {KEY} AS k, score, ln(1 + score) AS y FROM read_parquet(?)""", [files])
        con.execute(f"DELETE FROM t WHERE regexp_matches(k, '{PLACEHOLDER}')")
        con.execute("""CREATE TEMP TABLE g AS SELECT k, count(*) AS n, max(score) AS mx, quantile_disc(score, 0.5) AS med,
                       avg(y) AS gm FROM t GROUP BY k""")
        repeated = con.execute("""SELECT count(*), sum(n), count(*) FILTER (WHERE mx >= 271),
                                         count(*) FILTER (WHERE mx >= 271 AND med <= 2) FROM g WHERE n >= 2""").fetchone()
        explained = {}
        for size in (2, min_group):
            n, share = con.execute("""SELECT count(*), 1 - sum((t.y - g.gm)^2) / sum((t.y - m.m)^2)
                FROM t JOIN g USING (k), (SELECT avg(t.y) AS m FROM t JOIN g USING (k) WHERE g.n >= ?) m WHERE g.n >= ?""",
                [size, size]).fetchone()
            explained[f"groups_with_at_least_{size}_postings"] = {"postings": n, "variance_share_explained": share}
        largest = con.execute("""SELECT k, n, med, quantile_disc(score, 0.9), mx, avg((score >= 271)::INT)
            FROM t JOIN g USING (k) WHERE n >= 30 GROUP BY k, n, med, mx ORDER BY n DESC LIMIT 15""").fetchall()
        return {
            "repeated_titles": repeated[0], "postings_in_repeated_titles": repeated[1],
            "repeated_titles_with_a_post_ge_271": repeated[2],
            "of_which_median_posting_le_2": repeated[3],
            "share_of_viral_titles_whose_median_posting_le_2": repeated[3] / repeated[2] if repeated[2] else None,
            "variance_explained_by_title_identity": explained,
            "note": "In-sample group means (optimistic upper bound). Groups of 2 are heavily inflated; use the larger-group figure.",
            "largest_groups": [{"title_key": r[0], "postings": r[1], "median": r[2], "p90": r[3], "max": r[4], "share_ge_271": r[5]}
                               for r in largest],
        }
    finally:
        con.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default="data/processed/askreddit_clean")
    parser.add_argument("--output", default="reports/data_quality_dedup/title_ceiling.json")
    parser.add_argument("--min-group", type=int, default=5)
    args = parser.parse_args(argv)
    files = sorted(str(p) for p in Path(args.input).rglob("*.parquet"))
    result = analyse(files, args.min_group)
    Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "largest_groups"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
