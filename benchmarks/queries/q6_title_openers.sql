-- String functions over every title. Integer CASE values: a 1.0 literal is DECIMAL in Spark (avg keeps 5 decimals) but yields a double in DuckDB.
SELECT CASE WHEN lower(title) LIKE 'what%' THEN 'what'
            WHEN lower(title) LIKE 'why%' THEN 'why'
            WHEN lower(title) LIKE 'how%' THEN 'how'
            ELSE 'other' END AS opener,
       count(*) AS posts, avg(length(title)) AS mean_chars,
       avg(CASE WHEN score >= 2 THEN 1 ELSE 0 END) AS share_score_ge_2
FROM posts
GROUP BY 1
ORDER BY 1
