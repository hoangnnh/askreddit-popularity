-- Group by a partition column.
SELECT month, count(*) AS posts, avg(score) AS mean_score,
       sum(CASE WHEN score >= 2 THEN 1 ELSE 0 END) AS score_ge_2
FROM posts
GROUP BY month
ORDER BY month
