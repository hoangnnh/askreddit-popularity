-- Aggregate, join back to every row, aggregate again.
WITH monthly AS (SELECT month, median(score) AS median_score FROM posts GROUP BY month)
SELECT p.month, count(*) AS posts,
       sum(CASE WHEN p.score > m.median_score THEN 1 ELSE 0 END) AS above_month_median
FROM posts p JOIN monthly m ON p.month = m.month
GROUP BY p.month
ORDER BY p.month
