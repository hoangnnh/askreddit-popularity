-- Exact (interpolated) median per month.
SELECT month, median(score) AS median_score, max(score) AS max_score
FROM posts
GROUP BY month
ORDER BY month
