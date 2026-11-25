-- Derived UTC weekday (Monday=0) and hour with portable integer arithmetic.
SELECT CAST((floor(created_utc / 86400) + 3) % 7 AS INTEGER) AS weekday,
       CAST(floor(created_utc / 3600) % 24 AS INTEGER) AS hour,
       count(*) AS posts, avg(score) AS mean_score
FROM posts
GROUP BY 1, 2
ORDER BY 1, 2
