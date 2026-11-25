-- Full scan with global aggregates.
SELECT count(*) AS posts, min(created_utc) AS first_utc, max(created_utc) AS last_utc,
       sum(score) AS total_score, max(score) AS max_score
FROM posts
