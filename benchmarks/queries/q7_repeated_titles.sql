-- High-cardinality GROUP BY on a string column, then top-k.
SELECT title, count(*) AS posts
FROM posts
GROUP BY title
HAVING count(*) > 1
ORDER BY posts DESC, title
LIMIT 20
