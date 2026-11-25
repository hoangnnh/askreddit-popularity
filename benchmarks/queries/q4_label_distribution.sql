-- Chronological split and frozen eight-class labels (boundaries 2,3,4,6,11,46,271).
SELECT CASE WHEN created_utc < 1759276800 THEN 'train'
            WHEN created_utc < 1764547200 THEN 'validation'
            ELSE 'test' END AS split,
       CASE WHEN score < 2 THEN 0 WHEN score < 3 THEN 1 WHEN score < 4 THEN 2 WHEN score < 6 THEN 3
            WHEN score < 11 THEN 4 WHEN score < 46 THEN 5 WHEN score < 271 THEN 6 ELSE 7 END AS label,
       count(*) AS posts
FROM posts
GROUP BY 1, 2
ORDER BY 1, 2
