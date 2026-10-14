-- Every round in fact_round has exactly its number of holes in fact_hole.
SELECT r.pg_code, r.holes, count(h.hole_no) AS hole_rows
FROM {{ source('silver', 'fact_round') }} r
LEFT JOIN {{ source('silver', 'fact_hole') }} h ON h.pg_code = r.pg_code
GROUP BY r.pg_code, r.holes
HAVING count(h.hole_no) <> r.holes
