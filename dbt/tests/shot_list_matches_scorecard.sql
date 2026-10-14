-- Warn: swings + penalty relocations listed for a hole should equal the scorecard strokes,
-- or one less (the hole-detail list sometimes omits the final putt).
{{ config(severity = 'warn') }}
SELECT h.pg_code, h.hole_no, h.strokes, count(s.shot_no) AS listed
FROM {{ source('silver', 'fact_hole') }} h
LEFT JOIN {{ source('silver', 'fact_hole_shot') }} s ON s.pg_code = h.pg_code AND s.hole_no = h.hole_no
GROUP BY h.pg_code, h.hole_no, h.strokes
HAVING h.strokes - count(s.shot_no) NOT IN (0, 1)
