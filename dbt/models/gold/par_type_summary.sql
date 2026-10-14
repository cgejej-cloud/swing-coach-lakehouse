-- Where strokes go, by hole type: one row per round and par.
SELECT
    h.pg_code,
    r.played_at,
    r.course,
    h.par,
    count(*)                                   AS holes,
    avg(h.strokes)                             AS avg_strokes,
    sum(h.score_to_par)                        AS strokes_over_par,
    sum(h.penalties)                           AS penalty_strokes,
    avg(h.putts)                               AS putts_per_hole,
    avg(CAST(h.gir AS double))                 AS gir_rate,
    avg(CAST(h.fairway_hit AS double))         AS fairway_rate   -- NULL for par 3
FROM {{ source('silver', 'fact_hole') }} h
JOIN {{ source('silver', 'fact_round') }} r ON r.pg_code = h.pg_code
GROUP BY h.pg_code, r.played_at, r.course, h.par
