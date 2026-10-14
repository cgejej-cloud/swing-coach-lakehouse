-- Driver tee shots per round: how far left/right they land and where they end up.
-- x_m is metres from the centre line, + to the right. attr codes were matched against the
-- hole-detail lie of the same tee shot: 0 fairway, 1 rough, 2 bunker, 3 OB, 4 penalty area, 5 other.
SELECT
    d.pg_code,
    r.played_at,
    count(*)                                                 AS drives,
    avg(d.x_m)                                               AS mean_x_m,
    stddev(d.x_m)                                            AS sd_x_m,
    avg(d.y_m)                                               AS mean_y_m,
    avg(CASE WHEN d.attr = 0 THEN 1e0 ELSE 0e0 END)          AS fairway_rate,
    avg(CASE WHEN d.attr IN (3, 4) THEN 1e0 ELSE 0e0 END)    AS penalty_rate,
    sum(CASE WHEN d.x_m < -15 THEN 1 ELSE 0 END)             AS left_misses,
    sum(CASE WHEN d.x_m > 15 THEN 1 ELSE 0 END)              AS right_misses
FROM {{ source('silver', 'fact_driver_landing') }} d
LEFT JOIN {{ source('silver', 'fact_hole') }} h ON h.pg_code = d.pg_code AND h.hole_no = d.hole_no
LEFT JOIN {{ source('silver', 'fact_round') }} r ON r.pg_code = d.pg_code
WHERE h.par IS NULL OR h.par >= 4
GROUP BY d.pg_code, r.played_at
