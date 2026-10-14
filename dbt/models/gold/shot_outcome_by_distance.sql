-- What happens to full shots from each distance and lie: reach the green, or find a penalty?
-- Excludes putts (start on the green) and penalty relocations (no swing).
WITH shots AS (
    SELECT
        s.*,
        CASE
            WHEN s.start_remain_m < 50  THEN '000-050'
            WHEN s.start_remain_m < 100 THEN '050-100'
            WHEN s.start_remain_m < 130 THEN '100-130'
            WHEN s.start_remain_m < 160 THEN '130-160'
            WHEN s.start_remain_m < 190 THEN '160-190'
            WHEN s.start_remain_m < 220 THEN '190-220'
            ELSE '220+'
        END AS distance_bucket,
        CASE
            WHEN s.start_lie_code IS NULL OR s.start_lie_code = 13 THEN 'tee'
            WHEN s.start_lie_code IN (1, 27)                      THEN 'fairway'
            WHEN s.start_lie_code IN (2, 23)                      THEN 'rough'
            WHEN s.start_lie_code IN (4, 5)                       THEN 'bunker'
            WHEN s.start_lie_code = 10                            THEN 'apron'
            ELSE 'other'
        END AS start_lie_group
    FROM {{ source('silver', 'fact_hole_shot') }} s
    WHERE s.is_relocation = 0 AND s.start_on_green = 0 AND s.start_remain_m IS NOT NULL
)
SELECT
    distance_bucket,
    start_lie_group,
    count(*)                                                    AS shots,
    avg(CASE WHEN lie_code IN (3, 8) THEN 1e0 ELSE 0e0 END)     AS green_rate,
    avg(CAST(is_penalty_lie AS double))                         AS penalty_rate,
    approx_percentile(end_remain_m, 0.5)                        AS median_end_remain_m
FROM shots
GROUP BY distance_bucket, start_lie_group
