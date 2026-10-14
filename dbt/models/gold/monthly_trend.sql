-- Monthly trend of the metrics that drive the cards.
SELECT
    date_trunc('month', CAST(substr(v.shot_at, 1, 19) AS timestamp))   AS month,
    v.club_group,
    count(*)                                                            AS shots,
    {{ flight_rate(1, 'v.ball_flight_code') }}                          AS straight_rate,
    approx_percentile(s.m_pelvis_sway_top_sw, 0.5)                      AS median_pelvis_sway_top_sw,
    approx_percentile(s.m_tempo_ratio, 0.5)                             AS median_tempo
FROM {{ source('silver', 'fact_video') }} v
JOIN {{ source('silver', 'fact_swing') }} s ON s.video_id = v.video_id
WHERE v.club_group IS NOT NULL
GROUP BY 1, 2
