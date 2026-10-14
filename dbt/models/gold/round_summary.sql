-- One row per round and club group: what happened in that round.
SELECT
    v.session_key,
    max(v.session_label)                              AS session_label,
    min(v.shot_at)                                    AS first_shot_at,
    v.club_group,
    count(*)                                          AS shots,
    count(s.video_id)                                 AS analysed,
    {{ flight_rate(1, 'v.ball_flight_code') }}        AS straight_rate,
    {{ flight_rate(2, 'v.ball_flight_code') }}        AS slice_rate,
    {{ flight_rate(3, 'v.ball_flight_code') }}        AS hook_rate,
    avg(v.carry_m)                                    AS avg_carry_m,
    approx_percentile(s.m_tempo_ratio, 0.5)           AS median_tempo,
    approx_percentile(s.m_head_sway_top_sw, 0.5)      AS median_head_sway_top_sw,
    approx_percentile(s.m_pelvis_sway_top_sw, 0.5)    AS median_pelvis_sway_top_sw
FROM {{ source('silver', 'fact_video') }} v
LEFT JOIN {{ source('silver', 'fact_swing') }} s ON s.video_id = v.video_id
WHERE v.club_group IS NOT NULL
GROUP BY v.session_key, v.club_group
