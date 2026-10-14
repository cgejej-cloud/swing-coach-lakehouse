-- Long format: metric distribution per club group and ball flight (feeds the "why" analysis).
-- Only trustworthy swings: HD video and a plausible body scale.
{% set metrics = ['tempo_ratio', 'stance_width_sw', 'trail_shoulder_drop_deg', 'head_sway_top_sw',
                  'pelvis_sway_top_sw', 'head_move_imp_sw', 'pelvis_shift_imp_sw', 'lead_knee_imp_deg',
                  'hands_ahead_imp_sw', 'pelvis_over_lead_foot_fin_sw'] %}
SELECT
    v.club_group,
    v.ball_flight_code,
    m.metric,
    count(*)                              AS n,
    approx_percentile(m.value, 0.5)       AS median,
    avg(m.value)                          AS mean
FROM {{ source('silver', 'fact_video') }} v
JOIN {{ source('silver', 'fact_swing') }} s ON s.video_id = v.video_id
CROSS JOIN UNNEST(
    ARRAY[{% for m in metrics %}'{{ m }}'{% if not loop.last %}, {% endif %}{% endfor %}],
    ARRAY[{% for m in metrics %}s.m_{{ m }}{% if not loop.last %}, {% endif %}{% endfor %}]
) AS m(metric, value)
WHERE v.club_group IS NOT NULL AND v.width >= 1280 AND s.m_scale_ok = 1
GROUP BY v.club_group, v.ball_flight_code, m.metric
