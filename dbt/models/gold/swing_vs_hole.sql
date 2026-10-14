-- A driver swing video next to what happened on that hole: the card's textbook verdict,
-- where the tee shot finished, and the hole score. Golfzon videos carry the round
-- (session_key = pg_code) and hole number, which is the bridge.
SELECT
    v.video_id,
    h.pg_code,
    h.hole_no,
    h.par,
    v.ball_flight_code,
    v.carry_m,
    json_extract_scalar(c.evidence_json, '$._verdict.level')   AS verdict,
    c.fix_key,
    h.tee_lie,
    h.tee_lie_code IN (6, 9)                                  AS tee_penalty,
    h.score_to_par,
    h.penalties
FROM {{ source('silver', 'fact_video') }} v
JOIN {{ source('silver', 'fact_hole') }} h
  ON CAST(h.pg_code AS varchar) = v.session_key AND h.hole_no = v.hole
LEFT JOIN {{ source('silver', 'fact_card') }} c ON c.video_id = v.video_id
WHERE v.source = 'golfzon' AND v.club_group = 'driver'
