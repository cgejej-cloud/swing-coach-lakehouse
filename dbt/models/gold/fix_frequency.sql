-- How often each fault is the headline fix on a card, per club group.
SELECT
    v.club_group,
    coalesce(c.fix_key, '(none)')                                       AS fix_key,
    count(*)                                                            AS cards,
    count(*) * 1e0 / sum(count(*)) OVER (PARTITION BY v.club_group)     AS share
FROM {{ source('silver', 'fact_card') }} c
JOIN {{ source('silver', 'fact_video') }} v ON v.video_id = c.video_id
WHERE v.club_group IS NOT NULL
GROUP BY v.club_group, coalesce(c.fix_key, '(none)')
