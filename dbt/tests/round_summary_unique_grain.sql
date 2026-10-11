-- Grain check: one row per (session_key, club_group).
SELECT session_key, club_group, count(*) AS n
FROM {{ ref('round_summary') }}
GROUP BY session_key, club_group
HAVING count(*) > 1
