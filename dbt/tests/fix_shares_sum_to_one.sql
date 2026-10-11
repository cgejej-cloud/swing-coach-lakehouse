-- Within each club group the fix shares must add up to 1.
SELECT club_group, sum(share) AS total
FROM {{ ref('fix_frequency') }}
GROUP BY club_group
HAVING abs(sum(share) - 1) > 1e-9
