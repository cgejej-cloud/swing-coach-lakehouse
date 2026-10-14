SELECT 'shot_outcome_by_distance' AS model, green_rate AS rate
FROM {{ ref('shot_outcome_by_distance') }}
WHERE green_rate NOT BETWEEN 0 AND 1 OR penalty_rate NOT BETWEEN 0 AND 1
UNION ALL
SELECT 'driver_dispersion', fairway_rate
FROM {{ ref('driver_dispersion') }}
WHERE fairway_rate NOT BETWEEN 0 AND 1 OR penalty_rate NOT BETWEEN 0 AND 1
