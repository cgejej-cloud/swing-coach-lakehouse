-- Flight rates are shares of labelled shots: NULL (no labelled shot) or within [0, 1].
SELECT *
FROM {{ ref('round_summary') }}
WHERE straight_rate NOT BETWEEN 0 AND 1
   OR slice_rate NOT BETWEEN 0 AND 1
   OR hook_rate NOT BETWEEN 0 AND 1
   OR abs(coalesce(straight_rate + slice_rate + hook_rate, 1) - 1) > 1e-9
