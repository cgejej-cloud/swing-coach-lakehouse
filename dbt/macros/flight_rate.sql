{# Share of shots with ball flight `code` among shots whose flight is known (1-3).
   Uploads carry no flight label and must not count as "not straight". #}
{% macro flight_rate(code, column='ball_flight_code') -%}
avg(CASE WHEN {{ column }} = {{ code }} THEN 1e0 WHEN {{ column }} IN (1, 2, 3) THEN 0e0 END)
{%- endmacro %}
