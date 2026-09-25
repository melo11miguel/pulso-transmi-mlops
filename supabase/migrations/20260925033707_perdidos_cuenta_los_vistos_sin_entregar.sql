-- `dashboard_state()`: contar como perdido tambien el ciclo que se vio y nunca se entrego.
--
-- Contexto: el 2026-09-24 se perdieron dos ciclos porque la sesion de prediccion murio, y el KPI
-- del dashboard mostro "0 ciclos perdidos" mientras pasaba. El ciclo 07:00Z quedo en `seen` y el
-- contador solo miraba `missed` y `error`. Un ciclo ya cerrado que no quedo en `submitted` esta
-- perdido, sin importar en que estado se detuvo la maquina de estados.
--
--   antes:  count(*) filter (where outcome in ('missed','error'))               -> 0
--   ahora:  count(*) filter (where outcome <> 'submitted' and closes_at < now()) -> 1
--
-- Limite conocido: esta tabla solo conoce los ciclos que alguna ejecucion alcanzo a ver. El ciclo
-- 08:00Z nunca se registro porque no habia proceso vivo, asi que aqui sale 1 y no 2. La cifra
-- autoritativa de cobertura es la de /v1/leaderboard, que el monitor guarda en
-- `leaderboard_snapshots` y el dashboard muestra como KPI.

create or replace function public.dashboard_state()
returns jsonb
language sql
stable
set search_path to public
as $function$
with resolved as (
  select station_id, horizon_minutes, target_at, predicted, actual
  from prediction_results
  where is_official and actual is not null
),
per_station as (
  select station_id,
         sum(abs(actual - predicted)) / nullif(sum(actual), 0) as wape,
         count(*) as n
  from resolved group by station_id
),
per_horizon as (
  select horizon_minutes,
         sum(abs(actual - predicted)) / nullif(sum(actual), 0) as wape,
         count(*) as n
  from resolved group by horizon_minutes
),
per_hour_station as (
  select date_trunc('hour', target_at) as hora, station_id,
         sum(abs(actual - predicted)) / nullif(sum(actual), 0) as wape
  from resolved group by 1, 2
),
serie as (
  select hora, round(avg(100 * greatest(0, 1 - wape))::numeric, 2) as accuracy
  from per_hour_station group by hora order by hora
),
ultimo_nivel as (
  select distinct on (station_id) station_id, value, threshold, triggered,
         (details->>'streak')::int as streak
  from drift_signals where kind = 'data' and name = 'level_shift'
  order by station_id, computed_at desc
)
select jsonb_build_object(
  'generated_at', now(),
  'accuracy_global', (select round(avg(100 * greatest(0, 1 - wape))::numeric, 2) from per_station),
  'targets_evaluados', (select count(*) from resolved),
  'por_estacion', (select jsonb_agg(jsonb_build_object(
       'station_id', s.station_id, 'nombre', st.station_name,
       'accuracy', round((100 * greatest(0, 1 - s.wape))::numeric, 2), 'n', s.n) order by s.wape desc)
     from per_station s left join stations st on st.station_id = s.station_id),
  'por_horizonte', (select jsonb_agg(jsonb_build_object(
       'horizonte', horizon_minutes,
       'accuracy', round((100 * greatest(0, 1 - wape))::numeric, 2), 'n', n) order by horizon_minutes)
     from per_horizon),
  'serie_accuracy', (select jsonb_agg(jsonb_build_object('hora', hora, 'accuracy', accuracy)) from serie),
  'ciclos', (select jsonb_build_object(
       'vistos', count(*),
       'entregados', count(*) filter (where outcome = 'submitted'),
       -- Perdido = ya cerro y no quedo entregado. Incluye `seen`, que antes no se contaba.
       'perdidos', count(*) filter (where outcome <> 'submitted' and closes_at < now()),
       'ultimo_cierre', max(closes_at)) from forecast_cycles),
  'modelo', (select jsonb_build_object(
       'version', version, 'estado', status, 'entrenado', trained_at,
       'datos_hasta', training_data_end, 'commit', left(coalesce(git_commit,''), 7),
       'accuracy_validacion', round((validation->>'accuracy')::numeric, 2),
       'baseline_validacion', round((validation->>'baseline_profile_only_accuracy')::numeric, 2))
     from model_versions where status = 'champion'),
  'versiones', (select jsonb_agg(jsonb_build_object(
       'version', version, 'estado', status,
       'accuracy', round((validation->>'accuracy')::numeric, 2), 'razon', reason) order by created_at desc)
     from (select * from model_versions order by created_at desc limit 8) v),
  'drift', (select jsonb_agg(jsonb_build_object(
       'station_id', station_id, 'desplazamiento', round(value::numeric, 3),
       'umbral', round(threshold::numeric, 3), 'racha', streak, 'activo', triggered)
       order by abs(value) desc nulls last) from ultimo_nivel),
  'decisiones', (select jsonb_agg(jsonb_build_object(
       'cuando', decided_at, 'decision', decision, 'razon', reason) order by decided_at desc)
     from (select * from retrain_decisions order by decided_at desc limit 6) d),
  'operacion', jsonb_build_object(
     'ultimo_dato', (select max(observed_at) from observations),
     'ultima_ingesta', (select max(finished_at) from ingestion_runs where status = 'success'),
     'ultima_entrega', (select max(created_at) from submissions where status = 'accepted'),
     'ultimo_monitoreo', (select max(finished_at) from pipeline_runs where job = 'monitor' and status = 'success'),
     'ultimo_entrenamiento', (select max(finished_at) from pipeline_runs where job = 'train' and status = 'success'),
     'errores_24h', (select count(*) from pipeline_runs where status = 'error' and started_at > now() - interval '24 hours')
       + (select count(*) from ingestion_runs where status = 'error' and started_at > now() - interval '24 hours'))
);
$function$;
