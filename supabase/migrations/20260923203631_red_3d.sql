-- `red_3d()`: insumo del mapa 3D del dashboard. Aplicada el 2026-09-23.
--
-- Devuelve, en una sola llamada:
--   * las 12 estaciones con su corredor y sus coordenadas reales (para ubicarlas en el plano),
--   * el perfil de demanda promedio por estación y media hora local (48 franjas), separado en
--     día hábil y fin de semana,
--   * el accuracy oficial acumulado de cada estación, para poder colorear por desempeño.
--
-- Por qué una función aparte de `dashboard_public()`: este payload es un agregado histórico que
-- casi no cambia, y el dashboard se refresca cada minuto. Se pide una sola vez al cargar; así el
-- refresco periódico sigue siendo liviano.
--
-- Privacidad: solo agregados del histórico que el propio reto publica, más nuestro desempeño.
-- No hay datos de terceros ni filas crudas; las tablas siguen con RLS y sin permisos para `anon`.
--
-- Las horas se pasan a America/Bogota antes de agrupar: el eje del tiempo debe ser la hora a la
-- que la gente realmente viaja, no UTC.

create or replace function public.red_3d()
returns jsonb
language sql
stable
security definer
set search_path to public
as $function$
with marcado as (
  select o.station_id,
         o.demand,
         extract(hour from o.observed_at at time zone 'America/Bogota')::int * 2
           + extract(minute from o.observed_at at time zone 'America/Bogota')::int / 30 as slot,
         case when extract(isodow from o.observed_at at time zone 'America/Bogota') < 6
              then 'habil' else 'finde' end as tipo
  from observations o
),
-- Rejilla completa: cada estación debe traer las 48 franjas aunque falte alguna observación,
-- porque el cliente indexa la serie por posición.
rejilla as (
  select s.station_id, g.slot, d.tipo
  from stations s
  cross join generate_series(0, 47) as g(slot)
  cross join (values ('habil'), ('finde')) as d(tipo)
),
perfil as (
  select r.station_id, r.slot, r.tipo,
         round(coalesce(avg(m.demand), 0)::numeric, 1) as demanda
  from rejilla r
  left join marcado m
    on m.station_id = r.station_id and m.slot = r.slot and m.tipo = r.tipo
  group by r.station_id, r.slot, r.tipo
),
acc as (
  select station_id,
         round((100 * greatest(0, 1 - sum(abs(actual - predicted))
                                    / nullif(sum(actual), 0)))::numeric, 2) as accuracy,
         count(*) as n
  from prediction_results
  where is_official and actual is not null
  group by station_id
)
select jsonb_build_object(
  'generado', now(),
  'ventana', (select jsonb_build_object(
       'desde', min(observed_at),
       'hasta', max(observed_at),
       'dias', round(extract(epoch from max(observed_at) - min(observed_at))::numeric / 86400),
       'observaciones', count(*))
     from observations),
  'estaciones', (select jsonb_agg(jsonb_build_object(
       'id', s.station_id,
       'nombre', s.station_name,
       'corredor', s.corridor,
       'lat', s.latitude,
       'lon', s.longitude,
       'accuracy', a.accuracy,
       'evaluados', coalesce(a.n, 0),
       'pico', (select max(p.demanda) from perfil p
                where p.station_id = s.station_id and p.tipo = 'habil'),
       'media', (select round(avg(p.demanda), 1) from perfil p
                 where p.station_id = s.station_id and p.tipo = 'habil'))
       order by s.corridor, s.station_name)
     from stations s left join acc a on a.station_id = s.station_id),
  'perfil', (select jsonb_object_agg(tipo, series) from (
       select tipo, jsonb_object_agg(station_id, serie) as series from (
         select tipo, station_id, jsonb_agg(demanda order by slot) as serie
         from perfil group by tipo, station_id
       ) x group by tipo
     ) y)
);
$function$;

comment on function public.red_3d() is
  'Perfil agregado de demanda por estación y media hora local, con coordenadas y accuracy. Insumo del mapa 3D del dashboard.';

revoke all on function public.red_3d() from public;
grant execute on function public.red_3d() to anon, authenticated, service_role;
