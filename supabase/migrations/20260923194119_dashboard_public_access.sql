-- Acceso público de solo lectura para el dashboard, sin copiar credenciales a terceros.
--
-- 1. `leaderboard_snapshots`: el monitor (que tiene PULSO_API_KEY) guarda aquí nuestra posición.
--    Sin nombres de terceros: solo nuestro puesto y el mejor accuracy de la cohorte.
-- 2. `dashboard_public()`: security definer, ejecutable por `anon`. Envuelve a `dashboard_state()`
--    y le agrega el leaderboard. Es lo ÚNICO que la clave publicable puede hacer: las tablas
--    conservan RLS sin políticas y sin permisos para anon.
--
-- Comprobado con la clave publicable:
--   rpc/dashboard_public -> 200 · rpc/dashboard_state -> 401
--   observations, submissions, model_versions, leaderboard_snapshots -> 401

create table if not exists public.leaderboard_snapshots (
  id bigint generated always as identity primary key,
  captured_at timestamptz not null default now(),
  window_name text not null,
  rank integer,
  participants integer,
  accuracy double precision,
  coverage double precision,
  best_accuracy double precision,
  best_coverage double precision
);

create index if not exists leaderboard_snapshots_window_idx
  on public.leaderboard_snapshots (window_name, captured_at desc);

alter table public.leaderboard_snapshots enable row level security;

create or replace function public.dashboard_public()
returns jsonb
language sql
stable
security definer
set search_path to public
as $function$
  select public.dashboard_state() || jsonb_build_object(
    'leaderboard', (
      select jsonb_object_agg(window_name, datos) from (
        select window_name, jsonb_build_object(
                 'puesto', rank, 'participantes', participants,
                 'accuracy', round(accuracy::numeric, 2), 'cobertura', round(coverage::numeric, 3),
                 'mejor_accuracy', round(best_accuracy::numeric, 2),
                 'mejor_cobertura', round(best_coverage::numeric, 3),
                 'capturado', captured_at) as datos
        from (select distinct on (window_name) * from public.leaderboard_snapshots
              order by window_name, captured_at desc) ultimo
      ) t
    ),
    'historial_puesto', (
      select jsonb_agg(jsonb_build_object('cuando', captured_at, 'puesto', rank,
                                          'accuracy', round(accuracy::numeric, 2))
             order by captured_at)
      from (select * from public.leaderboard_snapshots
            where window_name = 'rolling_24h' order by captured_at desc limit 60) h
    )
  );
$function$;

revoke all on function public.dashboard_public() from public;
grant execute on function public.dashboard_public() to anon, authenticated, service_role;
