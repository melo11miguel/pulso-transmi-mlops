-- Esquema operativo: ciclos, modelos, entregas, predicciones, métricas, drift y decisiones.
-- Principio: todo lo que el pipeline decide o envía deja evidencia consultable.

-- La demanda liberada por el stream trae su instante de publicación (mide el retraso del collector).
alter table public.observations add column released_at timestamptz;

-- Registro de todas las ejecuciones de los workflows (entrenar, predecir, monitorear...)
create table public.pipeline_runs (
  id            bigint generated always as identity primary key,
  job           text not null,
  started_at    timestamptz not null default now(),
  finished_at   timestamptz,
  status        text not null default 'running'
                check (status in ('running','success','error','skipped')),
  summary       jsonb not null default '{}'::jsonb,
  error         text,
  github_run_id text,
  git_commit    text
);
create index pipeline_runs_job_started_idx on public.pipeline_runs (job, started_at desc);

-- Ciclos de pronóstico vistos en la API (el contrato de la entrega)
create table public.forecast_cycles (
  cycle_id             text primary key,
  state                text not null,
  origin_at            timestamptz not null,
  data_cutoff          timestamptz not null,
  opens_at             timestamptz,
  closes_at            timestamptz not null,
  expected_predictions integer not null,
  targets              jsonb not null,
  first_seen_at        timestamptz not null default now(),
  outcome              text not null default 'seen'
                       check (outcome in ('seen','submitted','missed','error'))
);

-- Versiones de modelo: identidad, linaje, validación y estado
create table public.model_versions (
  version             text primary key,
  status              text not null default 'candidate'
                      check (status in ('candidate','champion','archived','rejected')),
  algorithm           text not null,
  trained_at          timestamptz not null,
  training_data_start timestamptz,
  training_data_end   timestamptz not null,
  git_commit          text,
  features            jsonb not null default '[]'::jsonb,
  params              jsonb not null default '{}'::jsonb,
  validation          jsonb not null default '{}'::jsonb,
  artifact_path       text,
  artifact_sha256     text,
  parent_version      text references public.model_versions(version),
  reason              text,
  created_at          timestamptz not null default now(),
  promoted_at         timestamptz
);
-- A lo sumo un champion a la vez
create unique index model_versions_single_champion
  on public.model_versions ((true)) where status = 'champion';

-- Entregas: un registro por intento, creado ANTES de enviar (deja evidencia aunque falle la red)
create table public.submissions (
  id              bigint generated always as identity primary key,
  cycle_id        text not null references public.forecast_cycles(cycle_id),
  client_run_id   text not null,
  idempotency_key text not null unique,
  model_version   text references public.model_versions(version),
  status          text not null default 'pending'
                  check (status in ('pending','accepted','rejected','error')),
  attempt         integer,
  submission_id   text,
  is_official     boolean,
  http_status     integer,
  request_id      text,
  payload_hash    text,
  receipt         jsonb,
  error           text,
  github_run_id   text,
  git_commit      text,
  created_at      timestamptz not null default now(),
  updated_at      timestamptz not null default now()
);
create index submissions_cycle_idx on public.submissions (cycle_id);

create table public.predictions (
  submission_row_id bigint not null references public.submissions(id) on delete cascade,
  cycle_id          text not null references public.forecast_cycles(cycle_id),
  station_id        text not null references public.stations(station_id),
  target_at         timestamptz not null,
  horizon_minutes   integer not null check (horizon_minutes > 0),
  value             double precision not null check (value >= 0),
  made_at           timestamptz not null default now(),
  primary key (submission_row_id, station_id, target_at)
);
create index predictions_target_idx on public.predictions (station_id, target_at);

-- Instantáneas de desempeño calculadas por el monitor
create table public.metric_snapshots (
  id                bigint generated always as identity primary key,
  computed_at       timestamptz not null default now(),
  window_name       text not null check (window_name in ('cumulative','rolling_24h')),
  model_version     text,
  window_start      timestamptz,
  window_end        timestamptz,
  accuracy          double precision,
  wape              double precision,
  coverage          double precision,
  n_targets         integer,
  n_resolved        integer,
  baseline_accuracy double precision,
  by_station        jsonb,
  by_horizon        jsonb
);
create index metric_snapshots_idx on public.metric_snapshots (window_name, computed_at desc);

create table public.drift_signals (
  id          bigint generated always as identity primary key,
  computed_at timestamptz not null default now(),
  kind        text not null check (kind in ('performance','data','operational')),
  name        text not null,
  station_id  text,
  value       double precision,
  threshold   double precision,
  triggered   boolean not null,
  details     jsonb not null default '{}'::jsonb
);
create index drift_signals_idx on public.drift_signals (kind, computed_at desc);

create table public.retrain_decisions (
  id                bigint generated always as identity primary key,
  decided_at        timestamptz not null default now(),
  decision          text not null check (decision in ('keep','investigate','retrain')),
  reason            text not null,
  evidence          jsonb not null default '{}'::jsonb,
  model_version     text references public.model_versions(version),
  new_model_version text references public.model_versions(version),
  github_run_id     text
);
create index retrain_decisions_idx on public.retrain_decisions (decided_at desc);

-- Predicción + valor real (cuando el collector ya lo tiene). Base de la evaluación propia.
create view public.prediction_results with (security_invoker = on) as
select s.id as submission_row_id, s.cycle_id, s.model_version, s.is_official, s.status,
       p.station_id, p.target_at, p.horizon_minutes, p.value as predicted, o.demand as actual
from public.predictions p
join public.submissions s on s.id = p.submission_row_id
left join public.observations o
       on o.station_id = p.station_id and o.observed_at = p.target_at;

-- ---------------------------------------------------------------------------------------------
-- Funciones (RPC). Solo las ejecuta la service role.
-- ---------------------------------------------------------------------------------------------

-- Upsert de un lote del stream y avance del cursor en UNA transacción:
-- si algo falla, ni los datos ni el cursor cambian.
create or replace function public.ingest_observations(
  p_rows            jsonb,
  p_cursor_resource text,
  p_cursor_after    text
) returns jsonb
language plpgsql
set search_path = public
as $$
declare
  v_inserted integer;
  v_updated  integer;
begin
  with src as (
    select * from jsonb_to_recordset(p_rows)
      as x(station_id text, observed_at timestamptz, demand integer, released_at timestamptz)
  ), up as (
    insert into public.observations as o (station_id, observed_at, demand, released_at)
    select station_id, observed_at, demand, released_at from src
    on conflict (station_id, observed_at) do update
      set demand = excluded.demand,
          released_at = coalesce(excluded.released_at, o.released_at)
    returning (xmax = 0) as inserted
  )
  select count(*) filter (where inserted), count(*) filter (where not inserted)
    into v_inserted, v_updated from up;

  insert into public.collector_state as c (resource, cursor, updated_at)
  values (p_cursor_resource, p_cursor_after, now())
  on conflict (resource) do update set cursor = excluded.cursor, updated_at = now();

  return jsonb_build_object('inserted', v_inserted, 'updated', v_updated);
end;
$$;

-- Promueve una versión a champion (o hace rollback a una anterior) de forma atómica.
create or replace function public.promote_model(p_version text, p_reason text default null)
returns jsonb
language plpgsql
set search_path = public
as $$
declare
  v_previous text;
begin
  if not exists (select 1 from public.model_versions where version = p_version) then
    raise exception 'model version % does not exist', p_version;
  end if;
  select version into v_previous from public.model_versions where status = 'champion';
  if v_previous is not distinct from p_version then
    return jsonb_build_object('promoted', p_version, 'previous', v_previous, 'changed', false);
  end if;
  update public.model_versions set status = 'archived' where status = 'champion';
  update public.model_versions
     set status = 'champion', promoted_at = now(), reason = coalesce(p_reason, reason)
   where version = p_version;
  return jsonb_build_object('promoted', p_version, 'previous', v_previous, 'changed', true);
end;
$$;

-- ---------------------------------------------------------------------------------------------
-- Seguridad: RLS activado sin políticas y sin permisos para anon/authenticated.
-- El pipeline usa la service role; el dashboard (bono) recibirá vistas de solo lectura aparte.
-- ---------------------------------------------------------------------------------------------
alter table public.pipeline_runs      enable row level security;
alter table public.forecast_cycles    enable row level security;
alter table public.model_versions     enable row level security;
alter table public.submissions        enable row level security;
alter table public.predictions        enable row level security;
alter table public.metric_snapshots   enable row level security;
alter table public.drift_signals      enable row level security;
alter table public.retrain_decisions  enable row level security;

revoke all on all tables    in schema public from anon, authenticated;
revoke all on all sequences in schema public from anon, authenticated;
revoke execute on function public.ingest_observations(jsonb, text, text) from public, anon, authenticated;
revoke execute on function public.promote_model(text, text) from public, anon, authenticated;
grant  execute on function public.ingest_observations(jsonb, text, text) to service_role;
grant  execute on function public.promote_model(text, text) to service_role;

-- Bodega privada de artefactos de modelos
insert into storage.buckets (id, name, public)
values ('models', 'models', false)
on conflict (id) do nothing;
