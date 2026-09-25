-- Aplicada el 2026-09-18 al proyecto tjqrgdzvhfxfovmwgzgv (pulso-transmi).
-- Memoria de datos: catálogo, demanda, contexto y estado del collector.

create table public.stations (
  station_id   text primary key,
  station_name text not null,
  corridor     text,
  latitude     double precision,
  longitude    double precision
);

create table public.observations (
  observed_at timestamptz not null,
  station_id  text not null references public.stations(station_id),
  demand      integer not null check (demand >= 0),
  primary key (station_id, observed_at)
);
create index observations_observed_at_idx on public.observations (observed_at);

create table public.context (
  observed_at          timestamptz primary key,
  rain_mm              double precision,
  rain_forecast        double precision,
  temperature_c        double precision,
  temperature_forecast double precision,
  event_intensity      double precision
);

-- Bitácora de ejecuciones del collector / cargas
create table public.ingestion_runs (
  id            bigint generated always as identity primary key,
  source        text not null,
  started_at    timestamptz not null default now(),
  finished_at   timestamptz,
  status        text not null default 'running' check (status in ('running','success','error')),
  rows_received integer,
  rows_upserted integer,
  cursor_before text,
  cursor_after  text,
  error         text
);

-- Cursor confirmado por recurso (solo avanza tras confirmar la transacción)
create table public.collector_state (
  resource   text primary key,
  cursor     text,
  updated_at timestamptz not null default now()
);

-- Estas tablas solo se escriben con la service role (GitHub Actions); RLS activado sin políticas.
alter table public.stations        enable row level security;
alter table public.observations    enable row level security;
alter table public.context         enable row level security;
alter table public.ingestion_runs  enable row level security;
alter table public.collector_state enable row level security;
