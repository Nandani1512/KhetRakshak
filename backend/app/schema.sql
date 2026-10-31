-- KhetRakshak schema (Supabase Postgres). Applied idempotently at startup.
create table if not exists fields (
    id            text primary key,
    name          text not null,
    farmer_name   text,
    polygon       jsonb not null,
    centroid_lat  double precision not null,
    centroid_lon  double precision not null,
    area_ha       double precision not null,
    crop          text not null,
    sowing_date   date,
    language      text not null default 'hi-IN',
    state         text,
    district      text,
    transport_rate_per_qtl_km double precision,
    telegram_chat_id bigint,
    created_at    timestamptz not null default now()
);

create table if not exists runs (
    id           text primary key,
    field_id     text references fields(id) on delete cascade,
    mode         text not null,           -- monitor | ask
    as_of_date   date,
    question     text,
    status       text not null default 'running',
    critic_loops int  not null default 0,
    result       jsonb,
    started_at   timestamptz not null default now(),
    finished_at  timestamptz
);

create table if not exists run_events (
    id      bigserial primary key,
    run_id  text references runs(id) on delete cascade,
    seq     int not null,
    node    text not null,
    status  text not null,
    message text not null,
    data    jsonb,
    ts      timestamptz not null default now()
);
create index if not exists run_events_run on run_events(run_id, seq);

create table if not exists alerts (
    id           bigserial primary key,
    run_id       text references runs(id) on delete cascade,
    field_id     text references fields(id) on delete cascade,
    language     text not null,
    text_en      text not null,
    text_local   text,
    citations    jsonb,
    risk_types   text[],
    delivered    boolean not null default false,
    created_at   timestamptz not null default now()
);

create table if not exists api_cache (
    key        text primary key,
    payload    jsonb not null,
    fetched_at timestamptz not null default now()
);
