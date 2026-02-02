create extension if not exists vector;
create extension if not exists pgcrypto;

create table if not exists labels (
  id uuid primary key default gen_random_uuid(),
  name text not null,
  kind text not null,
  category text not null default 'topic',
  embedding vector(256),
  usage_count integer not null default 0,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  metadata jsonb not null default '{}'::jsonb,
  unique (name, kind, category)
);

create table if not exists facts (
  id uuid primary key default gen_random_uuid(),
  label_id uuid not null references labels(id) on delete cascade,
  content text not null,
  category text not null,
  confidence real not null default 0.5,
  source_role text not null,
  embedding vector(256),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  last_seen_at timestamptz not null default now(),
  evidence_count integer not null default 1,
  metadata jsonb not null default '{}'::jsonb
);

create table if not exists patterns (
  id uuid primary key default gen_random_uuid(),
  regex text not null unique,
  category text not null,
  template text not null,
  weight real not null default 0.5,
  uses integer not null default 0,
  successes integer not null default 0,
  updated_at timestamptz not null default now(),
  metadata jsonb not null default '{}'::jsonb
);

create table if not exists learned_patterns (
  id uuid primary key default gen_random_uuid(),
  signature text not null unique,
  template text not null,
  category text not null,
  confidence real not null default 0.5,
  uses integer not null default 0,
  successes integer not null default 0,
  updated_at timestamptz not null default now(),
  metadata jsonb not null default '{}'::jsonb
);

create table if not exists prompts (
  id uuid primary key default gen_random_uuid(),
  role text not null,
  text text not null,
  timestamp timestamptz not null,
  metadata jsonb not null default '{}'::jsonb
);

create index if not exists idx_labels_name on labels (name);
create index if not exists idx_facts_label_id on facts (label_id);
create index if not exists idx_facts_content_fts on facts using gin (to_tsvector('english', content));

create index if not exists idx_labels_embedding on labels using ivfflat (embedding vector_l2_ops);
create index if not exists idx_facts_embedding on facts using ivfflat (embedding vector_l2_ops);
create index if not exists idx_learned_patterns_category on learned_patterns (category);
