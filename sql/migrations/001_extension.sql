CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE TABLE IF NOT EXISTS schema_migrations (
  filename   text        PRIMARY KEY,
  applied_at timestamptz NOT NULL DEFAULT now()
);
