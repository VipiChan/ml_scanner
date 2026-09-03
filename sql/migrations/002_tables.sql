-- Intraday source of truth: Kite 5minute + completed WebSocket bars.
CREATE TABLE IF NOT EXISTS ohlcv_5m (
  ts                timestamptz NOT NULL,
  symbol            text        NOT NULL,
  instrument_token  integer     NOT NULL,
  open              double precision NOT NULL,
  high              double precision NOT NULL,
  low               double precision NOT NULL,
  close             double precision NOT NULL,
  volume            double precision NOT NULL,
  source            text        NOT NULL,
  PRIMARY KEY (symbol, ts),
  CONSTRAINT ohlcv_5m_source_chk
    CHECK (source IN ('historical', 'websocket', 'reconcile'))
);

SELECT create_hypertable(
  'ohlcv_5m',
  by_range('ts', INTERVAL '7 days'),
  if_not_exists => TRUE
);

SELECT add_dimension(
  'ohlcv_5m',
  by_hash('symbol', 8),
  if_not_exists => TRUE
);

ALTER TABLE ohlcv_5m SET (
  timescaledb.compress,
  timescaledb.compress_segmentby = 'symbol',
  timescaledb.compress_orderby = 'ts DESC'
);

SELECT add_compression_policy(
  'ohlcv_5m',
  INTERVAL '14 days',
  if_not_exists => TRUE
);

-- Native daily (adjusted). Not a continuous aggregate of 5m.
CREATE TABLE IF NOT EXISTS ohlcv_1d (
  ts                timestamptz NOT NULL,
  symbol            text        NOT NULL,
  instrument_token  integer     NOT NULL,
  open              double precision NOT NULL,
  high              double precision NOT NULL,
  low               double precision NOT NULL,
  close             double precision NOT NULL,
  volume            double precision NOT NULL,
  source            text        NOT NULL DEFAULT 'kite_day',
  PRIMARY KEY (symbol, ts)
);

SELECT create_hypertable(
  'ohlcv_1d',
  by_range('ts', INTERVAL '1 year'),
  if_not_exists => TRUE
);

CREATE TABLE IF NOT EXISTS ingest_watermark (
  symbol     text        NOT NULL,
  interval   text        NOT NULL,
  last_ts    timestamptz NOT NULL,
  updated_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (symbol, interval),
  CONSTRAINT ingest_watermark_interval_chk
    CHECK (interval IN ('5minute', 'day'))
);

CREATE TABLE IF NOT EXISTS ingest_gaps (
  symbol      text        NOT NULL,
  interval    text        NOT NULL,
  ts_missing  timestamptz NOT NULL,
  detected_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (symbol, interval, ts_missing)
);

CREATE TABLE IF NOT EXISTS ca_flags (
  symbol       text        NOT NULL,
  session_date date        NOT NULL,
  kite_close   double precision NOT NULL,
  rollup_close double precision,
  abs_rel_diff double precision,
  flagged      boolean     NOT NULL,
  PRIMARY KEY (symbol, session_date)
);
