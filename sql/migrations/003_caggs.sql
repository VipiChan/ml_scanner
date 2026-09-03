-- NSE cash session buckets must start at 09:15 IST, not Unix/UTC epoch.
-- Origin is a Monday session open: 2000-01-03 09:15 Asia/Kolkata.

CREATE MATERIALIZED VIEW IF NOT EXISTS ohlcv_15m
WITH (timescaledb.continuous, timescaledb.materialized_only = true) AS
SELECT
  symbol,
  time_bucket(
    INTERVAL '15 minutes',
    ts,
    TIMESTAMPTZ '2000-01-03 09:15:00+05:30'
  ) AS ts,
  first(open, ts)         AS open,
  max(high)               AS high,
  min(low)                AS low,
  last(close, ts)         AS close,
  sum(volume)             AS volume,
  min(instrument_token)   AS instrument_token,
  count(*)                AS n_5m
FROM ohlcv_5m
GROUP BY 1, 2
WITH NO DATA;

CREATE MATERIALIZED VIEW IF NOT EXISTS ohlcv_60m
WITH (timescaledb.continuous, timescaledb.materialized_only = true) AS
SELECT
  symbol,
  time_bucket(
    INTERVAL '60 minutes',
    ts,
    TIMESTAMPTZ '2000-01-03 09:15:00+05:30'
  ) AS ts,
  first(open, ts)         AS open,
  max(high)               AS high,
  min(low)                AS low,
  last(close, ts)         AS close,
  sum(volume)             AS volume,
  min(instrument_token)   AS instrument_token,
  count(*)                AS n_5m
FROM ohlcv_5m
GROUP BY 1, 2
WITH NO DATA;

SELECT add_continuous_aggregate_policy(
  'ohlcv_15m',
  start_offset => NULL,
  end_offset => INTERVAL '10 minutes',
  schedule_interval => INTERVAL '1 minute',
  if_not_exists => TRUE
);

SELECT add_continuous_aggregate_policy(
  'ohlcv_60m',
  start_offset => NULL,
  end_offset => INTERVAL '10 minutes',
  schedule_interval => INTERVAL '1 minute',
  if_not_exists => TRUE
);
