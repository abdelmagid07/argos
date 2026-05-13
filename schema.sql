-- Argos schema for PostgreSQL 
CREATE TABLE IF NOT EXISTS raw_transactions (
    id                SERIAL PRIMARY KEY,
    transaction_id    TEXT UNIQUE,
    user_id           BIGINT,
    amount            DOUBLE PRECISION,
    merchant_id       BIGINT,
    merchant_category TEXT,
    device            TEXT,
    country           TEXT,
    timestamp         DOUBLE PRECISION,
    is_fraud          INTEGER,
    created_at        TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_raw_user      ON raw_transactions (user_id);
CREATE INDEX IF NOT EXISTS idx_raw_merchant  ON raw_transactions (merchant_id);
CREATE INDEX IF NOT EXISTS idx_raw_timestamp ON raw_transactions (timestamp);

-- Per-user aggregates. Written wholesale by `python -m src.features`.
CREATE TABLE IF NOT EXISTS user_features (
    user_id                  BIGINT PRIMARY KEY,
    total_transactions       INTEGER,
    total_spend              DOUBLE PRECISION,
    avg_transaction_amount   DOUBLE PRECISION,
    max_transaction_amount   DOUBLE PRECISION,
    transaction_velocity_24h INTEGER,
    unique_merchants         INTEGER,
    unique_countries         INTEGER,
    fraud_rate               DOUBLE PRECISION,
    last_transaction_ts      DOUBLE PRECISION,
    updated_at               DOUBLE PRECISION
);

-- Per-merchant aggregates. Same idea as user_features.
CREATE TABLE IF NOT EXISTS merchant_features (
    merchant_id                 BIGINT PRIMARY KEY,
    merchant_total_transactions INTEGER,
    merchant_avg_amount         DOUBLE PRECISION,
    merchant_fraud_rate         DOUBLE PRECISION,
    merchant_unique_users       INTEGER,
    updated_at                  DOUBLE PRECISION
);

-- Sanity check after running this file:
--   SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';
-- Should list: raw_transactions, user_features, merchant_features.
