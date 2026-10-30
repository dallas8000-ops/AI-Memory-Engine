-- Stripe Installer — PostgreSQL schema
-- Tracks customers and subscriptions synced from Stripe webhooks

CREATE TABLE IF NOT EXISTS users (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  email         TEXT UNIQUE NOT NULL,
  name          TEXT,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS stripe_customers (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id         UUID REFERENCES users(id) ON DELETE CASCADE,
  auth_user_id    TEXT,
  stripe_customer_id TEXT UNIQUE NOT NULL,
  email           TEXT,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_stripe_customers_auth_user
  ON stripe_customers(auth_user_id) WHERE auth_user_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS subscriptions (
  id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id               UUID REFERENCES users(id) ON DELETE SET NULL,
  stripe_subscription_id TEXT UNIQUE NOT NULL,
  stripe_customer_id    TEXT NOT NULL,
  stripe_price_id       TEXT,
  status                TEXT NOT NULL,
  tier                  TEXT,
  current_period_end    TIMESTAMPTZ,
  cancel_at_period_end  BOOLEAN DEFAULT FALSE,
  created_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at            TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS webhook_events (
  id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  stripe_event_id TEXT UNIQUE NOT NULL,
  type        TEXT NOT NULL,
  processed   BOOLEAN DEFAULT FALSE,
  payload     JSONB,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_subscriptions_customer ON subscriptions(stripe_customer_id);
CREATE INDEX IF NOT EXISTS idx_subscriptions_status ON subscriptions(status);
CREATE INDEX IF NOT EXISTS idx_webhook_events_type ON webhook_events(type);

CREATE TABLE IF NOT EXISTS stripe_connect_accounts (
  id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  auth_user_id        TEXT,
  stripe_account_id   TEXT UNIQUE NOT NULL,
  charges_enabled     BOOLEAN DEFAULT FALSE,
  payouts_enabled     BOOLEAN DEFAULT FALSE,
  details_submitted   BOOLEAN DEFAULT FALSE,
  created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS stripe_transfers (
  id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  stripe_transfer_id  TEXT UNIQUE NOT NULL,
  stripe_account_id   TEXT NOT NULL,
  amount              INTEGER NOT NULL,
  currency            TEXT NOT NULL DEFAULT 'usd',
  status              TEXT,
  created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_connect_accounts_auth ON stripe_connect_accounts(auth_user_id);
CREATE INDEX IF NOT EXISTS idx_transfers_account ON stripe_transfers(stripe_account_id);
