# Neon PostgreSQL Setup
1. Create project at https://neon.tech
2. Copy pooled connection string
3. Store DATABASE_URL in Stripe Installer vault
4. Apply schema via Database panel or: psql $DATABASE_URL -f db/schema.sql
