# Deployment Guide

Platform: **railway**
Framework: **fastapi**
Production URL: http://localhost:8000

## Pre-deploy checklist
1. Run readiness — aim for 80+ score
2. Switch to **live** Stripe keys in vault
3. Set DATABASE_URL and apply schema
4. Set production URL in project settings

## Environment variables
```
NODE_ENV=production
APP_URL=http://localhost:8000
STRIPE_SECRET_KEY=sk_live_
STRIPE_PUBLISHABLE_KEY=pk_live_
STRIPE_WEBHOOK_SECRET=whsec_
DATABASE_URL=postgresql://user:pass@host:5432/db?sslmode=require
```

## Deploy
```bash
railway up
```

## Post-deploy
1. Verify SSL: http://localhost:8000
2. Test health: http://localhost:8000/health
3. Register production Stripe webhook
4. Schedule backups: scripts/backup-db.sh
