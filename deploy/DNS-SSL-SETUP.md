# Domain & SSL Setup

Production URL: http://localhost:8000
Domain: localhost
Framework: fastapi

## SSL
SSL/TLS is automatic on Vercel, Railway, and Fly.io custom domains.

## Stripe Webhook (production)
Update webhook URL to: `http://localhost:8000/api/stripe/webhook`

## Verification
```bash
curl http://localhost:8000/health
```
Run readiness from Stripe Installer after deploy.
