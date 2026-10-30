#!/usr/bin/env bash
# Database backup — run via cron: 0 2 * * * ./scripts/backup-db.sh
set -euo pipefail

if [ -z "${DATABASE_URL:-}" ]; then
  echo "DATABASE_URL not set"
  exit 1
fi

BACKUP_DIR="${BACKUP_DIR:-./backups}"
mkdir -p "$BACKUP_DIR"
FILE="$BACKUP_DIR/backup-$(date +%Y%m%d-%H%M%S).sql"

pg_dump "$DATABASE_URL" > "$FILE"
gzip "$FILE"
echo "Backup: $FILE.gz"

find "$BACKUP_DIR" -name "backup-*.sql.gz" -mtime +7 -delete
