# Database backup script (Windows)
param([string]$BackupDir = ".\backups")

if (-not $env:DATABASE_URL) { Write-Error "DATABASE_URL not set"; exit 1 }
New-Item -ItemType Directory -Force -Path $BackupDir | Out-Null
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$file = Join-Path $BackupDir "backup-$timestamp.sql"

pg_dump $env:DATABASE_URL -f $file
Write-Host "Backup: $file"

Get-ChildItem $BackupDir -Filter "backup-*.sql" |
  Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-7) } |
  Remove-Item -Force
