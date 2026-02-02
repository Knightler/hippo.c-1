#!/usr/bin/env bash
set -euo pipefail

docker compose up -d db
docker compose exec -T db psql -U hippo -d hippo -f /app/memory/schema.sql
echo "local db ready"
