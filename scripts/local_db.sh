#!/usr/bin/env bash
set -euo pipefail

docker compose up -d db
for i in $(seq 1 30); do
  if docker compose exec -T db pg_isready -U hippo -d hippo > /dev/null 2>&1; then
    break
  fi
  sleep 1
done

docker compose exec -T db psql -U hippo -d hippo -f /app/memory/schema.sql
echo "local db ready"
