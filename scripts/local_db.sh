#!/usr/bin/env bash
set -euo pipefail

docker compose up -d db
ready=0
for i in $(seq 1 60); do
  if docker compose exec -T db pg_isready -h 127.0.0.1 -U hippo -d hippo > /dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 1
done

if [ "$ready" -ne 1 ]; then
  echo "local db not ready"
  exit 1
fi

docker compose exec -T db psql -h 127.0.0.1 -U hippo -d hippo -f /app/memory/schema.sql
echo "local db ready"
