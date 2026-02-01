# Memory DB Setup (Supabase)

1) Open Supabase project → SQL Editor.
2) Create a new query.
3) Paste and run `memory/schema.sql`.
4) Set `SUPABASE_DATABASE_URL` in your `.env`.

Format:
```
postgresql://postgres:<PASSWORD>@db.<project-ref>.supabase.co:5432/postgres
```

Notes:
- Keep the password private.
- `pgvector` is required and enabled by the schema.
