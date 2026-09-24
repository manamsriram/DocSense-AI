-- Pseudonymization feature: org-scoped PII pseudonym mappings +
-- per-org entity-type override. Run against the Supabase project before
-- deploying this feature (see pseudonymize.py / README.md).
--
-- Source: docs/superpowers/plans/2026-09-22-pseudonymization.md, Task 1.
--
-- NOTE: if this migration was already applied to prod before the RLS and
-- (org_id, pseudonym) uniqueness statements below were added, re-run just
-- those two statements against prod -- they were missing from the
-- original apply:
--   alter table pseudonym_mappings enable row level security;
--   alter table pseudonym_mappings add constraint pseudonym_mappings_org_pseudonym_unique unique (org_id, pseudonym);

-- New table
create table pseudonym_mappings (
  id uuid primary key default gen_random_uuid(),
  org_id uuid not null,
  pseudonym text not null,
  real_value text not null,
  entity_type text not null,
  created_at timestamptz not null default now(),
  unique (org_id, real_value),
  -- Catches a pseudonym collision (two different real values hashing to
  -- the same digest for the same org) as a DB error instead of silently
  -- letting one upsert overwrite the other's mapping row.
  unique (org_id, pseudonym)
);
create index on pseudonym_mappings (org_id);

-- This table holds raw PII (real_value). The app only ever accesses it via
-- the service-role client, which bypasses RLS -- but on Supabase's default
-- Data API grants, the anon/authenticated keys can otherwise query it
-- directly with no end-user policies defined. Enable RLS with no policies
-- so the Data API has no path to this table for those roles.
alter table pseudonym_mappings enable row level security;

-- New column on the existing orgs table
alter table orgs add column pseudonymize_entities jsonb;
