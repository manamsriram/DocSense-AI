-- Pseudonymization feature: org-scoped PII pseudonym mappings +
-- per-org entity-type override. Run against the Supabase project before
-- deploying this feature (see pseudonymize.py / README.md).
--
-- Source: docs/superpowers/plans/2026-09-22-pseudonymization.md, Task 1.

-- New table
create table pseudonym_mappings (
  id uuid primary key default gen_random_uuid(),
  org_id uuid not null,
  pseudonym text not null,
  real_value text not null,
  entity_type text not null,
  created_at timestamptz not null default now(),
  unique (org_id, real_value)
);
create index on pseudonym_mappings (org_id);

-- New column on the existing orgs table
alter table orgs add column pseudonymize_entities jsonb;
