-- Follow-up to 2026-09-22-pseudonymization.sql: pseudonym_mappings.org_id
-- had no foreign key, so a typo'd or orphaned org_id could be inserted --
-- a mapping row that no legitimate org could ever look up (or, if org_id
-- values are ever recycled, one that a future different org would
-- incorrectly inherit). Run against the Supabase project after the
-- original migration.
--
-- Source: PR #3 review follow-up (branch worktree-pseudonymization).

alter table pseudonym_mappings
  add constraint pseudonym_mappings_org_id_fkey
  foreign key (org_id) references orgs (id) on delete cascade;
