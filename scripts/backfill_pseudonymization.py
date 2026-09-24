"""One-off backfill: register pseudonym mappings for every chunk indexed
BEFORE the pseudonymization feature existed.

Registration only happens inside index_pdf's flush() for newly-ingested
chunks (see app.py). Every chunk already in Qdrant when this feature was
deployed has no pseudonym_mappings row, so at query time
pseudonymize_text leaves its PII unchanged and it reaches the LLM raw.
This script scrolls every org's existing corpus and runs the same
detect_and_register_entities NER pass ingestion would have run, so old
documents get the same protection as newly-uploaded ones.

Run once per environment after deploying the pseudonymization migration,
on a machine with enough RAM for en_core_web_sm (NOT the 512MB Render
dyno — run it the same way ingest_worker.py runs, e.g. via the GitHub
Actions runner or a one-off local/CI job with the full requirements.txt
installed).

Usage: python scripts/backfill_pseudonymization.py
"""
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app  # noqa: E402  (env must be loaded before import; app.py does that itself)
import pseudonymize  # noqa: E402

logging.basicConfig(level=logging.INFO)


def _iter_org_ids():
    org_ids = set()
    offset = None
    while True:
        batch, offset = app.qdrant.scroll(
            collection_name=app.COLLECTION, with_payload=['org_id'], limit=1000, offset=offset
        )
        org_ids.update(r.payload.get('org_id') for r in batch if r.payload.get('org_id'))
        if offset is None:
            break
    return org_ids


def _backfill_org(org_id):
    registered = 0
    failed = 0
    offset = None
    while True:
        batch, offset = app.qdrant.scroll(
            collection_name=app.COLLECTION,
            scroll_filter=app.Filter(must=[
                app.FieldCondition(key='org_id', match=app.MatchValue(value=org_id))
            ]),
            with_payload=True,
            limit=1000,
            offset=offset,
        )
        for r in batch:
            text = r.payload.get('text', '')
            if not text:
                continue
            try:
                pseudonymize.detect_and_register_entities(text, org_id)
                registered += 1
            except Exception as e:
                failed += 1
                logging.error(f"[backfill] entity registration failed for org {org_id}, chunk {r.id}: {e}")
        if offset is None:
            break
    logging.info(f"[backfill] org {org_id}: {registered} chunks processed, {failed} failed")
    return registered, failed


def main():
    org_ids = _iter_org_ids()
    logging.info(f"[backfill] {len(org_ids)} orgs with existing chunks")
    total_registered = total_failed = 0
    for org_id in org_ids:
        registered, failed = _backfill_org(org_id)
        total_registered += registered
        total_failed += failed
    logging.info(f"[backfill] done: {total_registered} chunks processed, {total_failed} failed across {len(org_ids)} orgs")
    if total_failed:
        sys.exit(1)


if __name__ == '__main__':
    main()
