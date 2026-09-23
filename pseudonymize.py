"""Pseudonymize sensitive document/query text before it reaches
third-party LLM providers (Groq, OpenRouter, Gemini). See
docs/superpowers/specs/2026-09-22-pseudonymization-design.md.

Presidio/spaCy/pytesseract are imported lazily inside the functions that
need them, not here at module level -- this module is imported by app.py,
which runs on both the 512MB main dyno and the GitHub Actions ingestion
runner sharing one requirements.txt. A module-level import would add
baseline RAM to the dyno even on requests that never touch this code.
"""
import hashlib
import os
import threading
import time

import app  # for app.supabase_admin -- same pattern as scripts/ingest_worker.py

DEFAULT_ENTITY_TYPES = [
    'PERSON', 'EMAIL_ADDRESS', 'PHONE_NUMBER', 'US_SSN', 'CREDIT_CARD',
    'IBAN_CODE', 'LOCATION',
]


def get_org_entity_types(org_id):
    res = app.supabase_admin.table('orgs').select('pseudonymize_entities').eq('id', org_id).execute()
    if res.data and res.data[0].get('pseudonymize_entities'):
        return res.data[0]['pseudonymize_entities']
    return DEFAULT_ENTITY_TYPES


def _make_pseudonym(entity_type, real_value):
    digest = hashlib.sha256(real_value.encode()).hexdigest()[:4]
    return f'{entity_type}_{digest}'


def get_or_create_pseudonym(org_id, real_value, entity_type):
    existing = (
        app.supabase_admin.table('pseudonym_mappings')
        .select('pseudonym')
        .eq('org_id', org_id)
        .eq('real_value', real_value)
        .execute()
    )
    if existing.data:
        return existing.data[0]['pseudonym']

    pseudonym = _make_pseudonym(entity_type, real_value)
    app.supabase_admin.table('pseudonym_mappings').upsert({
        'org_id': org_id,
        'pseudonym': pseudonym,
        'real_value': real_value,
        'entity_type': entity_type,
    }, on_conflict='org_id,real_value').execute()
    _invalidate_mapping_cache(org_id)
    return pseudonym


# TTL cache, same shape as app.py's existing _kb_version_cache pattern
# (app.py:1894-1895). Without this, pseudonymize_text/deanonymize_text
# would hit Supabase once per call -- and a single /ask request calls them
# ~10-30 times (once per candidate chunk, plus the question, plus each
# reformulation, plus the final answer). One Supabase round-trip per org
# per _MAPPING_CACHE_TTL_S window instead of per call.
_MAPPING_CACHE_TTL_S = 30
_mapping_cache = {}   # org_id -> (fetched_at, {real_value: pseudonym})
_mapping_cache_lock = threading.Lock()


def _fetch_org_mapping(org_id):
    with _mapping_cache_lock:
        cached = _mapping_cache.get(org_id)
        if cached and time.monotonic() - cached[0] < _MAPPING_CACHE_TTL_S:
            return cached[1]

    res = app.supabase_admin.table('pseudonym_mappings').select('real_value,pseudonym').eq('org_id', org_id).execute()
    mapping = {row['real_value']: row['pseudonym'] for row in res.data}

    with _mapping_cache_lock:
        _mapping_cache[org_id] = (time.monotonic(), mapping)
    return mapping


def _fetch_org_mapping_reverse(org_id):
    return {pseudonym: real for real, pseudonym in _fetch_org_mapping(org_id).items()}


def _invalidate_mapping_cache(org_id):
    """Called by get_or_create_pseudonym after registering a new entity, so
    a chunk indexed moments ago doesn't wait out the TTL before its
    pseudonym is usable in a query."""
    with _mapping_cache_lock:
        _mapping_cache.pop(org_id, None)
