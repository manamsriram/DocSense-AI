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
import re
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
    # 8 hex chars (32 bits) -- 4 chars (16 bits) collides ~50% of the time
    # (birthday paradox) once an org has ~300 distinct names of one entity
    # type, which could put the wrong real name in a deanonymized answer.
    digest = hashlib.sha256(real_value.encode()).hexdigest()[:8]
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
_mapping_cache = {}   # org_id -> (fetched_at, {real_value: pseudonym}, compiled_pattern)
_mapping_cache_lock = threading.Lock()

# Reverse mapping cache with its own compiled pattern. Atomic tuple ensures
# pattern and mapping always come from the same fetch — prevents stale pattern
# matching against fresh (or vice versa) when pseudonymize/deanonymize are
# called at different cadences within a request.
_reverse_mapping_cache = {}  # org_id -> (fetched_at, {pseudonym: real_value}, compiled_reverse_pattern)
_reverse_mapping_cache_lock = threading.Lock()


def _compile_pattern(mapping):
    """Compile regex pattern with word boundaries to prevent partial matches.
    Example: "John" will not match inside "Johnny" or "Johnson"."""
    if not mapping:
        return None
    # Build pattern with word boundaries: \b(?:...|...)\b
    # This prevents "John" from matching inside "Johnny"
    pattern_str = r'\b(?:' + '|'.join(re.escape(k) for k in sorted(mapping, key=len, reverse=True)) + r')\b'
    return re.compile(pattern_str)


def _fetch_org_mapping(org_id):
    with _mapping_cache_lock:
        cached = _mapping_cache.get(org_id)
        if cached and time.monotonic() - cached[0] < _MAPPING_CACHE_TTL_S:
            return cached[1]

    res = app.supabase_admin.table('pseudonym_mappings').select('real_value,pseudonym').eq('org_id', org_id).execute()
    mapping = {row['real_value']: row['pseudonym'] for row in res.data}
    pattern = _compile_pattern(mapping)

    with _mapping_cache_lock:
        _mapping_cache[org_id] = (time.monotonic(), mapping, pattern)
    return mapping


def _fetch_org_mapping_reverse(org_id):
    """Reverse-mapping cache tracks the SAME fetch as the forward cache --
    it stores the forward cache's fetched_at (not its own independent
    fetch time), and rebuilds only when that fetched_at has moved on.
    This is deliberate: a reverse cache with its own independent TTL clock
    can serve a stale snapshot while the forward cache has already picked
    up a newly-registered entity mid-request, which can leave a literal
    pseudonym token in a deanonymized answer (or swap the wrong entity).
    Tying the two together means forward and reverse can never disagree
    within the same TTL window.
    """
    # Ensures the forward cache is fresh (refetches if its TTL expired);
    # idempotent no-op cost if it's still fresh.
    _fetch_org_mapping(org_id)
    with _mapping_cache_lock:
        forward_fetched_at, forward_mapping, _ = _mapping_cache[org_id]

    with _reverse_mapping_cache_lock:
        cached = _reverse_mapping_cache.get(org_id)
        if cached and cached[0] == forward_fetched_at:
            return cached[1]

    reverse_mapping = {pseudonym: real for real, pseudonym in forward_mapping.items()}
    reverse_pattern = _compile_pattern(reverse_mapping)

    with _reverse_mapping_cache_lock:
        _reverse_mapping_cache[org_id] = (forward_fetched_at, reverse_mapping, reverse_pattern)
    return reverse_mapping


def _invalidate_mapping_cache(org_id):
    """Called by get_or_create_pseudonym after registering a new entity, so
    a chunk indexed moments ago doesn't wait out the TTL before its
    pseudonym is usable in a query."""
    with _mapping_cache_lock:
        _mapping_cache.pop(org_id, None)
    with _reverse_mapping_cache_lock:
        _reverse_mapping_cache.pop(org_id, None)


_analyzer = None
_analyzer_lock = threading.Lock()


def _get_analyzer():
    """Lazy singleton -- see module docstring on why this isn't a
    top-level import. Uses double-checked locking to prevent concurrent
    threads from each constructing an AnalyzerEngine (which loads a spaCy
    model), avoiding transient memory spikes on memory-constrained dyno."""
    global _analyzer
    if _analyzer is None:
        with _analyzer_lock:
            if _analyzer is None:
                from presidio_analyzer import AnalyzerEngine
                _analyzer = AnalyzerEngine()
    return _analyzer


def detect_and_register_entities(text, org_id):
    """Run NER once and register every detected entity's pseudonym.
    Ingestion-time only (or once per unique question string) -- this is
    the only function in this module that runs actual NER."""
    entity_types = get_org_entity_types(org_id)
    results = _get_analyzer().analyze(text=text, entities=entity_types, language='en')
    for r in results:
        real_value = text[r.start:r.end]
        get_or_create_pseudonym(org_id, real_value, r.entity_type)


def _substitute(text, mapping, pattern):
    if not mapping:
        return text
    if pattern is None:
        # Fallback (shouldn't happen with new code, but keeps _substitute robust)
        pattern = _compile_pattern(mapping)
    return pattern.sub(lambda m: mapping[m.group(0)], text)


def pseudonymize_text(text, org_id):
    mapping = _fetch_org_mapping(org_id)
    # Pattern is cached in _mapping_cache, retrieve it
    with _mapping_cache_lock:
        cached = _mapping_cache.get(org_id)
        pattern = cached[2] if cached else None
    return _substitute(text, mapping, pattern)


def deanonymize_text(text, org_id):
    reverse_mapping = _fetch_org_mapping_reverse(org_id)
    # Pattern is cached atomically with reverse_mapping in _reverse_mapping_cache
    with _reverse_mapping_cache_lock:
        cached = _reverse_mapping_cache.get(org_id)
        pattern = cached[2] if cached else None
    return _substitute(text, reverse_mapping, pattern)


_image_redactor = None


def _get_image_redactor():
    global _image_redactor
    if _image_redactor is None:
        from presidio_image_redactor import ImageRedactorEngine
        _image_redactor = ImageRedactorEngine()
    return _image_redactor


def _png_bytes_to_pil_image(png_bytes):
    from io import BytesIO
    from PIL import Image
    return Image.open(BytesIO(png_bytes))


def _pil_image_to_png_bytes(image):
    from io import BytesIO
    buf = BytesIO()
    image.save(buf, format='PNG')
    return buf.getvalue()


def redact_image(png_bytes, org_id):
    """Box out detected entities in a page/figure image before it's sent
    to a vision LLM. Ingestion-time only (GH Actions runner) -- never
    called on the query path."""
    image = _png_bytes_to_pil_image(png_bytes)
    entity_types = get_org_entity_types(org_id)
    redacted = _get_image_redactor().redact(image, entities=entity_types)
    return _pil_image_to_png_bytes(redacted)
