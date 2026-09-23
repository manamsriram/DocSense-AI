"""Pseudonymize sensitive document/query text before it reaches
third-party LLM providers (Groq, OpenRouter, Gemini). See
docs/superpowers/specs/2026-09-22-pseudonymization-design.md.

Presidio/spaCy/pytesseract are imported lazily inside the functions that
need them, not here at module level -- this module is imported by app.py,
which runs on both the 512MB main dyno and the GitHub Actions ingestion
runner sharing one requirements.txt. A module-level import would add
baseline RAM to the dyno even on requests that never touch this code.

`import app` is also deliberately NOT at module level -- `pseudonymize.py`
and `app.py` import each other (app.py imports pseudonymize at module
level for the six LLM call sites it wraps), which works fine under
`gunicorn app:app` (app.py is imported exactly once, and by the time it
reaches `import pseudonymize`, `sys.modules['app']` already holds the
partially-initialized module, so pseudonymize's own `import app` -- were
it at module level -- would just reuse that same object). But a top-level
`import app` here would double-initialize Flask/clients if app.py were
ever run a second time as `__main__` (e.g. `python app.py` re-imports
itself as a fresh 'app' module, since the running script is registered in
sys.modules as '__main__', not 'app'). `app` below is a lazy proxy: the
real `import app` only happens on first attribute access (e.g. the first
`app.supabase_admin` a request actually triggers), not at pseudonymize.py
import time -- and because it's a real module-level attribute (not
something conjured only inside a function body), it stays compatible with
this module's existing test mocking pattern,
`patch('pseudonymize.app.supabase_admin', ...)`, which resolves
`pseudonymize.app` via `getattr()` before any pseudonymize function runs.
"""
import hashlib
import hmac
import os
import re
import threading
import time


class _LazyApp:
    """Defers `import app` until the first real attribute access. See the
    module docstring above for why this isn't a plain top-level import."""
    _module = None

    def __getattr__(self, name):
        if _LazyApp._module is None:
            import app as _app_module
            _LazyApp._module = _app_module
        return getattr(_LazyApp._module, name)


app = _LazyApp()

DEFAULT_ENTITY_TYPES = [
    'PERSON', 'EMAIL_ADDRESS', 'PHONE_NUMBER', 'US_SSN', 'CREDIT_CARD',
    'IBAN_CODE', 'LOCATION',
]

# Presidio's default score for a "confident" detection varies by recognizer,
# but sub-0.5 detections are routinely spaCy PERSON/LOCATION false positives
# (table headers, "US", short common words) -- registering those globally
# substitutes them everywhere in the org's prompts and degrades answer
# quality. MIN_ENTITY_LENGTH catches the short-token case score_threshold
# alone doesn't (a 2-char match can still score high).
NER_SCORE_THRESHOLD = 0.6
MIN_ENTITY_LENGTH = 3

# Required so pseudonyms are unreversible without this secret and org-scoped
# (no cross-tenant linkage) -- see _make_pseudonym. Read the same way other
# required secrets are read in this codebase (app.py's `_required` list +
# EnvironmentError, MODEL_SERVICE_SECRET's os.getenv). No production
# Supabase rows exist yet for this feature, so it's safe to require this at
# import time without a migration path for already-hashed values.
PSEUDONYM_SECRET = os.environ['PSEUDONYM_SECRET'].encode()


def get_org_entity_types(org_id):
    res = app.supabase_admin.table('orgs').select('pseudonymize_entities').eq('id', org_id).execute()
    if res.data and res.data[0].get('pseudonymize_entities'):
        return res.data[0]['pseudonymize_entities']
    return DEFAULT_ENTITY_TYPES


def _make_pseudonym(entity_type, real_value, org_id):
    # HMAC-SHA256 keyed by PSEUDONYM_SECRET, over "{org_id}:{real_value}" --
    # unlike a plain hash, this can't be brute-forced offline by anyone who
    # knows the scheme (the scheme is public, in this repo), and the org_id
    # in the HMAC input means the same real_value hashes to a different
    # pseudonym per org (no cross-tenant linkage). Still deterministic per
    # (org_id, real_value), so the upsert race in get_or_create_pseudonym
    # stays benign -- two concurrent callers compute the same pseudonym.
    # 8 hex chars (32 bits) -- 4 chars (16 bits) collides ~50% of the time
    # (birthday paradox) once an org has ~300 distinct names of one entity
    # type, which could put the wrong real name in a deanonymized answer.
    digest = hmac.new(
        PSEUDONYM_SECRET, f'{org_id}:{real_value}'.encode(), hashlib.sha256
    ).hexdigest()[:8]
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

    pseudonym = _make_pseudonym(entity_type, real_value, org_id)
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

# Supabase caps unpaginated .select() responses at 1000 rows -- any org
# with more distinct pseudonym_mappings rows than this would otherwise
# silently stop protecting the overflow (sent raw to LLMs, no error).
_SUPABASE_PAGE_SIZE = 1000


def _compile_pattern(mapping):
    """Compile a case-insensitive regex pattern with word boundaries to
    prevent partial matches. Example: "John" will not match inside "Johnny"
    or "Johnson". Case-insensitive because an upstream LLM (the graph
    extraction prompt) is instructed to lowercase entity names, so a
    pseudonym token like "PERSON_a1b2c3d4" can come back as
    "person_a1b2c3d4" -- matching must still find it and substitute the
    canonical stored value, not the case variant that was actually matched.
    """
    if not mapping:
        return None
    # Build pattern with word boundaries: \b(?:...|...)\b
    # This prevents "John" from matching inside "Johnny"
    pattern_str = r'\b(?:' + '|'.join(re.escape(k) for k in sorted(mapping, key=len, reverse=True)) + r')\b'
    return re.compile(pattern_str, re.IGNORECASE)


def _fetch_all_mapping_rows(org_id):
    """Page through pseudonym_mappings for org_id via .range(), accumulating
    every row -- a single unpaginated .select() silently truncates at
    Supabase's default 1000-row cap (Task 1 critical fix)."""
    rows = []
    start = 0
    while True:
        res = (
            app.supabase_admin.table('pseudonym_mappings')
            .select('real_value,pseudonym')
            .eq('org_id', org_id)
            .range(start, start + _SUPABASE_PAGE_SIZE - 1)
            .execute()
        )
        page = res.data or []
        rows.extend(page)
        if len(page) < _SUPABASE_PAGE_SIZE:
            break
        start += _SUPABASE_PAGE_SIZE
    return rows


def _fetch_org_mapping(org_id):
    """Returns (mapping, pattern) fetched atomically from the same cache
    entry -- callers must not re-read the pattern via a second lock
    acquisition (that reintroduces the stale-pattern race this replaced)."""
    with _mapping_cache_lock:
        cached = _mapping_cache.get(org_id)
        if cached and time.monotonic() - cached[0] < _MAPPING_CACHE_TTL_S:
            return cached[1], cached[2]

    rows = _fetch_all_mapping_rows(org_id)
    mapping = {row['real_value']: row['pseudonym'] for row in rows}
    pattern = _compile_pattern(mapping)

    with _mapping_cache_lock:
        _mapping_cache[org_id] = (time.monotonic(), mapping, pattern)
    return mapping, pattern


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

    Returns (reverse_mapping, pattern) atomically, same contract as
    _fetch_org_mapping.
    """
    # Ensures the forward cache is fresh (refetches if its TTL expired);
    # idempotent no-op cost if it's still fresh.
    _fetch_org_mapping(org_id)
    with _mapping_cache_lock:
        forward_fetched_at, forward_mapping, _ = _mapping_cache[org_id]

    with _reverse_mapping_cache_lock:
        cached = _reverse_mapping_cache.get(org_id)
        if cached and cached[0] == forward_fetched_at:
            return cached[1], cached[2]

    reverse_mapping = {pseudonym: real for real, pseudonym in forward_mapping.items()}
    reverse_pattern = _compile_pattern(reverse_mapping)

    with _reverse_mapping_cache_lock:
        _reverse_mapping_cache[org_id] = (forward_fetched_at, reverse_mapping, reverse_pattern)
    return reverse_mapping, reverse_pattern


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

# Presidio's AnalyzerEngine/NlpEngineProvider default to en_core_web_lg
# (~400-560MB) when no nlp_engine is passed explicitly -- this project's
# Dockerfile and .github/workflows/ingest.yml only install en_core_web_sm,
# so the unconfigured default either downloads the wrong model at runtime
# or fails outright. Force en_core_web_sm explicitly.
_SPACY_NLP_CONFIGURATION = {
    'nlp_engine_name': 'spacy',
    'models': [{'lang_code': 'en', 'model_name': 'en_core_web_sm'}],
}


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
                from presidio_analyzer.nlp_engine import NlpEngineProvider
                nlp_engine = NlpEngineProvider(
                    nlp_configuration=_SPACY_NLP_CONFIGURATION
                ).create_engine()
                _analyzer = AnalyzerEngine(nlp_engine=nlp_engine)
    return _analyzer


def detect_and_register_entities(text, org_id):
    """Run NER once and register every detected entity's pseudonym.
    Ingestion-time only (or once per unique question string) -- this is
    the only function in this module that runs actual NER."""
    entity_types = get_org_entity_types(org_id)
    results = _get_analyzer().analyze(
        text=text, entities=entity_types, language='en',
        score_threshold=NER_SCORE_THRESHOLD,
    )
    for r in results:
        real_value = text[r.start:r.end]
        if len(real_value.strip()) < MIN_ENTITY_LENGTH:
            continue
        get_or_create_pseudonym(org_id, real_value, r.entity_type)


def _substitute(text, mapping, pattern):
    if not mapping or pattern is None:
        return text
    # mapping keys may not match the matched substring's case (pattern is
    # compiled case-insensitively -- see _compile_pattern) -- look up
    # case-insensitively but always substitute the canonical stored value.
    lower_mapping = {k.lower(): v for k, v in mapping.items()}
    return pattern.sub(lambda m: lower_mapping[m.group(0).lower()], text)


def pseudonymize_text(text, org_id):
    mapping, pattern = _fetch_org_mapping(org_id)
    return _substitute(text, mapping, pattern)


def deanonymize_text(text, org_id):
    reverse_mapping, pattern = _fetch_org_mapping_reverse(org_id)
    return _substitute(text, reverse_mapping, pattern)


_image_redactor = None


def _get_image_redactor():
    global _image_redactor
    if _image_redactor is None:
        from presidio_image_redactor import ImageRedactorEngine, ImageAnalyzerEngine
        # Share the same AnalyzerEngine instance (built via _get_analyzer(),
        # already configured for en_core_web_sm) instead of letting
        # ImageRedactorEngine build its own separate AnalyzerEngine, which
        # would default to en_core_web_lg the same way _get_analyzer() used
        # to before this fix.
        image_analyzer = ImageAnalyzerEngine(analyzer_engine=_get_analyzer())
        _image_redactor = ImageRedactorEngine(image_analyzer_engine=image_analyzer)
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
