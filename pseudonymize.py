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
import logging
import os
import re
import threading
import time
from typing import NamedTuple, Optional, Pattern


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

# NOTE: Presidio's SpacyRecognizer assigns a FIXED 0.85 score to every
# spaCy NER hit (PERSON, LOCATION, ...) regardless of the model's actual
# confidence -- https://github.com/microsoft/presidio/issues/1372. That
# means this threshold does NOT discriminate between confident and
# marginal spaCy detections; it only filters the (mostly non-spaCy)
# recognizers that report varying scores. spaCy PERSON/LOCATION false
# positives (table headers, "US", capitalized common words like "May"/
# "Will"/"Bill" from en_core_web_sm) pass through regardless of this
# threshold. MIN_ENTITY_LENGTH below only catches the short-token case.
NER_SCORE_THRESHOLD = 0.6
MIN_ENTITY_LENGTH = 3

# Required so pseudonyms are unreversible without this secret and org-scoped
# (no cross-tenant linkage) -- see _make_pseudonym. Read the same way other
# required secrets are read in this codebase (app.py's `_required` list +
# EnvironmentError, MODEL_SERVICE_SECRET's os.getenv). No production
# Supabase rows exist yet for this feature, so it's safe to require this at
# import time without a migration path for already-hashed values.
PSEUDONYM_SECRET = os.environ['PSEUDONYM_SECRET'].encode()


# TTL cache so a multi-chunk ingestion pass (one call per chunk from
# detect_and_register_entities) doesn't hit Supabase once per chunk for a
# value that essentially never changes mid-ingestion.
_ENTITY_TYPES_CACHE_TTL_S = 60
_entity_types_cache = {}  # org_id -> (fetched_at, entity_types)
_entity_types_cache_lock = threading.Lock()


def get_org_entity_types(org_id):
    with _entity_types_cache_lock:
        cached = _entity_types_cache.get(org_id)
        if cached and time.monotonic() - cached[0] < _ENTITY_TYPES_CACHE_TTL_S:
            return cached[1]

    res = app.supabase_admin.table('orgs').select('pseudonymize_entities').eq('id', org_id).execute()
    raw = res.data[0].get('pseudonymize_entities') if res.data else None
    # pseudonymize_entities is unconstrained jsonb -- validate its shape here
    # (list of entity-type strings) rather than handing a malformed value
    # straight to Presidio's entities= param, where it fails opaquely mid-NER
    # instead of with a clear error naming the bad config.
    if raw and isinstance(raw, list) and all(isinstance(v, str) for v in raw):
        entity_types = raw
    elif raw:
        logging.error(
            f"[pseudonymize] org {org_id} has malformed pseudonymize_entities "
            f"({raw!r}, expected a list of strings) -- falling back to DEFAULT_ENTITY_TYPES"
        )
        entity_types = DEFAULT_ENTITY_TYPES
    else:
        entity_types = DEFAULT_ENTITY_TYPES

    with _entity_types_cache_lock:
        _entity_types_cache[org_id] = (time.monotonic(), entity_types)
    return entity_types


def _make_pseudonym(entity_type, real_value, org_id):
    # HMAC-SHA256 keyed by PSEUDONYM_SECRET, over "{org_id}:{real_value}" --
    # unlike a plain hash, this can't be brute-forced offline by anyone who
    # knows the scheme (the scheme is public, in this repo), and the org_id
    # in the HMAC input means the same real_value hashes to a different
    # pseudonym per org (no cross-tenant linkage). Still deterministic per
    # (org_id, real_value), so the upsert race in get_or_create_pseudonym
    # stays benign -- two concurrent callers compute the same pseudonym.
    # 16 hex chars (64 bits) -- the DB has no uniqueness constraint on
    # (org_id, pseudonym) (only on (org_id, real_value)), so a collision
    # between two different real values would silently let one upsert
    # overwrite the other's mapping row and deanonymize to the wrong
    # value. 64 bits pushes the birthday-bound 50%-collision point to
    # ~5 billion distinct values per org.
    digest = hmac.new(
        PSEUDONYM_SECRET, f'{org_id}:{real_value}'.encode(), hashlib.sha256
    ).hexdigest()[:16]
    return f'{entity_type}_{digest}'


def get_or_create_pseudonym(org_id, real_value, entity_type):
    # Cheap in-memory check first -- avoids a per-entity Supabase SELECT for
    # a real_value that's already registered (the common case for a name
    # mentioned across many chunks of one document). Falls through to the
    # SELECT below only on an actual cache miss.
    cached_mapping, _ = _fetch_org_mapping_with_retry(org_id)
    if real_value in cached_mapping:
        return cached_mapping[real_value]

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
_mapping_cache = {}   # org_id -> _MappingCacheEntry
_mapping_cache_lock = threading.Lock()
# org_id -> int, bumped by _invalidate_mapping_cache. Lets an in-flight
# refresh detect that it raced a registration and was invalidated mid-fetch,
# so it doesn't republish a snapshot missing the newly-registered entity.
_mapping_generation = {}

# Reverse mapping cache with its own compiled pattern.
_reverse_mapping_cache = {}  # org_id -> _ReverseMappingCacheEntry
_reverse_mapping_cache_lock = threading.Lock()

# Supabase caps unpaginated .select() responses at 1000 rows -- any org
# with more distinct pseudonym_mappings rows than this would otherwise
# silently stop protecting the overflow (sent raw to LLMs, no error).
_SUPABASE_PAGE_SIZE = 1000


class _MappingCacheEntry(NamedTuple):
    """Named fields instead of a bare tuple so "these travel together" is a
    type fact, not just a comment -- a future edit that reorders fields in
    one place but not another fails immediately instead of silently reading
    a pattern as a mapping."""
    fetched_at: float
    mapping: dict
    pattern: Optional[Pattern]
    redis_generation: int


class _ReverseMappingCacheEntry(NamedTuple):
    forward_fetched_at: float
    mapping: dict
    pattern: Optional[Pattern]


def _redis_mapping_generation(org_id):
    """Cross-process cache-invalidation marker. get_or_create_pseudonym runs
    in whichever process is ingesting (the GitHub Actions worker in
    production, this process in local dev) and _invalidate_mapping_cache
    only clears THAT process's in-memory cache -- the web dyno's cache is
    untouched and would otherwise keep serving a mapping missing the
    newly-registered entity for up to _MAPPING_CACHE_TTL_S. Checking this
    (one cheap Redis GET) on every _fetch_org_mapping call closes that
    window instead of only refreshing on TTL expiry. Falls back to TTL-only
    staleness (always returns 0) if Redis isn't configured.
    """
    redis_client = getattr(app, 'redis_client', None)
    if not redis_client:
        return 0
    try:
        raw = redis_client.get(f'pseudomapgen:{org_id}')
        return int(raw) if raw is not None else 0
    except Exception:
        return 0


def _bump_redis_mapping_generation(org_id):
    redis_client = getattr(app, 'redis_client', None)
    if not redis_client:
        return
    try:
        redis_client.incr(f'pseudomapgen:{org_id}')
    except Exception:
        pass


def _compile_pattern(mapping, case_insensitive):
    """Compile a regex pattern with word boundaries to prevent partial
    matches. Example: "John" will not match inside "Johnny" or "Johnson".

    case_insensitive=True is for the REVERSE (pseudonym -> real) pattern
    only: an upstream LLM (the graph extraction prompt) is instructed to
    lowercase entity names, so a pseudonym token like "PERSON_a1b2c3d4" can
    come back as "person_a1b2c3d4" and matching must still find it.

    The FORWARD (real -> pseudonym) pattern is case-sensitive: real_value's
    case comes straight from NER on the original text, and matching it
    case-insensitively would also substitute any lowercase word that only
    coincidentally shares a spelling with a registered value -- e.g. a
    spaCy PERSON false positive on "May"/"Will"/"Bill" would then also
    pseudonymize every lowercase "may"/"will"/"bill" in unrelated prompts.
    """
    if not mapping:
        return None
    # Anchor on adjacent word characters rather than \b: a real_value like
    # a phone number ("+1 (555) 123-4567") starts/ends with punctuation, so
    # \b (a transition between \w and \W) never matches there and the value
    # would never be substituted. (?<!\w)/(?!\w) still block "John" from
    # matching inside "Johnny", but also match at a leading/trailing
    # non-word character.
    alternation = '|'.join(re.escape(k) for k in sorted(mapping, key=len, reverse=True))
    pattern_str = r'(?<!\w)(?:' + alternation + r')(?!\w)'
    return re.compile(pattern_str, re.IGNORECASE if case_insensitive else 0)


def _fetch_all_mapping_rows(org_id):
    """Page through pseudonym_mappings for org_id via .range(), accumulating
    every row -- a single unpaginated .select() silently truncates at
    Supabase's default 1000-row cap (Task 1 critical fix). .order('id') is
    required for .range() to page consistently: LIMIT/OFFSET over an
    unordered scan can return the same row on two pages or skip it
    entirely under concurrent writes, and a skipped row is PII that's
    never pseudonymized."""
    rows = []
    start = 0
    while True:
        res = (
            app.supabase_admin.table('pseudonym_mappings')
            .select('real_value,pseudonym')
            .eq('org_id', org_id)
            .order('id')
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
    redis_generation = _redis_mapping_generation(org_id)
    with _mapping_cache_lock:
        cached = _mapping_cache.get(org_id)
        if (
            cached
            and time.monotonic() - cached.fetched_at < _MAPPING_CACHE_TTL_S
            and cached.redis_generation == redis_generation
        ):
            return cached.mapping, cached.pattern
        generation = _mapping_generation.get(org_id, 0)

    rows = _fetch_all_mapping_rows(org_id)
    mapping = {row['real_value']: row['pseudonym'] for row in rows}
    pattern = _compile_pattern(mapping, case_insensitive=False)

    with _mapping_cache_lock:
        # If a registration invalidated the cache (bumped the generation)
        # while this fetch was in flight, this snapshot may already be
        # stale -- don't publish it over the invalidation. Retry instead
        # of returning it directly, so the cache always ends up holding
        # a snapshot at least as fresh as the last invalidation.
        if _mapping_generation.get(org_id, 0) != generation:
            return None
        _mapping_cache[org_id] = _MappingCacheEntry(time.monotonic(), mapping, pattern, redis_generation)
    return mapping, pattern


# _fetch_org_mapping returns None only on the generation-race window (a
# registration invalidated the cache while a fetch was in flight) -- that
# window is normally microseconds, but under sustained concurrent
# get_or_create_pseudonym calls (bulk ingestion registering many entities
# in parallel) every pseudonymize_text/deanonymize_text caller could spin
# tightly with no forward-progress guarantee. A tiny sleep bounds CPU use
# on the memory-constrained dyno; the cap turns "stuck forever" into a
# loud, debuggable RuntimeError instead of an invisible hang the caller's
# own timeout eventually papers over.
_MAPPING_FETCH_RETRY_SLEEP_S = 0.01
_MAPPING_FETCH_MAX_RETRIES = 200


def _fetch_org_mapping_with_retry(org_id):
    for _ in range(_MAPPING_FETCH_MAX_RETRIES):
        result = _fetch_org_mapping(org_id)
        if result is not None:
            return result
        time.sleep(_MAPPING_FETCH_RETRY_SLEEP_S)
    raise RuntimeError(f"_fetch_org_mapping_with_retry: gave up after {_MAPPING_FETCH_MAX_RETRIES} retries for org {org_id} (sustained cache invalidation race)")


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
    while True:
        # Ensures the forward cache is fresh (refetches if its TTL expired
        # or a cross-process invalidation landed); idempotent no-op cost if
        # it's still fresh.
        _fetch_org_mapping_with_retry(org_id)
        with _mapping_cache_lock:
            forward_entry = _mapping_cache.get(org_id)
        if forward_entry is not None:
            break
        # A concurrent _invalidate_mapping_cache popped the entry between
        # the unlocked fetch above and this lock acquisition -- retry the
        # whole fetch instead of KeyError'ing on a missing cache entry.
    forward_fetched_at, forward_mapping = forward_entry.fetched_at, forward_entry.mapping

    with _reverse_mapping_cache_lock:
        cached = _reverse_mapping_cache.get(org_id)
        if cached and cached.forward_fetched_at == forward_fetched_at:
            return cached.mapping, cached.pattern

    reverse_mapping = {pseudonym: real for real, pseudonym in forward_mapping.items()}
    reverse_pattern = _compile_pattern(reverse_mapping, case_insensitive=True)

    with _reverse_mapping_cache_lock:
        _reverse_mapping_cache[org_id] = _ReverseMappingCacheEntry(forward_fetched_at, reverse_mapping, reverse_pattern)
    return reverse_mapping, reverse_pattern


def _invalidate_mapping_cache(org_id):
    """Called by get_or_create_pseudonym after registering a new entity, so
    a chunk indexed moments ago doesn't wait out the TTL before its
    pseudonym is usable in a query. Also bumps the Redis-backed generation
    marker so a DIFFERENT process's cache (e.g. the web dyno, when this
    runs in the ingestion worker) picks up the invalidation too."""
    with _mapping_cache_lock:
        _mapping_cache.pop(org_id, None)
        _mapping_generation[org_id] = _mapping_generation.get(org_id, 0) + 1
    with _reverse_mapping_cache_lock:
        _reverse_mapping_cache.pop(org_id, None)
    _bump_redis_mapping_generation(org_id)


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
    """Run full NER (including PERSON/LOCATION, which need spaCy) and
    register every detected entity's pseudonym. Ingestion-time only -- this
    loads en_core_web_sm, which only the ingestion worker (GH Actions, 7GB
    RAM) or local dev should pay for, not the main 512MB dyno. For the
    query path (a question or conversation-history turn, which was never
    ingested and so has no existing mapping row) see
    detect_and_register_query_entities below instead."""
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


# Structured-PII patterns for query-time detection -- deliberately
# regex-only (no spaCy/NlpEngine) so this can run on every question without
# loading en_core_web_sm on the main dyno. A question or conversation-
# history turn was never ingested, so pseudonymize_text alone would leave
# any PII typed directly into it unchanged (it only substitutes values
# already in pseudonym_mappings). PERSON/LOCATION are NOT covered here --
# those require the full NER pass in detect_and_register_entities, which
# is deliberately not run on the query path for the RAM reason above; a
# name/place typed directly into a question (not already present in an
# indexed document) is a known, accepted gap.
_QUERY_TIME_PATTERNS = {
    'EMAIL_ADDRESS': re.compile(r'[\w.+-]+@[\w-]+\.[\w.-]+'),
    'US_SSN': re.compile(r'(?<!\w)\d{3}-\d{2}-\d{4}(?!\w)'),
    'PHONE_NUMBER': re.compile(r'(?<!\w)(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}(?!\w)'),
    'CREDIT_CARD': re.compile(r'(?<!\w)(?:\d[ -]?){13,16}(?!\w)'),
}


def detect_and_register_query_entities(text, org_id):
    """Cheap regex-only counterpart to detect_and_register_entities for the
    query path (a question, or a conversation-history turn the first time
    it's the current question) -- see module comment on _QUERY_TIME_PATTERNS
    for why this doesn't run full NER."""
    entity_types = get_org_entity_types(org_id)
    for entity_type, pattern in _QUERY_TIME_PATTERNS.items():
        if entity_type not in entity_types:
            continue
        for m in pattern.finditer(text):
            real_value = m.group(0)
            if len(real_value.strip()) < MIN_ENTITY_LENGTH:
                continue
            get_or_create_pseudonym(org_id, real_value, entity_type)


def _substitute(text, mapping, pattern):
    if not mapping or pattern is None:
        return text
    if pattern.flags & re.IGNORECASE:
        # Reverse pattern only (see _compile_pattern) -- mapping keys may
        # not match the matched substring's case; look up case-insensitively
        # but always substitute the canonical stored value.
        lower_mapping = {k.lower(): v for k, v in mapping.items()}
        return pattern.sub(lambda m: lower_mapping[m.group(0).lower()], text)
    return pattern.sub(lambda m: mapping[m.group(0)], text)


def pseudonymize_text(text, org_id):
    mapping, pattern = _fetch_org_mapping_with_retry(org_id)
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
