# Pseudonymization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent raw sensitive text (PII per an org-configurable entity
list) from reaching third-party LLM providers (Groq, OpenRouter, Gemini)
in any call DocSense-AI makes, while leaving retrieval quality, citations,
and the final answer shown to the user completely unaffected.

**Architecture:** Presidio NER runs once per chunk at ingestion time, only
to populate an org-scoped `pseudonym_mappings` table (real_value ↔
pseudonym). A pure string-substitution helper (`pseudonymize_text`/
`deanonymize_text`, no NER) wraps the six places raw text crosses into or
out of a third-party LLM call. The retrieval pipeline (`hybrid_search`,
`find_relevant_chunks(_with_graph)`, `_promote_table_chunk_for_aggregation`,
the reranker) is untouched — it keeps operating on raw text end-to-end.

**Tech Stack:** `presidio-analyzer` + `presidio-anonymizer` (text NER +
substitution engine), `presidio-image-redactor` + `pytesseract` (OCR-based
image redaction for the vision fallback), a spaCy model for Presidio's NER
backend, Supabase (new table + one new column), Redis (mapping cache,
mirrors the existing `cache_response` pattern).

**Spec:** `docs/superpowers/specs/2026-09-22-pseudonymization-design.md`
(read the **Revision** section at the top first — it corrects the
original design after tracing the real call sites; the rest of the spec
should be read as already corrected).

## Global Constraints

- Zero added latency on the `/ask` request path: no NER at query time for
  chunk text (only a one-time, per-request NER pass over the user's short
  question string, which is negligible).
- Retrieval pipeline (`hybrid_search`, `find_relevant_chunks(_with_graph)`,
  `_promote_table_chunk_for_aggregation`, reranking) must not change —
  it keeps consuming raw `text` exactly as today.
- `presidio-analyzer`/`presidio-anonymizer`/spaCy/`presidio-image-redactor`/
  `pytesseract` must be **lazily imported** (inside the functions that use
  them, not at module top-level) — `requirements.txt` is shared between
  the 512MB main Render dyno and the GitHub Actions ingestion runner
  (`.github/workflows/ingest.yml:18`), and a module-level import would add
  baseline RAM to the dyno that already OOM'd once under normal load (see
  `model_service/service.py`'s embed/rerank split comment).
- Citations and the final answer shown to the user must always contain
  real values, never pseudonyms — only the LLM-bound copy is substituted.
- After code changes ship, re-run `python evals/run_eval.py` against a
  live deploy per CLAUDE.md's regression gates: `retrieval_recall`,
  `citation_quality`, and `groundedness` must not regress (they're
  untouched by this change, but bugs in the wiring could still degrade
  them — e.g. an over-eager entity match corrupting `prompt_text`).

---

## File Structure

- **Create `pseudonymize.py`** (new, top-level, alongside `app.py`) — the
  entire pseudonymization subsystem: Presidio setup (lazy-imported),
  Supabase mapping CRUD, the substitution helpers, and image redaction.
  Kept separate from `app.py` (already large) so this feature has one
  clear file to read, and so its heavy imports are visibly scoped to one
  module.
- **Create `tests/test_pseudonymize.py`** — unit tests for everything in
  `pseudonymize.py`, using a fake/mocked Supabase table (matching how
  `tests/test_app.py` already mocks `supabase_admin`).
- **Modify `app.py`** — six call sites (see spec's "Query-time: wrapping
  the actual call sites" section), plus `index_pdf`'s `flush()` and
  `describe_image_with_groq`'s ingestion-time caller.
- **Modify `requirements.txt`** — add the five new packages.
- **Modify `Dockerfile`** — add `tesseract-ocr` to the runtime stage's
  `apt-get install`.
- **Modify `.github/workflows/ingest.yml`** — add a `tesseract-ocr` apt
  install step (the `ubuntu-latest` runner doesn't guarantee it).
- **Modify `tests/test_app.py`** — update tests touching `build_source`
  (new `org_id` parameter) and the wrapped call sites.

---

### Task 1: Data model — Supabase table, org config column, mapping CRUD

**Files:**
- Create: `pseudonymize.py`
- Test: `tests/test_pseudonymize.py`

**Interfaces:**
- Consumes: `app.supabase_admin` (already exists, same client used
  throughout `app.py` for `orgs`/`org_members`/`graph_nodes` — import it
  from `app` the same way `scripts/ingest_worker.py` does: `import app`
  then `app.supabase_admin`).
- Produces: `get_org_entity_types(org_id) -> list[str]`,
  `get_or_create_pseudonym(org_id, real_value, entity_type) -> str`,
  `_fetch_org_mapping(org_id) -> dict[str, str]` (real_value → pseudonym),
  `_fetch_org_mapping_reverse(org_id) -> dict[str, str]` (pseudonym →
  real_value). These four are used by every later task.

Before writing code, the Supabase side needs (documented here since no SQL
lives in this repo — same convention as `orgs`/`org_members`, created via
dashboard/migration outside this repo, matching how the existing org
tables were set up):

```sql
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
```

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_pseudonymize.py
import pytest
from unittest.mock import MagicMock, patch
import pseudonymize

DEFAULT_ENTITY_TYPES = [
    'PERSON', 'EMAIL_ADDRESS', 'PHONE_NUMBER', 'US_SSN', 'CREDIT_CARD',
    'IBAN_CODE', 'LOCATION',
]


@pytest.fixture(autouse=True)
def _clear_mapping_cache():
    """_fetch_org_mapping's TTL cache is module-level state (Task 1). Most
    tests below reuse org_id='org-1', so a mapping cached by one test
    would otherwise leak into the next and produce order-dependent
    failures."""
    pseudonymize._mapping_cache.clear()
    yield
    pseudonymize._mapping_cache.clear()


def _mock_supabase_table(rows_by_table):
    """Returns a MagicMock standing in for app.supabase_admin, where
    .table(name).select/insert/upsert(...).execute() chains return
    canned data from rows_by_table[name]."""
    mock = MagicMock()

    def table_side_effect(name):
        tbl = MagicMock()
        result = MagicMock()
        result.data = rows_by_table.get(name, [])
        tbl.select.return_value.eq.return_value.execute.return_value = result
        tbl.select.return_value.eq.return_value.eq.return_value.execute.return_value = result
        tbl.insert.return_value.execute.return_value = result
        tbl.upsert.return_value.execute.return_value = result
        return tbl

    mock.table.side_effect = table_side_effect
    return mock


def test_get_org_entity_types_returns_default_when_org_column_is_null():
    fake_supabase = _mock_supabase_table({'orgs': [{'pseudonymize_entities': None}]})
    with patch('pseudonymize.app.supabase_admin', fake_supabase):
        assert pseudonymize.get_org_entity_types('org-1') == DEFAULT_ENTITY_TYPES


def test_get_org_entity_types_returns_org_override():
    fake_supabase = _mock_supabase_table(
        {'orgs': [{'pseudonymize_entities': ['PERSON', 'US_SSN']}]}
    )
    with patch('pseudonymize.app.supabase_admin', fake_supabase):
        assert pseudonymize.get_org_entity_types('org-1') == ['PERSON', 'US_SSN']


def test_get_or_create_pseudonym_creates_new_mapping_when_absent():
    fake_supabase = _mock_supabase_table({'pseudonym_mappings': []})
    with patch('pseudonymize.app.supabase_admin', fake_supabase):
        pseudonym = pseudonymize.get_or_create_pseudonym('org-1', 'Jane Doe', 'PERSON')
    assert pseudonym.startswith('PERSON_')
    fake_supabase.table.return_value.upsert.assert_called_once()


def test_get_or_create_pseudonym_reuses_existing_mapping():
    fake_supabase = _mock_supabase_table(
        {'pseudonym_mappings': [{'pseudonym': 'PERSON_ab12', 'real_value': 'Jane Doe'}]}
    )
    with patch('pseudonymize.app.supabase_admin', fake_supabase):
        pseudonym = pseudonize_call = pseudonymize.get_or_create_pseudonym(
            'org-1', 'Jane Doe', 'PERSON'
        )
    assert pseudonym == 'PERSON_ab12'
    fake_supabase.table.return_value.upsert.assert_not_called()


def test_fetch_org_mapping_returns_real_value_to_pseudonym_dict():
    fake_supabase = _mock_supabase_table(
        {'pseudonym_mappings': [
            {'pseudonym': 'PERSON_ab12', 'real_value': 'Jane Doe'},
            {'pseudonym': 'EMAIL_ADDRESS_cd34', 'real_value': 'jane@example.com'},
        ]}
    )
    with patch('pseudonymize.app.supabase_admin', fake_supabase):
        mapping = pseudonymize._fetch_org_mapping('org-1')
    assert mapping == {'Jane Doe': 'PERSON_ab12', 'jane@example.com': 'EMAIL_ADDRESS_cd34'}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_pseudonymize.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pseudonymize'`

- [ ] **Step 3: Write minimal implementation**

```python
# pseudonymize.py
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


import threading
import time

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_pseudonymize.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add pseudonymize.py tests/test_pseudonymize.py
git commit -m "feat(pseudonymize): mapping CRUD against Supabase pseudonym_mappings"
```

**Manual step (not code, do before Task 4 ships to prod):** create the
`pseudonym_mappings` table and the `orgs.pseudonymize_entities` column in
the Supabase dashboard using the SQL above.

---

### Task 2: Ingestion-time NER — populate the mapping

**Files:**
- Modify: `pseudonymize.py`
- Test: `tests/test_pseudonymize.py`

**Interfaces:**
- Consumes: `get_org_entity_types`, `get_or_create_pseudonym` (Task 1).
- Produces: `detect_and_register_entities(text, org_id) -> None`. Used by
  Task 4 (ingestion) and Task 5 (graph extraction).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pseudonymize.py (append)
def test_detect_and_register_entities_registers_each_detected_entity():
    fake_supabase = _mock_supabase_table({
        'orgs': [{'pseudonymize_entities': ['PERSON']}],
        'pseudonym_mappings': [],
    })
    fake_analyzer_result = [MagicMock(entity_type='PERSON', start=0, end=8)]
    with patch('pseudonymize.app.supabase_admin', fake_supabase), \
         patch('pseudonymize._get_analyzer') as mock_get_analyzer:
        mock_get_analyzer.return_value.analyze.return_value = fake_analyzer_result
        pseudonymize.detect_and_register_entities('Jane Doe filed the report.', 'org-1')
    fake_supabase.table.return_value.upsert.assert_called_once()
    upserted = fake_supabase.table.return_value.upsert.call_args[0][0]
    assert upserted['real_value'] == 'Jane Doe'
    assert upserted['entity_type'] == 'PERSON'
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pseudonymize.py::test_detect_and_register_entities_registers_each_detected_entity -v`
Expected: FAIL with `AttributeError: module 'pseudonymize' has no attribute 'detect_and_register_entities'`

- [ ] **Step 3: Write minimal implementation**

```python
# pseudonymize.py (append)
_analyzer = None


def _get_analyzer():
    """Lazy singleton -- see module docstring on why this isn't a
    top-level import."""
    global _analyzer
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_pseudonymize.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add pseudonymize.py tests/test_pseudonymize.py
git commit -m "feat(pseudonymize): ingestion-time NER populates the mapping table"
```

---

### Task 3: Substitution helpers — pseudonymize_text / deanonymize_text

**Files:**
- Modify: `pseudonymize.py`
- Test: `tests/test_pseudonymize.py`

**Interfaces:**
- Consumes: `_fetch_org_mapping`, `_fetch_org_mapping_reverse` (Task 1).
- Produces: `pseudonymize_text(text, org_id) -> str`,
  `deanonymize_text(text, org_id) -> str`. Used by every `app.py` call
  site in Tasks 5-7.

This is the function every LLM call site wraps with — no NER, pure
string replace against the mapping already built by Task 2. Longest
real values are replaced first so "John Smith" matches before a lone
"John" inside it.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_pseudonymize.py (append)
def test_pseudonymize_text_replaces_known_real_values():
    fake_supabase = _mock_supabase_table({'pseudonym_mappings': [
        {'real_value': 'Jane Doe', 'pseudonym': 'PERSON_ab12'},
    ]})
    with patch('pseudonymize.app.supabase_admin', fake_supabase):
        result = pseudonymize.pseudonymize_text('Jane Doe signed the report.', 'org-1')
    assert result == 'PERSON_ab12 signed the report.'


def test_pseudonymize_text_prefers_longest_match():
    fake_supabase = _mock_supabase_table({'pseudonym_mappings': [
        {'real_value': 'John', 'pseudonym': 'PERSON_aaaa'},
        {'real_value': 'John Smith', 'pseudonym': 'PERSON_bbbb'},
    ]})
    with patch('pseudonymize.app.supabase_admin', fake_supabase):
        result = pseudonymize.pseudonymize_text('John Smith and John both signed.', 'org-1')
    assert result == 'PERSON_bbbb and PERSON_aaaa both signed.'


def test_pseudonymize_text_leaves_unknown_text_unchanged():
    fake_supabase = _mock_supabase_table({'pseudonym_mappings': []})
    with patch('pseudonymize.app.supabase_admin', fake_supabase):
        result = pseudonymize.pseudonymize_text('Nothing sensitive here.', 'org-1')
    assert result == 'Nothing sensitive here.'


def test_deanonymize_text_reverses_pseudonymize_text():
    fake_supabase = _mock_supabase_table({'pseudonym_mappings': [
        {'real_value': 'Jane Doe', 'pseudonym': 'PERSON_ab12'},
    ]})
    with patch('pseudonymize.app.supabase_admin', fake_supabase):
        pseudo = pseudonymize.pseudonymize_text('Jane Doe signed.', 'org-1')
        original = pseudonymize.deanonymize_text(pseudo, 'org-1')
    assert original == 'Jane Doe signed.'
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_pseudonymize.py -k "pseudonymize_text or deanonymize_text" -v`
Expected: FAIL with `AttributeError`

- [ ] **Step 3: Write minimal implementation**

```python
# pseudonymize.py (append)
import re


def _substitute(text, mapping):
    if not mapping:
        return text
    # Longest keys first so "John Smith" wins over a bare "John" inside it.
    pattern = re.compile('|'.join(re.escape(k) for k in sorted(mapping, key=len, reverse=True)))
    return pattern.sub(lambda m: mapping[m.group(0)], text)


def pseudonymize_text(text, org_id):
    return _substitute(text, _fetch_org_mapping(org_id))


def deanonymize_text(text, org_id):
    return _substitute(text, _fetch_org_mapping_reverse(org_id))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_pseudonymize.py -v`
Expected: PASS (10 tests)

- [ ] **Step 5: Commit**

```bash
git add pseudonymize.py tests/test_pseudonymize.py
git commit -m "feat(pseudonymize): pure substitution helpers for LLM call boundaries"
```

**Caught in self-review:** a single `/ask` request calls
`pseudonymize_text`/`deanonymize_text` ~10-30 times (once per candidate
chunk, plus the question, plus each reformulation, plus the final
answer). Without caching, that's 10-30 Supabase round-trips per request —
directly against the "zero added latency" goal. Task 1's
`_fetch_org_mapping` already has a 30s TTL cache for exactly this reason
(see Task 1, `_mapping_cache`) — one Supabase fetch per org per cache
window, not per call. Nothing further needed here.

---

### Task 4: Wire ingestion — `index_pdf`'s `flush()`

**Files:**
- Modify: `app.py:739-765` (the `flush()` closure inside `index_pdf`)
- Test: `tests/test_app.py`

**Interfaces:**
- Consumes: `pseudonymize.detect_and_register_entities(text, org_id)`
  (Task 2).
- Produces: nothing new — this task has no downstream consumers, it just
  ensures every indexed chunk's entities are registered before any query
  ever needs them.

- [ ] **Step 1: Write the failing test**

Find the existing ingestion test that exercises `flush()` (search
`tests/test_app.py` for `index_pdf` — there's already coverage of the
upsert path per `test_incremental_bm25_update_adds_new_document_chunks`
and neighbors). Add:

```python
# tests/test_app.py (append near existing index_pdf tests)
def test_index_pdf_registers_entities_for_each_flushed_chunk(tmp_path, monkeypatch):
    # Reuse this file's existing fixture pattern for a minimal one-page PDF
    # and mocked qdrant/embedding_model (see test_incremental_bm25_update_*
    # for the fixture shape already in this file).
    calls = []
    monkeypatch.setattr(
        'pseudonymize.detect_and_register_entities',
        lambda text, org_id: calls.append((text, org_id)),
    )
    # ... build/point at a minimal PDF fixture, call app.index_pdf(...) ...
    assert len(calls) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_app.py::test_index_pdf_registers_entities_for_each_flushed_chunk -v`
Expected: FAIL (no call to `detect_and_register_entities` yet)

- [ ] **Step 3: Modify `flush()`**

```python
# app.py — inside index_pdf's flush(), after building embed_texts/points,
# before or alongside the existing qdrant.upsert call:
import pseudonymize  # add to app.py's imports, near the top

def flush():
    nonlocal points, embed_texts, total_indexed
    if not points:
        return
    for _, _, display_text, _ in points:
        pseudonymize.detect_and_register_entities(display_text, org_id)
    vecs = list(get_embedding_model().embed(embed_texts))
    qdrant.upsert(
        collection_name=COLLECTION,
        points=[
            PointStruct(
                id=point_id,
                vector=vec.tolist(),
                payload={'source': filename, 'page': page_num, 'text': display_text,
                         'user_id': user_id, 'org_id': org_id, **extra}
            )
            for (point_id, page_num, display_text, extra), vec in zip(points, vecs)
        ]
    )
    # ... rest unchanged
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_app.py -k index_pdf -v`
Expected: PASS, and no pre-existing `index_pdf` test regresses (the
`qdrant.upsert` payload shape is unchanged — this only adds a call
before it).

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_app.py
git commit -m "feat: register chunk entities with pseudonymize at ingestion time"
```

---

### Task 5: Wire `extract_and_store_graph` (ingestion-time OpenRouter call)

**Files:**
- Modify: `app.py:1018-1039` (`extract_and_store_graph`)
- Test: `tests/test_app.py` (existing graph-extraction tests, search
  `test_build_graph_from_supabase*`/`extract_and_store_graph*`)

**Interfaces:**
- Consumes: `pseudonymize.pseudonymize_text(text, org_id)` (Task 3).
- Produces: nothing new.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_app.py (append near existing graph-extraction tests)
def test_extract_and_store_graph_sends_pseudonymized_text_to_openrouter(monkeypatch):
    captured = {}

    def fake_openrouter_helper(user_content, max_tokens):
        captured['prompt'] = user_content
        return '[]'  # no entities, simplest valid response

    monkeypatch.setattr('app._call_openrouter_helper', fake_openrouter_helper)
    monkeypatch.setattr('pseudonymize.pseudonymize_text', lambda text, org_id: 'REDACTED')
    monkeypatch.setattr('app.get_or_create_org_for_user', lambda user_id: 'org-1')

    app.extract_and_store_graph([('id1', 1, 'Jane Doe filed this.')], 'user-1', 'doc.pdf')

    assert 'REDACTED' in captured['prompt']
    assert 'Jane Doe' not in captured['prompt']
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_app.py::test_extract_and_store_graph_sends_pseudonymized_text_to_openrouter -v`
Expected: FAIL (`Jane Doe` still present — no pseudonymization applied yet)

- [ ] **Step 3: Modify `extract_and_store_graph`**

```python
# app.py:1018-1039, modify the formatted = ... line:
def extract_and_store_graph(batch_chunks, user_id, source_doc):
    org_id = get_or_create_org_for_user(user_id)
    formatted = '\n\n'.join(
        f'[{i}] {pseudonymize.pseudonymize_text(text, org_id)[:600]}'
        for i, (_, _, text) in enumerate(batch_chunks)
    )
    # ... rest unchanged (org_id was already computed here before formatted;
    # move the existing `org_id = get_or_create_org_for_user(user_id)` line,
    # if present later in the function, up to before this loop instead of
    # duplicating it)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_app.py -k graph -v`
Expected: PASS, including all pre-existing graph tests (they mock
`_call_openrouter_helper`'s response, not its input, so this change is
invisible to them unless they assert on `formatted`'s content — check for
that and update if found).

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_app.py
git commit -m "feat: pseudonymize chunk text before graph-extraction OpenRouter call"
```

---

### Task 6: Wire the `/ask` request path — decompose, grade, reformulate, synthesize

**Files:**
- Modify: `app.py:1750-1770` (`build_source`)
- Modify: `app.py:1773-1795` (`ask_file`)
- Modify: `app.py:1798-1880` (`ask_file_agentic`)
- Test: `tests/test_app.py` (existing `test_ask_file_agentic_*` tests)

**Interfaces:**
- Consumes: `pseudonymize.pseudonymize_text`, `pseudonymize.deanonymize_text`
  (Task 3).
- Produces: `build_source(score, text, org_id)` — signature change, one
  new required parameter. Any other caller of `build_source` in the
  codebase must be found and updated (`grep -n "build_source(" app.py`
  before starting this task to catch all call sites, not just the two
  shown in the spec).

This is the task with the most call sites; do it as one task since
`ask_file`/`ask_file_agentic`/`build_source` all change together and none
of them is independently testable mid-change (an org_id threaded halfway
through would break the ones not yet updated).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_app.py (append)
def test_build_source_keeps_citation_raw_but_pseudonymizes_prompt_text(monkeypatch):
    monkeypatch.setattr(
        'pseudonymize.pseudonymize_text',
        lambda text, org_id: text.replace('Jane Doe', 'PERSON_ab12'),
    )
    source, prompt_text = app.build_source(0.9, '[page 1, doc.pdf] Jane Doe signed.', 'org-1')
    assert source['text'] == 'Jane Doe signed.'  # citation stays raw
    assert prompt_text == 'PERSON_ab12 signed.'   # LLM-bound copy is pseudonymized


def test_ask_file_agentic_sends_pseudonymized_prompt_to_generate_text(monkeypatch):
    # Reuse this file's existing ask_file_agentic fixture pattern (see
    # test_ask_file_agentic_stops_iterating_once_relevant_chunks_found for
    # the mocked decompose_query/find_relevant_chunks_with_graph/grade_chunks
    # shape already established in this file).
    captured = {}

    def fake_generate_text(prompt, conversation_history=None):
        captured['prompt'] = prompt
        return 'The answer.'

    monkeypatch.setattr('app.generate_text', fake_generate_text)
    monkeypatch.setattr('app.decompose_query', lambda q: [q])
    monkeypatch.setattr(
        'app.find_relevant_chunks_with_graph',
        lambda *a, **kw: [(0.9, '[page 1, doc.pdf] Jane Doe signed.')],
    )
    monkeypatch.setattr('app.grade_chunks', lambda q, texts: (texts, []))
    monkeypatch.setattr('app.get_or_create_org_for_user', lambda user_id: 'org-1')
    monkeypatch.setattr(
        'pseudonymize.pseudonymize_text',
        lambda text, org_id: text.replace('Jane Doe', 'PERSON_ab12'),
    )
    monkeypatch.setattr('pseudonymize.deanonymize_text', lambda text, org_id: text)

    app.ask_file_agentic('Who signed?', 'user-1')

    assert 'PERSON_ab12' in captured['prompt']
    assert 'Jane Doe' not in captured['prompt']


def test_ask_file_agentic_deanonymizes_final_answer(monkeypatch):
    monkeypatch.setattr('app.generate_text', lambda prompt, conversation_history=None: 'Signed by PERSON_ab12.')
    monkeypatch.setattr('app.decompose_query', lambda q: [q])
    monkeypatch.setattr(
        'app.find_relevant_chunks_with_graph',
        lambda *a, **kw: [(0.9, '[page 1, doc.pdf] some text')],
    )
    monkeypatch.setattr('app.grade_chunks', lambda q, texts: (texts, []))
    monkeypatch.setattr('app.get_or_create_org_for_user', lambda user_id: 'org-1')
    monkeypatch.setattr('pseudonymize.pseudonymize_text', lambda text, org_id: text)
    monkeypatch.setattr(
        'pseudonymize.deanonymize_text',
        lambda text, org_id: text.replace('PERSON_ab12', 'Jane Doe'),
    )

    response, _ = app.ask_file_agentic('Who signed?', 'user-1')

    assert response == 'Signed by Jane Doe.'
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_app.py -k "build_source or pseudonymized_prompt or deanonymizes_final_answer" -v`
Expected: FAIL — `build_source` doesn't accept `org_id` yet, and neither
wrapping happens yet.

- [ ] **Step 3: Modify `build_source`**

```python
# app.py:1750-1770
def build_source(score, text, org_id):
    """Parse a display chunk into a source dict. Returns (source, prompt_text).

    source['text'] is always the raw value (citations show real data).
    prompt_text is pseudonymized -- it's the only copy that reaches an LLM.
    """
    m_fig = _FIGURE_MARKER_RE.search(text)
    clean = _FIGURE_MARKER_RE.sub('', text)
    m = _SOURCE_RE.match(clean)
    if m:
        source = {
            'page': int(m.group(1)),
            'source': m.group(2),
            'text': m.group(3).strip(),
            'score': round(score, 4),
        }
    else:
        source = {'page': 0, 'source': 'unknown', 'text': clean, 'score': round(score, 4)}
    if m_fig:
        source['image_path'] = m_fig.group(1)
    return source, pseudonymize.pseudonymize_text(clean, org_id)
```

- [ ] **Step 4: Modify `ask_file`**

```python
# app.py:1773-1795
def ask_file(question, user_id, conversation_history=None):
    """Return (response_text, sources) for a specific user's documents."""
    org_id = get_or_create_org_for_user(user_id)
    complexity = estimate_query_complexity(question)
    scored_chunks = find_relevant_chunks(question, user_id, top_n=complexity['top_n'], top_k=complexity['top_k'])
    if not scored_chunks:
        return "No documents have been indexed yet. Please upload a PDF first.", []

    sources = []
    prompt = (
        "Based on the following excerpts from documents, answer the question. "
        "Include page numbers when citing information. "
        "If the answer is not in the excerpts, say so.\n\n"
    )
    for score, text in scored_chunks:
        source, prompt_text = build_source(score, text, org_id)
        prompt += f"{prompt_text}\n\n"
        sources.append(source)

    prompt += f"Question: {pseudonymize.pseudonymize_text(question, org_id)}\nAnswer:"
    response = generate_text(prompt, conversation_history=conversation_history)
    if response is None:
        return None, []
    return pseudonymize.deanonymize_text(response, org_id), sources
```

- [ ] **Step 5: Modify `ask_file_agentic`**

```python
# app.py:1798-1880
def ask_file_agentic(question, user_id, conversation_history=None):
    """Agentic RAG: query decomp + CRAG loop + synthesis. Falls back to ask_file() on error."""
    t_total = time.perf_counter()
    try:
        if get_collection_count(user_id) == 0:
            return "No documents have been indexed yet. Please upload a PDF first.", []

        org_id = get_or_create_org_for_user(user_id)
        pseudo_question = pseudonymize.pseudonymize_text(question, org_id)

        sub_queries_pseudo = _timed("decompose_query", decompose_query, pseudo_question)
        sub_queries = [pseudonymize.deanonymize_text(sq, org_id) for sq in sub_queries_pseudo]
        logging.info(f"[agentic] decomposed into {len(sub_queries)} sub-queries: {sub_queries}")
        complexity = estimate_query_complexity(question, sub_query_count=len(sub_queries))

        prior_q = conversation_history[-1]['question'] if conversation_history else None

        all_chunks = []
        seen_texts = set()
        reformulation_count = 0

        for sub_q in sub_queries:
            retrieval_query = f"{prior_q} {sub_q}" if prior_q else sub_q
            iter_top_k = complexity['top_k']
            scored, relevant = [], []

            for iteration in range(MAX_CRAG_ITERATIONS):
                if time.perf_counter() - t_total > CRAG_WALL_CLOCK_BUDGET_S:
                    logging.warning(f"[agentic] wall-clock budget exceeded, stopping CRAG loop for '{sub_q}'")
                    break

                scored = _timed(f"retrieval_iter{iteration}", find_relevant_chunks_with_graph,
                                 retrieval_query, user_id, top_n=complexity['top_n'], top_k=iter_top_k)
                if not scored:
                    break

                texts = [text for _, text in scored]
                pseudo_texts = [pseudonymize.pseudonymize_text(t, org_id) for t in texts]
                pseudo_sub_q = pseudonymize.pseudonymize_text(sub_q, org_id)
                try:
                    relevant_pseudo, _ = _timed(f"grade_chunks_iter{iteration}", grade_chunks, pseudo_sub_q, pseudo_texts)
                    # grade_chunks returns a subset of its input list by value;
                    # map back to the matching raw texts by position.
                    relevant = [texts[pseudo_texts.index(pt)] for pt in relevant_pseudo]
                except GradingUnavailableError:
                    logging.warning(f"[agentic] grading unavailable, using retrieved chunks as-is for '{sub_q}'")
                    relevant = texts
                    break

                if relevant:
                    break

                logging.info(f"[agentic] no relevant chunks for '{retrieval_query}' (iter {iteration}), reformulating")
                pseudo_reformulated = _timed(
                    "reformulate_query", reformulate_query,
                    pseudonymize.pseudonymize_text(retrieval_query, org_id),
                )
                retrieval_query = pseudonymize.deanonymize_text(pseudo_reformulated, org_id)
                iter_top_k = int(iter_top_k * 1.5)
                reformulation_count += 1

            for score, text in scored:
                if text in seen_texts or text not in relevant:
                    continue
                seen_texts.add(text)
                all_chunks.append((score, text))

        logging.info(f"[eval] reformulations={reformulation_count} total_chunks={len(all_chunks)}")

        if not all_chunks:
            return "I couldn't find relevant information in your documents to answer this question.", []

        sources = []
        prompt = (
            "Based on the following excerpts from documents, answer the question. "
            "Include page numbers when citing information. "
            "If the answer is not in the excerpts, say so.\n\n"
        )
        for score, text in sorted(all_chunks, key=lambda x: x[0], reverse=True)[:complexity['synthesis_top_n']]:
            source, prompt_text = build_source(score, text, org_id)
            prompt += f"{prompt_text}\n\n"
            sources.append(source)

        prompt += f"Question: {pseudo_question}\nAnswer:"
        response = _timed("generate_text", generate_text, prompt, conversation_history=conversation_history)
        if response is None:
            return None, []

        response = pseudonymize.deanonymize_text(response, org_id)
        logging.info(f"[perf] ask_file_agentic total: {(time.perf_counter() - t_total) * 1000:.1f}ms")
        return response, sources

    except Exception as e:
        logging.error(f"[agentic] pipeline error, falling back to ask_file: {e}", exc_info=True)
        return ask_file(question, user_id, conversation_history=conversation_history)
```

Note the `grade_chunks` index-mapping line
(`relevant = [texts[pseudo_texts.index(pt)] for pt in relevant_pseudo]`):
`grade_chunks` returns items *by value* from whatever list it's given
(`app.py:1643`: `relevant = [chunks[i] for i in relevant_indices ...]`), so
passing it `pseudo_texts` means its output is pseudonymized strings —
translate back to the matching raw strings by position before using them
downstream (retrieval, `seen_texts` dedup, `build_source`). This only
works correctly if `pseudo_texts` has no duplicate values; if two
different chunks pseudonymize to an identical string (only possible if
they were byte-identical raw chunks — already deduplicated by
`seen_texts` from a *previous* sub-query, but not within the same
iteration), `.index()` picks the first match, which is harmless here
since duplicate raw text chunks are interchangeable anyway.

- [ ] **Step 6: Update the two `/ask` route call sites building `ask()`'s
  semantic-cache query text** (the ones touched by the embed-model-swap
  commit earlier today, `app.py` around the `ask()` route) to keep using
  raw `question` for `query_vec` — no change needed there, just confirm
  while touching this area that nothing was accidentally pseudonymized on
  the embedding path.

- [ ] **Step 7: Run tests to verify they pass**

Run: `pytest tests/test_app.py -v`
Expected: PASS for all tests, including every pre-existing
`test_ask_file_agentic_*` test — check each one's mocks for a
`build_source` or `pseudonymize`/`deanonymize` assumption and update
call signatures (`build_source(score, text)` → `build_source(score, text, org_id)`)
where the existing tests call it directly.

- [ ] **Step 8: Commit**

```bash
git add app.py tests/test_app.py
git commit -m "feat: pseudonymize question/chunks for decompose/grade/reformulate/synthesize, deanonymize response"
```

---

### Task 7: Image redaction for the vision fallback

**Files:**
- Modify: `pseudonymize.py`
- Modify: `app.py` (ingestion-time caller of `describe_image_with_groq`
  — search `describe_image_with_groq(` for call sites inside `index_pdf`)
- Test: `tests/test_pseudonymize.py`, `tests/test_app.py`

**Interfaces:**
- Consumes: `get_org_entity_types` (Task 1).
- Produces: `redact_image(png_bytes, org_id) -> bytes`. Called
  immediately before every `describe_image_with_groq` call at ingestion
  time.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pseudonymize.py (append)
def test_redact_image_calls_presidio_image_redactor(monkeypatch):
    fake_supabase = _mock_supabase_table({'orgs': [{'pseudonymize_entities': ['PERSON']}]})
    fake_redacted_bytes = b'redacted-png-bytes'

    class FakeRedactor:
        def redact(self, image, entities=None):
            return image  # PIL Image passthrough for this test

    with patch('pseudonymize.app.supabase_admin', fake_supabase), \
         patch('pseudonymize._get_image_redactor', return_value=FakeRedactor()), \
         patch('pseudonymize._png_bytes_to_pil_image') as mock_to_pil, \
         patch('pseudonymize._pil_image_to_png_bytes', return_value=fake_redacted_bytes) as mock_to_bytes:
        result = pseudonymize.redact_image(b'original-png-bytes', 'org-1')

    mock_to_pil.assert_called_once_with(b'original-png-bytes')
    mock_to_bytes.assert_called_once()
    assert result == fake_redacted_bytes
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pseudonymize.py::test_redact_image_calls_presidio_image_redactor -v`
Expected: FAIL with `AttributeError: module 'pseudonymize' has no attribute 'redact_image'`

- [ ] **Step 3: Write minimal implementation**

```python
# pseudonymize.py (append)
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
```

- [ ] **Step 4: Wire the ingestion-time caller**

Find every call to `describe_image_with_groq(` inside `index_pdf` (search
`app.py` for it — the vision-fallback comment at `app.py:476-481`
describes it as called for "content rule-based extraction can't read").
For each, redact before calling:

```python
# app.py, at each describe_image_with_groq(png_bytes, ...) call site inside index_pdf:
redacted_bytes = pseudonymize.redact_image(png_bytes, org_id)
description = describe_image_with_groq(redacted_bytes, prompt)
```

Keep the original `png_bytes` in use for the non-AI rule-based extraction
path (that never leaves the app process, per the spec) — only the copy
handed to `describe_image_with_groq` gets redacted.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_pseudonymize.py tests/test_app.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add pseudonymize.py app.py tests/test_app.py tests/test_pseudonymize.py
git commit -m "feat: redact detected entities from page images before Groq vision calls"
```

---

### Task 8: Dependencies and deploy config

**Files:**
- Modify: `requirements.txt`
- Modify: `Dockerfile`
- Modify: `.github/workflows/ingest.yml`

**Interfaces:** none — this task only makes Tasks 1-7's imports resolvable
in every environment that runs this code.

- [ ] **Step 1: Add packages to `requirements.txt`**

```
presidio-analyzer==2.2.360
presidio-anonymizer==2.2.360
presidio-image-redactor==0.0.55
pytesseract==0.3.13
spacy==3.7.5
```

Pin exact versions (check current latest compatible set at
implementation time via `pip index versions presidio-analyzer` — Presidio
ships frequently, don't guess a version that doesn't exist).

- [ ] **Step 2: Add the spaCy model download to the Docker build**

```dockerfile
# Dockerfile, builder stage, after `pip install ... -r requirements.txt`:
RUN python -m spacy download en_core_web_sm --target=/install/lib/python3.11/site-packages
```

(Adjust the `--target` path to match the actual `site-packages` location
the builder stage installs into — verify against the existing
`--prefix=/install` layout before committing.)

- [ ] **Step 3: Add `tesseract-ocr` to the Docker runtime stage**

```dockerfile
# Dockerfile, runtime stage, before COPY --from=builder:
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*
```

- [ ] **Step 4: Add `tesseract-ocr` to the GH Actions ingestion runner**

```yaml
# .github/workflows/ingest.yml, after actions/setup-python, before pip install:
- run: sudo apt-get update && sudo apt-get install -y tesseract-ocr
- run: pip install -r requirements.txt
- run: python -m spacy download en_core_web_sm
```

- [ ] **Step 5: Verify the full test suite still passes**

Run: `pytest tests/ -v`
Expected: PASS, all tests (this task doesn't touch app logic, just
confirms nothing in Tasks 1-7 was silently depending on an unpinned or
unavailable package).

- [ ] **Step 6: Commit**

```bash
git add requirements.txt Dockerfile .github/workflows/ingest.yml
git commit -m "chore: add presidio/spacy/pytesseract deps for pseudonymization"
```

- [ ] **Step 7: Push and deploy**

```bash
git push origin master
```

Wait for both the Render main-app redeploy and confirm the next PDF
upload dispatches successfully to the GH Actions ingestion workflow (it
now installs two new things — a Docker layer change on the main app and
a slower `ingest.yml` run via the spaCy model download — watch the first
post-deploy ingestion run in the Actions tab for a clean pass before
trusting the pipeline).

- [ ] **Step 8: Re-run the eval per CLAUDE.md's regression gates**

```bash
python evals/run_eval.py
```

Compare against `evals/history.json`'s current best run. Per the spec's
Testing section: `retrieval_recall`, `citation_quality`, and
`groundedness` should be unchanged (this feature never touches raw
`text`/retrieval); `weighted_rag_score` should be roughly flat. If any of
those regress, the bug is almost certainly in Task 6's index-mapping
(`grade_chunks` pseudo→raw translation) or a missed `build_source` call
site — re-check both before assuming it's noise.

---

## Notes for the implementer

- Tasks 1-3 (`pseudonymize.py`'s core) can be fully unit-tested without
  touching `app.py` at all — do them first and get them solid.
- Task 6 is the highest-risk task (most call sites, an index-mapping
  subtlety in the `grade_chunks` wrapping). Don't parallelize it with
  Tasks 4/5/7 even though they touch different files — Task 6 is where a
  mistake would most plausibly regress `weighted_rag_score`.
- The `grep -n "build_source("` step at the start of Task 6 matters: this
  plan found two call sites (`ask_file`, `ask_file_agentic`) by reading
  the code, but if a third exists (e.g. in a test helper or an unused
  code path) and gets missed, it'll be a hard `TypeError: build_source()
  missing 1 required positional argument` at runtime, not a silent bug —
  should surface immediately in Step 7's full test run.
