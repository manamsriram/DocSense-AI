# Pseudonymization Design

**Date:** 2026-09-22
**Status:** Approved for implementation

**Revision (same day, during plan-writing):** tracing the actual retrieval
call sites surfaced two problems with the design as first written. Both are
corrected below rather than left as implementation surprises.

1. **Two more call sites send raw text to third-party LLMs**, missed in the
   original scope: `decompose_query` (`app.py:1601`, question → Groq, called
   at `app.py:1805`) and `extract_and_store_graph` (`app.py:1018`, chunk
   text → OpenRouter for graph-triple extraction, called at ingestion time
   from `app.py:841-846`). Both are now in scope — see their sections below.
2. **The "dual `text`/`pseudo_text` Qdrant field" plan doesn't fit how the
   pipeline actually moves data.** Retrieval (`hybrid_search`,
   `find_relevant_chunks(_with_graph)`, `_promote_table_chunk_for_aggregation`,
   the reranker) passes plain `(score, text)` tuples end-to-end, and
   `_promote_table_chunk_for_aggregation` indexes into a flat `texts` list
   by position. Threading a second parallel list through all of that to
   keep two values index-aligned is a large, regression-risky diff to an
   already carefully-tuned pipeline (see backlog.md item 1's whole saga
   over a single top_n change) — and it's unnecessary. Corrected approach:
   **don't store `pseudo_text` at all.** Ingestion-time Presidio still runs
   once per chunk, but only to populate the `pseudonym_mappings` table
   (real_value → pseudonym, org-scoped). At the LLM-call boundary, a cheap
   substitution helper (`pseudonymize_text(text, org_id)`) does a plain
   multi-string replace using the org's already-known mapping — no NER at
   request time for chunk text, so the "no query-latency cost" goal still
   holds, and the **entire retrieval pipeline is untouched**: `hybrid_search`,
   `find_relevant_chunks(_with_graph)`, `_promote_table_chunk_for_aggregation`,
   and the reranker keep operating on raw text exactly as today. Only the
   handful of places that build a prompt for or consume output from Groq/
   OpenRouter/Gemini change. The one remaining per-request NER cost is the
   user's own question (to catch entities that appear in a question but
   never in any indexed document) — same as originally scoped, still a
   single short string.

This changes "Ingestion-time pseudonymization" and "Query-time (per-request)
pseudonymization" below; read them as corrected, not the original framing.

---

## Context

DocSense-AI sends retrieved document chunks and user questions to third-party
LLM providers (Groq, OpenRouter, Gemini) for grading, query reformulation,
answer synthesis, and vision fallback on scanned pages/figures. None of that
text is currently pseudonymized — names, emails, financial identifiers, etc.
in uploaded documents reach those providers as raw text (and, for scanned
pages, as raw page images).

**Threat model:** protect sensitive data in document text from third-party
LLM providers specifically. Qdrant Cloud (the vector store) is a separate,
pre-existing third-party dependency — closing that gap conflicts with
retrieval quality (see "Rejected: pseudonymizing Qdrant-stored text" below)
and is deferred to backlog item 6 (self-hosted vector store per org).

**Goals:**
- No raw sensitive text reaches Groq/OpenRouter/Gemini in query
  decomposition, grading, reformulation, synthesis, or graph-extraction
  calls.
- No raw sensitive pixels reach Groq's vision model for scanned
  pages/figures.
- Reversible: the user still sees real values in the final answer and in
  citations — pseudonymization is transparent to them.
- Entity types to protect are configurable per org, not hardcoded.
- Zero added latency/memory risk on the `/ask` request path (the 512MB
  query-side dynos already OOM'd once under normal load — see
  `model_service/service.py` comments on `EMBED_BATCH_SIZE`/
  `RERANK_BATCH_SIZE`).
- Retrieval quality (BM25 + dense embeddings) must not regress — both
  index on `payload['text']` today and must keep doing so on raw text.

**Non-goals:**
- Protecting data from Qdrant Cloud itself (backlog item 6).
- Protecting data from other org members, or from DB/Qdrant credential
  compromise (out of scope per the threat-model decision).

---

## Why ingestion-time, not per-request

Presidio's NER (spaCy-backed `AnalyzerEngine`) is real CPU/memory work.
Running it fresh on every retrieved chunk for every question would add
latency on the `/ask` hot path — a real risk given the eval's own latency
gate (`weighted_rag_score` regression gates in `CLAUDE.md`; current live
latency is already ~29s avg, close to Render's gateway timeout per
`backlog.md`).

Ingestion already runs off that hot path: `app.py:67-69` offloads PDF
ingestion (parse + embed + graph extraction + vision) to a GitHub Actions
runner specifically because the 512MB Render dyno can't carry heavier
processing. NER fits the same pattern — run it once per chunk at ingestion
time, on the 7GB runner, and cache the result. The only per-request NER-like
work is pseudonymizing the user's one-line question, which is cheap
regardless of where it runs.

---

## Data model

New Supabase table `pseudonym_mappings`:

| column | type | notes |
|---|---|---|
| `org_id` | uuid | FK-equivalent to `orgs.id`, matches existing org-scoping pattern |
| `pseudonym` | text | e.g. `PERSON_a3f1` (entity type + short hash) |
| `real_value` | text | the original sensitive string |
| `entity_type` | text | Presidio entity type (`PERSON`, `EMAIL_ADDRESS`, ...) |
| `created_at` | timestamptz | default `now()` |

Unique constraint on `(org_id, real_value)` — the same real value always
maps to the same pseudonym within an org, so pseudonyms stay consistent
across documents and conversation turns. Looked up by `(org_id, pseudonym)`
on the reverse path. Created via Supabase dashboard/migration, same as
`orgs`/`org_members`/`graph_nodes` (no SQL lives in this repo today).

Config: one new JSONB column on the existing `orgs` table,
`pseudonymize_entities` (list of Presidio entity type strings). `null`
falls back to a default list (`PERSON`, `EMAIL_ADDRESS`, `PHONE_NUMBER`,
`US_SSN`, `CREDIT_CARD`, `IBAN_CODE`, `LOCATION`). Chosen over a new config
table to match the existing org-scoped-column pattern and avoid a config
subsystem this app doesn't otherwise have.

---

## Ingestion-time: populate the mapping only

In `index_pdf`'s `flush()` (`app.py:739-765`), before the existing
`qdrant.upsert`, run Presidio's `AnalyzerEngine` (recognizers built from the
org's `pseudonymize_entities` list) over each `display_text` and, for every
detected entity, look up or create its pseudonym in `pseudonym_mappings`
(org-scoped, `(org_id, real_value)` unique). Nothing else changes:
`qdrant.upsert`'s payload keeps only `text` (raw), exactly as today.
Embeddings, BM25 tokenization (`app.py:291-412`), citations, and
`_promote_table_chunk_for_aggregation`'s table-marker matching are
untouched — they never see a pseudonymized value.

By the time a document finishes indexing, every sensitive value it contains
has a stable pseudonym in the mapping table, ready for the substitution
helper below to use without any further NER.

## The substitution helper

Two functions, used everywhere raw text crosses into or out of a
third-party LLM call:

- `pseudonymize_text(text, org_id) -> str` — fetches the org's full
  mapping (real_value → pseudonym; batch-fetched and cached per request,
  not one Supabase round-trip per entity), sorts known real values longest
  first (so e.g. "John Smith" matches before a lone "John"), and does a
  plain string replace. No NER.
- `deanonymize_text(text, org_id) -> str` — same mapping, reverse
  direction (pseudonym → real_value), plain string replace.

Chunk text pseudonymization is therefore **free of NER at request time** —
it's a string-replace pass over text already fetched from Qdrant, using a
mapping already computed at ingestion. This is what keeps the "zero added
latency on `/ask`" goal true without needing to store a second copy of
every chunk.

### Vision fallback (image redaction)

`describe_image_with_groq` (`app.py:476`) sends raw page-image PNG bytes to
Groq's vision model — Presidio's text NER doesn't apply to pixels. Add
Presidio's image-redactor module (OCR + bounding-box redaction) as a step
before the vision call, at ingestion time (same GH Actions runner, so same
no-latency-cost reasoning as the text path applies). Detected entity regions
get boxed/blurred in the PNG before it's sent to Groq. This is a new
dependency (OCR, via the image-redactor's `pytesseract` backend) and a new
failure mode — over-redaction could blur content Groq actually needs to
read (e.g. a table's data cells). Mitigate by scoping image redaction to
the same configurable entity-type list (so e.g. `LOCATION` never triggers a
redaction box over numeric table data), and by keeping the un-redacted PNG
available for the rule-based (non-AI) extraction path, which never leaves
the app process.

---

## Query-time: wrapping the actual call sites

The full pipeline, traced through `ask_file_agentic` (`app.py:1798-1880`)
and `ask_file` (`app.py:1773-1795`), passes raw text as `(score, text)`
tuples from `hybrid_search` through reranking and
`_promote_table_chunk_for_aggregation` to `build_source`. None of that
changes. Pseudonymization wraps six call sites, each using
`pseudonymize_text`/`deanonymize_text` from the previous section:

**`decompose_query(question)` (`app.py:1601`, called `app.py:1805`).**
Sends the raw question to Groq. Wrap: pseudonymize `question` before the
call; the returned `sub_queries` list drives further Qdrant retrieval
(`app.py:1827`), so deanonymize each sub-query string before it's used as
`retrieval_query`.

**`grade_chunks(sub_q, texts)` (called `app.py:1834`).** Both arguments go
to Groq. Wrap: pseudonymize `sub_q` and each string in `texts` before the
call. Returns relevance **indices** only (`app.py:1642-1644`), which index
into the caller's original raw-text list — no reversal needed on the
output.

**`reformulate_query(retrieval_query)` (`app.py:1663-1669`, called
`app.py:1844`).** An LLM call (needs pseudonymized input) whose *output*
gets used for another Qdrant search (needs raw semantics). Wrap:
pseudonymize the input, then **deanonymize the output** before it
replaces `retrieval_query`. Same mapping, both directions, no new storage.

**`build_source(score, text)` (`app.py:1750-1770`), for the synthesis
prompt only.** Today `clean` (marker-stripped text) is used for *both* the
citation `source` dict and the `prompt_text` returned for the synthesis
prompt — they're currently the same string. Split them: `source` keeps
`clean` (raw — the user is meant to see real values in citations,
unchanged); `prompt_text` becomes `pseudonymize_text(clean, org_id)`. This
needs `org_id` threaded into `build_source`'s signature (available in both
callers — `ask_file`/`ask_file_agentic` already resolve it or can resolve
it once via `get_or_create_org_for_user(user_id)`).

**`generate_text(prompt, ...)` (called `app.py:1792`, `app.py:1871`).**
The prompt (built from already-pseudonymized `prompt_text` via
`build_source`, plus the pseudonymized question) goes to
Groq/OpenRouter/Gemini as today — no change to `generate_text` itself.
Its *return value* may echo pseudonyms verbatim (e.g. "filed by
PERSON_a3f1"). Deanonymize the response before it's returned to the
caller or cached (`cache_response`, `semantic_cache_store`).

**`extract_and_store_graph(batch_chunks, user_id, source_doc)`
(`app.py:1018`, ingestion-time, called `app.py:841-846`).** Sends raw
chunk text to OpenRouter for entity/triple extraction. Wrap: pseudonymize
each chunk's text before formatting into `_GRAPH_EXTRACT_PROMPT`. Runs at
ingestion (GH Actions runner), so no request-latency concern — but this
is a second entity-substitution pass over the same text already
pseudonymized once for the mapping population above; both reads use the
already-built mapping, so it's still just string-replace, not NER.

Citations/sources already read the raw `text` field
(`app.py:2322`: `{'type': ..., 'text': p.payload.get('text', '')}`) and
`build_source`'s `source` dict above — no change needed there; the user
was always meant to see real values in citations.

### Semantic cache interaction

The semantic cache (`semantic_cache_lookup`/`store`, `app.py:1957-2001`,
already being re-keyed on `EMBED_MODEL_VERSION` per the concurrent embed-model
swap) stores the **de-anonymized** final answer, keyed by `query_vec`
computed from `raw_question`. No interaction with the pseudonymization
mapping beyond that — cached answers are already in their final,
user-facing (real-value) form.

---

## Rejected: pseudonymizing Qdrant-stored text

Considered making the Qdrant `text` payload field itself the pseudonymized
version, closing the Qdrant-as-third-party gap. Rejected: `text` is the
exact field both BM25 tokenization (`app.py:291-412`) and dense embedding
generation read. Pseudonymizing it would tokenize placeholder strings
instead of real words (breaking keyword search) and drift the document
vector space away from a query embedded from the user's raw question
(breaking dense recall) — a direct, foreseeable hit to `retrieval_recall`,
`citation_quality`, and `groundedness`, the exact regressions CLAUDE.md's
gates exist to catch. Logged as backlog item 6 (self-hosted vector store
per org) instead, since that's the only way to remove Qdrant Cloud from
the trust boundary without degrading retrieval.

---

## New dependencies

- `presidio-analyzer`, `presidio-anonymizer` (text NER + substitution)
- `presidio-image-redactor` + `pytesseract` (image redaction for the
  vision path)
- A spaCy model (`en_core_web_sm` or similar) for Presidio's NER backend

All installed in the ingestion path's environment (GitHub Actions runner
requirements, not the 512MB query-side dynos) except the lightweight
per-question Presidio call, which needs Presidio's analyzer (not the image
redactor) available in the main app process too.

---

## Testing

- Unit: `detect_and_register_entities(text, org_id)` (ingestion-time NER +
  mapping upsert) creates one mapping row per new entity, and reuses the
  existing pseudonym on a second call with the same `real_value` — this is
  what makes pseudonyms stable across documents.
- Unit: `pseudonymize_text(text, org_id)` / `deanonymize_text(text, org_id)`
  (pure substitution, no NER) round-trip to the original given a seeded
  mapping, and leave text with no known entities unchanged.
- Unit: `reformulate_query`'s reversal step — pseudonymized input in,
  raw-term output out, given a seeded mapping.
- Unit: response de-anonymization — a synthesized answer containing
  pseudonym tokens comes back with real values substituted.
- Unit: `build_source` — `source['text']` stays raw while the returned
  `prompt_text` is pseudonymized, given a seeded mapping.
- Integration: run `evals/run_eval.py` against a deploy with this change —
  per CLAUDE.md's regression gates, `retrieval_recall`, `citation_quality`,
  and `groundedness` must not regress, since `text` (raw) still drives
  retrieval and citations are unaffected. `weighted_rag_score` should be
  roughly flat; this feature isn't expected to move it either direction.
