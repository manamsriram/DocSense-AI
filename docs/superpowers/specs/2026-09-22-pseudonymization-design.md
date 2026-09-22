# Pseudonymization Design

**Date:** 2026-09-22
**Status:** Approved for implementation

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
- No raw sensitive text reaches Groq/OpenRouter/Gemini in grading,
  reformulation, or synthesis calls.
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

## Ingestion-time pseudonymization

In `index_pdf`'s `flush()` (`app.py:739-765`), before the existing
`qdrant.upsert`:

1. Run Presidio's `AnalyzerEngine` (recognizers built from the org's
   `pseudonymize_entities` list) over each `display_text`.
2. For each detected entity, look up or create its pseudonym in
   `pseudonym_mappings` (org-scoped).
3. Substitute to produce `pseudo_text`.
4. Store **both** fields in the Qdrant payload:
   `payload={'source': ..., 'text': display_text, 'pseudo_text': pseudo_text, ...}`.

`text` is unchanged in every existing consumer — embeddings (`embed_texts`),
BM25 tokenization (`app.py:291-412`), citations, and
`_promote_table_chunk_for_aggregation`'s table-marker matching all keep
using raw text exactly as today. `pseudo_text` is new and additive.

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

## Query-time (per-request) pseudonymization

The user's question is a single short string — Presidio's cost here is
negligible even with spaCy, so this runs inline in the `/ask` request
path without the latency concern above.

At the top of the `/ask` flow (`app.py` around `ask()`/`ask_file_agentic`):

1. `raw_question` = the user's text, as today. Used for Qdrant vector
   search (`hybrid_search`, `app.py:950`) — retrieval must stay in the
   real semantic space, so this embedding call is unchanged.
2. `pseudo_question` = `raw_question` run through the same org-scoped
   Presidio + mapping substitution used at ingestion. Used for every LLM
   call: `grade_chunks`, `reformulate_query`, `generate_text` (synthesis).

### `grade_chunks`

Consumes `pseudo_question` + the chunks' `pseudo_text` instead of raw.
Returns relevance **indices** only (`app.py:1642-1644`), which still index
into the caller's raw-text list — no reversal needed on this call's output.

### `reformulate_query` — the one call with a reversal step

`reformulate_query` (`app.py:1663-1669`) is an LLM call (needs
pseudonymized input, same as grading/synthesis) but its *output* gets
re-embedded for a second Qdrant search (needs raw semantics — a
pseudonymized rewritten query would search the wrong vector space).
Resolution: pseudonymize the input as normal, then **reverse-substitute
the LLM's output** back to raw terms (using the same org mapping, pseudonym
→ real_value direction) before using it for retrieval. Same mapping table,
used in both directions — no new storage.

### `generate_text` (synthesis) and response de-anonymization

Builds its prompt from `pseudo_question` + the graded chunks' `pseudo_text`.
The returned answer may echo pseudonyms the LLM saw verbatim (e.g. "the
report was filed by PERSON_a3f1"). Before returning the answer to the user
or caching it (`cache_response`, `semantic_cache_store`), reverse-substitute
every pseudonym back to its real value — batch-fetch the org's mapping
rows once per request (not one Supabase round-trip per entity found).

Citations/sources already read the raw `text` field
(`app.py:2322`: `{'type': ..., 'text': p.payload.get('text', '')}`) — no
change needed there; the user was always meant to see real values in
citations.

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

- Unit: Presidio substitution is deterministic given a fixed mapping —
  test `pseudonymize_text(text, org_id) -> (pseudo_text, entities_found)`
  and its reverse `deanonymize_text(text, org_id)` round-trip to the
  original.
- Unit: `reformulate_query`'s reversal step — pseudonymized input in,
  raw-term output out, given a seeded mapping.
- Unit: response de-anonymization — a synthesized answer containing
  pseudonym tokens comes back with real values substituted.
- Integration: run `evals/run_eval.py` against a deploy with this change —
  per CLAUDE.md's regression gates, `retrieval_recall`, `citation_quality`,
  and `groundedness` must not regress, since `text` (raw) still drives
  retrieval and citations are unaffected. `weighted_rag_score` should be
  roughly flat; this feature isn't expected to move it either direction.
