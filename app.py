from flask import Flask, request, jsonify, render_template, g
import networkx as nx
from werkzeug.utils import secure_filename
from functools import wraps
from cachetools import TTLCache
import base64
import tempfile
import redis
import json
import math
import os
import re
import time
import itertools
from collections import OrderedDict
import pymupdf
import numpy as np
import logging
import threading
import uuid
import gc
from datetime import datetime, timezone
from dotenv import load_dotenv
import requests
from groq import Groq
from openai import OpenAI
from google import genai
from google.genai import types as genai_types
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, PayloadSchemaType
from rank_bm25 import BM25Okapi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from supabase import create_client, Client

load_dotenv()
logging.basicConfig(level=logging.INFO)

app = Flask(__name__, static_url_path='', static_folder='.')

# LLM via Groq (free tier)
groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))
# llama-3.3-70b-versatile was deprecated 2026-06-17; gpt-oss-120b is Groq's own
# recommended replacement (qwen3.6-27b, the other suggestion, leaks <think> blocks
# into message content).
GROQ_MODEL = os.getenv('GROQ_MODEL', 'openai/gpt-oss-120b')
GROQ_VISION_MODEL = os.getenv('GROQ_VISION_MODEL', 'qwen/qwen3.6-27b')
# High-volume graph-triple extraction during ingestion runs on OpenRouter instead
# of Groq — the 70B Groq model's tight TPM/RPM limits caused 429 storms across the
# many sequential batches in a single upload, blowing past the HTTP gateway timeout.
_openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
openrouter_client = OpenAI(
    api_key=_openrouter_api_key, base_url='https://openrouter.ai/api/v1'
) if _openrouter_api_key else None
GRAPH_MODEL = os.getenv('GRAPH_MODEL', 'google/gemini-2.5-flash-lite')
# Primary model for final answer generation, routed through OpenRouter. Both Groq's
# daily token quota and Gemini's free-tier daily quota can be exhausted independently,
# so OpenRouter — a third, separately-billed provider — goes first; Groq/Gemini remain
# as fallbacks below.
ANSWER_MODEL = os.getenv('ANSWER_MODEL', 'google/gemini-2.5-flash')

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-2.0-flash')
_gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

COLLECTION = 'documents'

# Offload PDF ingestion (parse + embed + graph extraction + vision) to a GitHub
# Actions runner: the 512MB Render instance OOMs on large PDFs, the runner has 7GB.
# Unset in local dev -> /upload falls back to indexing inline on the request thread.
GITHUB_DISPATCH_TOKEN = os.getenv('GITHUB_DISPATCH_TOKEN')
GITHUB_REPO = os.getenv('GITHUB_REPO')

# Fail fast if required env vars are missing
_required = ['SUPABASE_URL', 'SUPABASE_ANON_KEY', 'SUPABASE_SERVICE_KEY', 'GROQ_API_KEY', 'QDRANT_URL', 'QDRANT_API_KEY']
_missing = [v for v in _required if not os.getenv(v)]
if _missing:
    raise EnvironmentError(f"Missing required environment variables: {', '.join(_missing)}")

# Supabase clients
supabase_anon: Client = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_ANON_KEY'))
supabase_admin: Client = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_SERVICE_KEY'))

# Redis via Upstash (optional — caching disabled gracefully if unavailable)
REDIS_URL = os.getenv('REDIS_URL')
redis_client = None
if REDIS_URL:
    try:
        redis_client = redis.from_url(REDIS_URL, socket_connect_timeout=3)
        redis_client.ping()
        logging.info("Redis (Upstash) connected")
    except Exception:
        redis_client = None
        logging.warning("Redis not available — caching disabled")

# Two-tier cache: L1 in-memory (fast, lost on restart) → L2 Upstash Redis (persistent)
_memory_cache: TTLCache = TTLCache(maxsize=100, ttl=1800)
_memory_cache_lock = threading.Lock()

# Semantic models — loaded lazily on first use to reduce startup memory
_embedding_model = None
_reranker_model = None
_model_lock = threading.Lock()


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        with _model_lock:
            if _embedding_model is None:
                logging.info("Loading embedding model via fastembed...")
                # threads=1: caps onnxruntime's intra-op thread pool/arena allocation,
                # which otherwise scales with CPU count and can exceed the 512MB dyno.
                _embedding_model = TextEmbedding(
                    model_name='sentence-transformers/all-MiniLM-L6-v2',
                    cache_dir=os.getenv('FASTEMBED_CACHE_PATH', None),
                    threads=1
                )
    return _embedding_model


def get_reranker_model():
    global _reranker_model
    if _reranker_model is None:
        with _model_lock:
            if _reranker_model is None:
                logging.info("Loading reranker via fastembed...")
                _reranker_model = TextCrossEncoder(
                    model_name='Xenova/ms-marco-MiniLM-L-6-v2',
                    cache_dir=os.getenv('FASTEMBED_CACHE_PATH', None),
                    threads=1
                )
    return _reranker_model


# Qdrant Cloud vector store
# .strip(): a stray newline in the CI secret makes the api-key header illegal
qdrant = QdrantClient(
    url=(os.getenv('QDRANT_URL') or '').strip(),
    api_key=(os.getenv('QDRANT_API_KEY') or '').strip(),
)
if not qdrant.collection_exists(COLLECTION):
    qdrant.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )
    for field in ('source', 'user_id', 'org_id'):
        qdrant.create_payload_index(
            collection_name=COLLECTION,
            field_name=field,
            field_schema=PayloadSchemaType.KEYWORD
        )
    logging.info(f"Created Qdrant collection: {COLLECTION}")
else:
    # Ensure user_id/org_id indexes exist on pre-existing collections (e.g. those
    # created before the org-tier migration added org_id-filtered scrolls).
    for field in ('user_id', 'org_id'):
        try:
            qdrant.create_payload_index(
                collection_name=COLLECTION,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD
            )
        except Exception:
            pass

# Sentence-boundary-aware chunker with overlap
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""]
)

# BM25 sparse index — per-user dicts, rebuilt from Qdrant on startup/upload
bm25_indices = {}   # user_id -> BM25Okapi
bm25_corpora = {}   # user_id -> list of (point_id, display_text)
bm25_lock = threading.Lock()
_bm25_ready = False

# Graph store — per-user NetworkX DiGraph, lazy-loaded from Supabase
_graph_store = {}
_graph_lock = threading.Lock()

# Org tier — user_id -> org_id, lazy-resolved/created via org_members
_org_id_store = {}
_org_id_lock = threading.Lock()


# ---- Perf helpers ----

def _timed(label, fn, *args, **kwargs):
    t = time.perf_counter()
    result = fn(*args, **kwargs)
    logging.info(f"[perf] {label}: {(time.perf_counter() - t) * 1000:.1f}ms")
    return result


# ---- Auth ----

def _require_auth_impl(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization', '')
        if not auth_header.startswith('Bearer '):
            return jsonify({'error': 'Unauthorized'}), 401
        token = auth_header[7:]
        try:
            user_response = supabase_admin.auth.get_user(token)
            if not user_response.user:
                return jsonify({'error': 'Unauthorized'}), 401
            g.user_id = str(user_response.user.id)
        except Exception:
            return jsonify({'error': 'Unauthorized'}), 401
        return f(*args, **kwargs)
    return decorated


def require_auth(f):
    import sys
    _self = require_auth  # capture the original function object as a closure variable
    @wraps(f)
    def decorated(*args, **kwargs):
        current = sys.modules[__name__].require_auth
        # If patched, delegate to the patched version; otherwise run impl directly.
        if current is not _self:
            return current(f)(*args, **kwargs)
        return _require_auth_impl(f)(*args, **kwargs)
    return decorated


# ---- Indexing ----

def _bm25_force_full_rebuild():
    """Escape hatch: set BM25_FORCE_FULL_REBUILD=true to revert to the old full-scan
    rebuild-on-every-upload behavior if the incremental path misbehaves in production,
    without needing a redeploy."""
    return os.getenv('BM25_FORCE_FULL_REBUILD', 'false').lower() == 'true'


def _full_rebuild_bm25_for_user(org_id):
    records, _ = qdrant.scroll(
        collection_name=COLLECTION,
        scroll_filter=Filter(must=[FieldCondition(key='org_id', match=MatchValue(value=org_id))]),
        with_payload=True,
        limit=10000
    )
    if not records:
        with bm25_lock:
            bm25_indices.pop(org_id, None)
            bm25_corpora.pop(org_id, None)
        return

    texts = [r.payload.get('text', '') for r in records]
    ids = [str(r.id) for r in records]
    tokenized = [t.lower().split() for t in texts]

    with bm25_lock:
        bm25_indices[org_id] = BM25Okapi(tokenized)
        bm25_corpora[org_id] = list(zip(ids, texts))
    logging.info(f"BM25 rebuilt for org {org_id} with {len(texts)} chunks")


def rebuild_bm25_for_user(user_id, source_filename=None):
    """Update a user's org-shared BM25 index after an upload.

    When source_filename is given (the normal /upload case), only that document's
    chunks are fetched from Qdrant and merged into the existing in-memory corpus —
    avoids rescanning the org's entire collection on every upload. Falls back to a
    full rebuild on first load (no source_filename), when explicitly forced via
    BM25_FORCE_FULL_REBUILD, or if the incremental path errors.

    # ponytail: no transactional guarantee between Qdrant and this in-memory index —
    # if a process crashes between the Qdrant upsert and this call, they drift apart
    # with no self-healing. Add periodic rebuild_all_bm25() as a scheduled
    # reconciliation job if drift becomes an observed problem.
    """
    org_id = get_or_create_org_for_user(user_id)
    if source_filename and not _bm25_force_full_rebuild():
        try:
            _incremental_update_bm25_for_user(org_id, source_filename)
            return
        except Exception as e:
            logging.warning(f"[bm25] incremental update failed for org {org_id}, "
                             f"falling back to full rebuild: {e}")
    _full_rebuild_bm25_for_user(org_id)


def _incremental_update_bm25_for_user(org_id, source_filename):
    """Fetch only source_filename's chunks and merge into the existing org corpus.

    Held under bm25_lock for the entire read-merge-rebuild so a concurrent upload
    within the same org can't interleave and lose an append.
    """
    records, _ = qdrant.scroll(
        collection_name=COLLECTION,
        scroll_filter=Filter(must=[
            FieldCondition(key='org_id', match=MatchValue(value=org_id)),
            FieldCondition(key='source', match=MatchValue(value=source_filename)),
        ]),
        with_payload=True,
        limit=10000
    )
    ids = [str(r.id) for r in records]
    texts = [r.payload.get('text', '') for r in records]

    # display_text always starts with "[Page N, Source: <filename>]" (see index_pdf) —
    # use that marker to drop this source's stale entries regardless of whether their
    # point ids changed across a reindex (e.g. the doc got shorter/longer).
    source_marker = f", Source: {source_filename}]"

    with bm25_lock:
        existing = [(cid, text) for cid, text in bm25_corpora.get(org_id, [])
                    if source_marker not in text]
        combined = existing + list(zip(ids, texts))
        if not combined:
            bm25_indices.pop(org_id, None)
            bm25_corpora.pop(org_id, None)
            return
        tokenized = [t.lower().split() for _, t in combined]
        bm25_indices[org_id] = BM25Okapi(tokenized)
        bm25_corpora[org_id] = combined
    logging.info(f"[bm25] incrementally updated org {org_id} "
                 f"({len(records)} chunks for {source_filename!r}, {len(combined)} total)")


def rebuild_all_bm25():
    """Rebuild per-org BM25 indices one org at a time (filtered scrolls) instead
    of loading every org's chunks into memory at once — bounds peak memory to a
    single org's corpus regardless of total collection size."""
    org_ids = set()
    offset = None
    while True:
        batch, offset = qdrant.scroll(
            collection_name=COLLECTION, with_payload=['org_id'], limit=1000, offset=offset
        )
        org_ids.update(r.payload.get('org_id') for r in batch if r.payload.get('org_id'))
        if offset is None:
            break

    with bm25_lock:
        bm25_indices.clear()
        bm25_corpora.clear()

    for org_id in org_ids:
        ids, texts = [], []
        offset = None
        while True:
            batch, offset = qdrant.scroll(
                collection_name=COLLECTION,
                scroll_filter=Filter(must=[FieldCondition(key='org_id', match=MatchValue(value=org_id))]),
                with_payload=True,
                limit=1000,
                offset=offset
            )
            ids.extend(str(r.id) for r in batch)
            texts.extend(r.payload.get('text', '') for r in batch)
            if offset is None:
                break
        tokenized = [t.lower().split() for t in texts]
        with bm25_lock:
            bm25_indices[org_id] = BM25Okapi(tokenized)
            bm25_corpora[org_id] = list(zip(ids, texts))
        del ids, texts, tokenized
        gc.collect()
    logging.info(f"BM25 rebuilt for {len(org_ids)} org(s)")


def preprocess(text):
    text = text.replace('\n', ' ')
    return re.sub(r'\s+', ' ', text).strip()


def pdf_to_pages(path):
    try:
        doc = pymupdf.open(path)
        pages = []
        for page_num in range(doc.page_count):
            text = preprocess(doc[page_num].get_text('text'))
            if text:
                pages.append((page_num + 1, text))
        doc.close()
        return pages
    except Exception as e:
        logging.error(f"Error reading PDF {path}: {e}")
        return []


# ---- Multimodal extraction (rule-based, no LLM) ----

CAPTION_RE = re.compile(r'^\s*(figure|fig\.|table|chart|exhibit)\s*\d', re.IGNORECASE)
MIN_FIGURE_DIM_PT = 60          # skip icons/logos smaller than this
MAX_FIGURE_PAGE_RATIO = 0.9     # skip full-page scans (no caption to harvest)
MAX_FIGURES_PER_PAGE = 4
CAPTION_SEARCH_MARGIN_PT = 90   # max vertical gap between image and its caption
FIGURE_RENDER_DPI = 150
MAX_TABLE_ROWS_PER_CHUNK = 25
SCANNED_PAGE_RENDER_DPI = 120
MAX_VISION_B64_BYTES = 4 * 1024 * 1024   # Groq base64 image limit
MAX_VISION_CALLS_PER_DOC = 20            # rate-limit guard for large uploads

GRAPH_BATCH_SIZE = 4
MAX_GRAPH_BATCHES_PER_DOC = 50
MAX_GRAPH_NEIGHBORS = 10
INDEX_FLUSH_CHUNK_SIZE = 200   # embed+upsert every N chunks instead of buffering the whole PDF

_FIGURE_CAPTION_PROMPT = (
    'Describe this figure from a document in 2-3 sentences for search indexing. '
    'State the chart type, axis labels, and key values or trends if visible. '
    'Return only the description.'
)

_SCANNED_PAGE_PROMPT = (
    'Transcribe all readable text from this scanned document page in reading order. '
    'Render any tables as markdown. Skip unreadable parts silently. '
    'Return only the transcription.'
)


def describe_image_with_groq(png_bytes, prompt, max_tokens=300):
    """Vision fallback (Groq Llama 4) for content rule-based extraction can't read.

    Returns description text, or None on oversized input or API failure so
    callers degrade to the no-AI behavior.
    """
    b64 = base64.b64encode(png_bytes).decode()
    if len(b64) > MAX_VISION_B64_BYTES:
        logging.warning(f"Image exceeds Groq vision size limit ({len(b64)} b64 bytes), skipping")
        return None
    try:
        resp = groq_client.chat.completions.create(
            model=GROQ_VISION_MODEL,
            messages=[{
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': prompt},
                    {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{b64}'}},
                ],
            }],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        return preprocess(resp.choices[0].message.content) or None
    except Exception as e:
        logging.warning(f"Groq vision call failed: {e}")
        return None


def _rows_to_markdown(rows):
    """Render table rows (list of lists) as markdown; first row is the header."""
    def fmt(cell):
        return preprocess(str(cell)) if cell is not None else ''
    header = [fmt(c) for c in rows[0]]
    lines = [
        '| ' + ' | '.join(header) + ' |',
        '| ' + ' | '.join(['---'] * len(header)) + ' |',
    ]
    for row in rows[1:]:
        lines.append('| ' + ' | '.join(fmt(c) for c in row) + ' |')
    return '\n'.join(lines)


def extract_page_tables(page):
    """Detect tables via PyMuPDF's rule-based finder. Returns markdown strings.

    Long tables are split into row groups with the header repeated so each
    chunk stands alone for retrieval.
    """
    try:
        tabs = page.find_tables()
    except Exception as e:
        logging.warning(f"find_tables failed on page {page.number + 1}: {e}")
        return []
    out = []
    for tab in tabs.tables:
        rows = [r for r in tab.extract() if any(c not in (None, '') for c in r)]
        if len(rows) < 2 or max(len(r) for r in rows) < 2:
            continue
        header, body = rows[0], rows[1:]
        for i in range(0, len(body), MAX_TABLE_ROWS_PER_CHUNK):
            out.append(_rows_to_markdown([header] + body[i:i + MAX_TABLE_ROWS_PER_CHUNK]))
    return out


def _find_caption(blocks, img_rect):
    """Nearest horizontally-overlapping text block above/below an image.

    Blocks matching 'Figure N'-style patterns win over plain proximity.
    """
    candidates = []
    for x0, y0, x1, y1, text, *_ in blocks:
        text = preprocess(text)
        if not text:
            continue
        if x1 < img_rect.x0 or x0 > img_rect.x1:
            continue
        if y0 >= img_rect.y1:
            gap = y0 - img_rect.y1       # block below image
        elif y1 <= img_rect.y0:
            gap = img_rect.y0 - y1       # block above image
        else:
            continue
        if gap > CAPTION_SEARCH_MARGIN_PT:
            continue
        candidates.append((not CAPTION_RE.match(text), gap, text))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][2]


def extract_page_figures(page, vision_budget=None):
    """Returns [(caption, png_bytes)] for page images worth indexing.

    Captions come from nearby text when available; otherwise a Groq vision
    call describes the figure, spending from vision_budget ({'remaining': N}).
    With no budget and no nearby text the figure is skipped — nothing to index.
    """
    try:
        infos = page.get_image_info()
    except Exception as e:
        logging.warning(f"get_image_info failed on page {page.number + 1}: {e}")
        return []
    page_rect = page.rect
    blocks = [b for b in page.get_text('blocks') if b[6] == 0]
    figures = []
    for info in infos:
        if len(figures) >= MAX_FIGURES_PER_PAGE:
            break
        rect = pymupdf.Rect(info['bbox'])
        if rect.width < MIN_FIGURE_DIM_PT or rect.height < MIN_FIGURE_DIM_PT:
            continue
        if rect.get_area() > MAX_FIGURE_PAGE_RATIO * page_rect.get_area():
            continue
        try:
            pix = page.get_pixmap(clip=rect & page_rect, dpi=FIGURE_RENDER_DPI)
            png_bytes = pix.tobytes('png')
        except Exception as e:
            logging.warning(f"Figure render failed on page {page.number + 1}: {e}")
            continue
        caption = _find_caption(blocks, rect)
        if not caption and vision_budget and vision_budget['remaining'] > 0:
            caption = describe_image_with_groq(png_bytes, _FIGURE_CAPTION_PROMPT)
            if caption:
                vision_budget['remaining'] -= 1
        if not caption:
            continue
        figures.append((caption, png_bytes))
    return figures


def _delete_stored_figures(figure_dir):
    try:
        entries = supabase_admin.storage.from_('pdfs').list(figure_dir)
        paths = [f"{figure_dir}/{e['name']}" for e in (entries or [])]
        if paths:
            supabase_admin.storage.from_('pdfs').remove(paths)
    except Exception as e:
        logging.warning(f"Could not clear old figures in {figure_dir}: {e}")


def _upload_figure(path, png_bytes):
    try:
        supabase_admin.storage.from_('pdfs').upload(
            path=path,
            file=png_bytes,
            file_options={'content-type': 'image/png'}
        )
        return True
    except Exception as e:
        logging.warning(f"Figure upload failed for {path}: {e}")
        return False


def index_pdf(pdf_path, user_id, force=False, display_name=None):
    """Extract, chunk, embed, and upsert a PDF into Qdrant for a specific user.

    Payload carries both user_id (provenance/ownership — this user's own
    dedup/force-reindex checks below stay scoped to it) and org_id (the
    retrieval-scoping key shared across org members).
    """
    org_id = get_or_create_org_for_user(user_id)
    filename = display_name or os.path.basename(pdf_path)

    if force:
        qdrant.delete(
            collection_name=COLLECTION,
            points_selector=Filter(must=[
                FieldCondition(key='source', match=MatchValue(value=filename)),
                FieldCondition(key='user_id', match=MatchValue(value=user_id))
            ])
        )
        logging.info(f"Deleted existing chunks for {filename} (user {user_id[:8]}...)")
        try:
            supabase_admin.table('graph_edges').delete()\
                .eq('user_id', user_id).eq('source_doc', filename).execute()
        except Exception as e:
            logging.warning(f"[graph] edge cleanup failed for {filename}: {e}")
    else:
        existing, _ = qdrant.scroll(
            collection_name=COLLECTION,
            scroll_filter=Filter(must=[
                FieldCondition(key='source', match=MatchValue(value=filename)),
                FieldCondition(key='user_id', match=MatchValue(value=user_id))
            ]),
            limit=1
        )
        if existing:
            logging.info(f"Already indexed: {filename}")
            return 0

    try:
        doc = pymupdf.open(pdf_path)
    except Exception as e:
        logging.error(f"Error reading PDF {pdf_path}: {e}")
        return 0

    figure_dir = f"{user_id}/figures/{secure_filename(filename)}"
    if force:
        _delete_stored_figures(figure_dir)

    points = []       # (point_id, page_num, display_text, extra_payload) — current flush buffer
    embed_texts = []
    graph_eligible = []   # (point_id, page_num, display_text) across all flushes, for graph pass below
    total_indexed = 0
    counts = {'text': 0, 'table': 0, 'figure': 0, 'scanned': 0}
    vision_budget = {'remaining': MAX_VISION_CALLS_PER_DOC}

    def flush():
        nonlocal points, embed_texts, total_indexed
        if not points:
            return
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
        graph_eligible.extend(
            (point_id, page_num, display_text)
            for point_id, page_num, display_text, extra in points
            if extra.get('type') in ('text', 'table', 'scanned')
        )
        total_indexed += len(points)
        points = []
        embed_texts = []
        del vecs
        gc.collect()

    for page_num in range(doc.page_count):
        page = doc[page_num]
        pno = page_num + 1

        page_text = preprocess(page.get_text('text'))
        chunk_type = 'text'
        if not page_text and vision_budget['remaining'] > 0:
            # No text layer — likely a scanned page; transcribe via Groq vision
            try:
                pix = page.get_pixmap(dpi=SCANNED_PAGE_RENDER_DPI)
                page_text = describe_image_with_groq(
                    pix.tobytes('png'), _SCANNED_PAGE_PROMPT, max_tokens=1500
                )
            except Exception as e:
                logging.warning(f"Scanned-page render failed on page {pno}: {e}")
                page_text = None
            if page_text:
                vision_budget['remaining'] -= 1
                chunk_type = 'scanned'

        if page_text:
            for chunk_idx, chunk in enumerate(text_splitter.split_text(page_text)):
                if not chunk.strip():
                    continue
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{user_id}__{filename}__p{pno}__c{chunk_idx}"))
                display_text = f"[Page {pno}, Source: {filename}] {chunk}"
                embed_texts.append(chunk)
                points.append((point_id, pno, display_text, {'type': chunk_type}))
                counts[chunk_type] += 1

        for t_idx, table_md in enumerate(extract_page_tables(page)):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{user_id}__{filename}__p{pno}__t{t_idx}"))
            display_text = f"[Page {pno}, Source: {filename}] [Table]\n{table_md}"
            embed_texts.append(table_md)
            points.append((point_id, pno, display_text, {'type': 'table'}))
            counts['table'] += 1

        for f_idx, (caption, png_bytes) in enumerate(extract_page_figures(page, vision_budget=vision_budget)):
            image_path = f"{figure_dir}/p{pno}_f{f_idx}.png"
            extra = {'type': 'figure'}
            marker = ''
            if _upload_figure(image_path, png_bytes):
                extra['image_path'] = image_path
                marker = f"[Figure: {image_path}] "
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{user_id}__{filename}__p{pno}__f{f_idx}"))
            display_text = f"[Page {pno}, Source: {filename}] {marker}{caption}"
            embed_texts.append(caption)
            points.append((point_id, pno, display_text, extra))
            counts['figure'] += 1

        if len(points) >= INDEX_FLUSH_CHUNK_SIZE:
            flush()

    doc.close()
    flush()

    if total_indexed == 0:
        return 0

    logging.info(
        f"Indexed {total_indexed} chunks from {filename} for user {user_id[:8]}... "
        f"({counts['text']} text, {counts['table']} table, {counts['figure']} figure, "
        f"{counts['scanned']} scanned)"
    )

    batches_run = 0
    for i in range(0, len(graph_eligible), GRAPH_BATCH_SIZE):
        if batches_run >= MAX_GRAPH_BATCHES_PER_DOC:
            logging.info(f"[graph] batch cap reached for {filename}")
            break
        extract_and_store_graph(
            graph_eligible[i: i + GRAPH_BATCH_SIZE], user_id, filename
        )
        batches_run += 1
    logging.info(f"[graph] ran {batches_run} extraction batches for {filename}")

    return total_indexed


def init_index():
    global _bm25_ready
    rebuild_all_bm25()
    _bm25_ready = True
    logging.info("Startup init complete.")


# ---- Retrieval ----

def get_collection_count(user_id):
    org_id = get_or_create_org_for_user(user_id)
    try:
        result = qdrant.count(
            collection_name=COLLECTION,
            count_filter=Filter(must=[FieldCondition(key='org_id', match=MatchValue(value=org_id))])
        )
        return result.count
    except Exception:
        return 0


def get_total_collection_count():
    """Total point count across all users — the signal for Pass 2 item F
    (Qdrant sharding). Not scoped by user_id, unlike get_collection_count."""
    try:
        return qdrant.count(collection_name=COLLECTION).count
    except Exception:
        return 0


def check_qdrant_shard_threshold():
    """Instrumentation-only, no migration: log a warning once the collection
    approaches a size where a single Qdrant collection may need sharding.
    Threshold is env-overridable; see backlog item F for the decision this
    should trigger (not built until the threshold is actually approached)."""
    threshold = int(os.getenv('QDRANT_SHARD_ALERT_THRESHOLD', '1000000'))
    count = get_total_collection_count()
    if count >= threshold:
        logging.warning(
            f"[qdrant] collection size {count} has reached the shard alert "
            f"threshold ({threshold}) — see backlog item F for sharding strategy"
        )


_COMPLEXITY_HINT_RE = re.compile(r'\b(and|versus|vs\.?|compare|between|difference)\b', re.IGNORECASE)


def estimate_query_complexity(question, sub_query_count=1):
    """Heuristic retrieval-width sizing. Weights sub_query_count (a real signal already
    computed by decompose_query) over word count, which is a weak, easily-fooled proxy —
    e.g. "compare X and Y" is short but hard, a long single-fact question is easy."""
    is_complex = sub_query_count > 1 or bool(_COMPLEXITY_HINT_RE.search(question))
    if is_complex:
        return {'top_k': 30, 'top_n': 8, 'synthesis_top_n': 8}
    return {'top_k': 20, 'top_n': 5, 'synthesis_top_n': 6}


def reciprocal_rank_fusion(ranked_lists, k=60):
    scores = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def hybrid_search(query, user_id, top_k=20):
    """Combine dense (Qdrant) and sparse (BM25) retrieval via RRF, org-scoped
    so every member of a user's org shares the same retrieval corpus."""
    org_id = get_or_create_org_for_user(user_id)
    count = get_collection_count(user_id)
    if count == 0:
        return []

    # Dense retrieval via Qdrant, filtered to this org
    query_vec = list(get_embedding_model().embed([query]))[0].tolist()
    hits = qdrant.query_points(
        collection_name=COLLECTION,
        query=query_vec,
        query_filter=Filter(must=[FieldCondition(key='org_id', match=MatchValue(value=org_id))]),
        limit=min(top_k, count),
        with_payload=True
    ).points
    dense_ids = [str(h.id) for h in hits]
    dense_doc_map = {str(h.id): h.payload.get('text', '') for h in hits}

    # Sparse retrieval via the org-shared BM25 index
    with bm25_lock:
        local_bm25 = bm25_indices.get(org_id)
        local_corpus = list(bm25_corpora.get(org_id, []))

    sparse_ids = []
    if local_bm25 and local_corpus:
        scores = local_bm25.get_scores(query.lower().split())
        top_indices = np.argsort(scores)[::-1][:top_k]
        sparse_ids = [local_corpus[i][0] for i in top_indices if scores[i] > 0]

    # RRF fusion
    fused = reciprocal_rank_fusion([dense_ids, sparse_ids])
    corpus_map = {cid: text for cid, text in local_corpus}
    corpus_map.update(dense_doc_map)

    candidates = []
    for doc_id, _ in fused[:top_k]:
        text = corpus_map.get(doc_id)
        if text:
            candidates.append((doc_id, text))
    return candidates


def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def find_relevant_chunks(query, user_id, top_n=3, top_k=20):
    """Return [(score, text), ...] reranked for a specific user."""
    if get_collection_count(user_id) == 0:
        return []

    candidates = hybrid_search(query, user_id, top_k=top_k)
    if not candidates:
        return []

    texts = [text for _, text in candidates]
    scores = list(get_reranker_model().rerank(query, texts))
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return [(_sigmoid(float(score)), texts[idx]) for idx, score in ranked[:top_n]]


# ---- Graph RAG ----

def extract_and_store_graph(batch_chunks, user_id, source_doc):
    """Extract entity triples from a chunk batch and persist to Supabase.

    batch_chunks: list of (chunk_id, page_num, text)
    Failures are logged and silently skipped — must never block document indexing.
    Rows are org-scoped (org_id is the query/uniqueness key, shared across org
    members) while user_id is kept for provenance/ownership only.
    """
    org_id = get_or_create_org_for_user(user_id)
    formatted = '\n\n'.join(
        f'[{i}] {text[:600]}' for i, (_, _, text) in enumerate(batch_chunks)
    )
    try:
        raw = _call_openrouter_helper(
            _GRAPH_EXTRACT_PROMPT.format(chunks=formatted), max_tokens=700
        )
        results = _parse_llm_json(raw)
        if not isinstance(results, list):
            return
    except Exception as e:
        logging.warning(f"[graph] extraction failed for batch in {source_doc}: {e}")
        return

    nodes_to_upsert = {}
    edges_to_insert = []

    for item in results:
        idx = item.get('chunk_index', 0)
        if idx >= len(batch_chunks):
            continue
        chunk_id, page_num, _ = batch_chunks[idx]

        for ent in item.get('entities', []):
            name = ent.get('name', '').strip().lower()
            if not name:
                continue
            aliases = {a.strip().lower() for a in ent.get('aliases', []) if a and a.strip()}
            if name not in nodes_to_upsert:
                nodes_to_upsert[name] = {
                    'entity_type': ent.get('type'),
                    'aliases': aliases,
                }
            else:
                nodes_to_upsert[name]['aliases'] |= aliases

        for triple in item.get('triples', []):
            subj = triple.get('subject', '').strip().lower()
            rel  = triple.get('relation', '').strip().lower()
            obj  = triple.get('object', '').strip().lower()
            if not (subj and rel and obj):
                continue
            edges_to_insert.append({
                'user_id': user_id,
                'org_id': org_id,
                'source_entity': subj,
                'relation': rel,
                'target_entity': obj,
                'chunk_id': chunk_id,
                'source_doc': source_doc,
                'page_num': page_num,
            })

    if nodes_to_upsert:
        try:
            existing_res = supabase_admin.table('graph_nodes')\
                .select('entity_name,aliases').eq('org_id', org_id).execute()
            existing_aliases = {
                n['entity_name']: set(n.get('aliases') or []) for n in (existing_res.data or [])
            }
        except Exception as e:
            logging.warning(f"[graph] existing-alias lookup failed, upserting without merge: {e}")
            existing_aliases = {}

        rows = [
            {
                'user_id': user_id,
                'org_id': org_id,
                'entity_name': name,
                'entity_type': node['entity_type'],
                'aliases': sorted(node['aliases'] | existing_aliases.get(name, set())),
            }
            for name, node in nodes_to_upsert.items()
        ]
        try:
            supabase_admin.table('graph_nodes').upsert(
                rows, on_conflict='org_id,entity_name'
            ).execute()
        except Exception as e:
            logging.warning(f"[graph] node upsert failed: {e}")

    if edges_to_insert:
        try:
            supabase_admin.table('graph_edges').insert(edges_to_insert).execute()
        except Exception as e:
            logging.warning(f"[graph] edge insert failed: {e}")


def _build_graph_from_supabase(org_id):
    try:
        nodes_res = supabase_admin.table('graph_nodes')\
            .select('entity_name,entity_type,aliases')\
            .eq('org_id', org_id).execute()
        edges_res = supabase_admin.table('graph_edges')\
            .select('source_entity,relation,target_entity,chunk_id,source_doc,page_num')\
            .eq('org_id', org_id).execute()
    except Exception as e:
        logging.warning(f"[graph] failed to load graph for org {org_id}: {e}")
        return nx.DiGraph()

    G = nx.DiGraph()
    for node in (nodes_res.data or []):
        G.add_node(node['entity_name'], entity_type=node.get('entity_type'),
                   aliases=node.get('aliases') or [])
    for edge in (edges_res.data or []):
        G.add_edge(
            edge['source_entity'], edge['target_entity'],
            relation=edge['relation'],
            chunk_id=edge['chunk_id'],
            source_doc=edge['source_doc'],
            page_num=edge.get('page_num'),
        )
    logging.info(
        f"[graph] loaded {G.number_of_nodes()} nodes, {G.number_of_edges()} "
        f"edges for org {org_id}"
    )
    return G


def get_graph_for_user(user_id):
    org_id = get_or_create_org_for_user(user_id)
    with _graph_lock:
        if org_id in _graph_store:
            return _graph_store[org_id]
    G = _build_graph_from_supabase(org_id)
    with _graph_lock:
        if org_id not in _graph_store:
            _graph_store[org_id] = G
        return _graph_store[org_id]


def get_or_create_org_for_user(user_id):
    """Resolve a user's current org, creating a solo org on first call.

    Single choke point every org-scoped layer (graph, BM25, Qdrant, semantic
    cache) reads through — every user is a member of exactly one org (1:1),
    defaulting to a solo org they administer.
    """
    with _org_id_lock:
        if user_id in _org_id_store:
            return _org_id_store[user_id]

    membership = supabase_admin.table('org_members')\
        .select('org_id').eq('user_id', user_id).execute()
    rows = membership.data or []
    if rows:
        org_id = rows[0]['org_id']
    else:
        org_res = supabase_admin.table('orgs').insert({'name': None}).execute()
        org_id = org_res.data[0]['id']
        supabase_admin.table('org_members').insert({
            'org_id': org_id, 'user_id': user_id, 'role': 'admin',
        }).execute()

    with _org_id_lock:
        _org_id_store[user_id] = org_id
        return _org_id_store[user_id]


def request_join_org(user_id, org_id):
    """File a pending request for user_id to join org_id.

    Does not touch the user's current membership — that only changes on
    approve_join_request, so a pending request never leaks visibility into
    the target org.
    """
    supabase_admin.table('org_join_requests').insert({
        'org_id': org_id, 'user_id': user_id, 'status': 'pending',
    }).execute()


def approve_join_request(request_id, approving_admin_id):
    """Approve a pending join request: swap membership and migrate the user's
    data from their old (solo) org into the target org.

    Raises PermissionError if approving_admin_id isn't an admin of the
    request's target org.
    """
    req_res = supabase_admin.table('org_join_requests')\
        .select('*').eq('id', request_id).execute()
    req = req_res.data[0]
    target_org_id = req['org_id']
    joining_user_id = req['user_id']

    admin_check = supabase_admin.table('org_members').select('user_id')\
        .eq('org_id', target_org_id).eq('user_id', approving_admin_id)\
        .eq('role', 'admin').execute()
    if not admin_check.data:
        raise PermissionError(
            f"{approving_admin_id} is not an admin of org {target_org_id}"
        )

    old_membership = supabase_admin.table('org_members').select('org_id')\
        .eq('user_id', joining_user_id).execute()
    old_org_id = old_membership.data[0]['org_id'] if old_membership.data else None

    supabase_admin.table('org_members').delete().eq('user_id', joining_user_id).execute()
    supabase_admin.table('org_members').insert({
        'org_id': target_org_id, 'user_id': joining_user_id, 'role': 'member',
    }).execute()
    supabase_admin.table('org_join_requests').update({
        'status': 'approved', 'decided_by': approving_admin_id,
        'decided_at': datetime.now(timezone.utc).isoformat(),
    }).eq('id', request_id).execute()

    with _org_id_lock:
        _org_id_store[joining_user_id] = target_org_id

    if old_org_id and old_org_id != target_org_id:
        _migrate_org_graph_data(old_org_id, target_org_id)
        _migrate_org_qdrant_payloads(old_org_id, target_org_id)

    return target_org_id


def _migrate_org_graph_data(old_org_id, new_org_id):
    """Reassign all graph rows from old_org_id to new_org_id.

    old_org_id is always the joining user's prior solo org (1:1 membership),
    so every row under it belongs solely to that user — safe to move in full
    rather than filtering by user_id. Node aliases are merged (org-scoped
    uniqueness on entity_name), edges are reassigned outright since they
    carry no such constraint.
    """
    nodes_res = supabase_admin.table('graph_nodes')\
        .select('entity_name,entity_type,aliases').eq('org_id', old_org_id).execute()
    target_res = supabase_admin.table('graph_nodes')\
        .select('entity_name,aliases').eq('org_id', new_org_id).execute()
    existing_aliases = {
        n['entity_name']: set(n.get('aliases') or []) for n in (target_res.data or [])
    }
    rows = [
        {
            'org_id': new_org_id,
            'entity_name': n['entity_name'],
            'entity_type': n.get('entity_type'),
            'aliases': sorted(set(n.get('aliases') or []) | existing_aliases.get(n['entity_name'], set())),
        }
        for n in (nodes_res.data or [])
    ]
    if rows:
        supabase_admin.table('graph_nodes').upsert(rows, on_conflict='org_id,entity_name').execute()
        supabase_admin.table('graph_nodes').delete().eq('org_id', old_org_id).execute()
    supabase_admin.table('graph_edges').update({'org_id': new_org_id}).eq('org_id', old_org_id).execute()

    with _graph_lock:
        _graph_store.pop(old_org_id, None)
        _graph_store.pop(new_org_id, None)


def _migrate_org_qdrant_payloads(old_org_id, new_org_id):
    """Repoint every Qdrant point tagged with old_org_id to new_org_id.

    old_org_id is always the joining user's prior solo org, so every point
    under it belongs solely to that user. BM25's cached indices for both
    orgs are evicted so the next lazy rebuild reflects the merged corpus.
    """
    qdrant.set_payload(
        collection_name=COLLECTION,
        payload={'org_id': new_org_id},
        points_selector=Filter(must=[FieldCondition(key='org_id', match=MatchValue(value=old_org_id))]),
    )
    with bm25_lock:
        bm25_indices.pop(old_org_id, None)
        bm25_corpora.pop(old_org_id, None)
        bm25_indices.pop(new_org_id, None)
        bm25_corpora.pop(new_org_id, None)


def backfill_qdrant_org_ids():
    """One-time migration for points that predate org_id and carry only user_id.

    Scrolls the whole collection, groups legacy points by user_id, and batches
    set_payload calls per resolved org. Safe to re-run — points that already
    have org_id are skipped.
    """
    all_records = []
    offset = None
    while True:
        batch, offset = qdrant.scroll(
            collection_name=COLLECTION, with_payload=True, limit=1000, offset=offset
        )
        all_records.extend(batch)
        if offset is None:
            break

    legacy_by_user = {}
    for r in all_records:
        if r.payload.get('org_id'):
            continue
        uid = r.payload.get('user_id')
        if uid:
            legacy_by_user.setdefault(uid, []).append(str(r.id))

    updated = 0
    for uid, point_ids in legacy_by_user.items():
        org_id = get_or_create_org_for_user(uid)
        qdrant.set_payload(collection_name=COLLECTION, payload={'org_id': org_id}, points=point_ids)
        updated += len(point_ids)

    logging.info(f"[qdrant] backfilled org_id for {updated} point(s) across {len(legacy_by_user)} user(s)")
    return updated


def _graph_force_full_rebuild():
    """Escape hatch mirroring BM25_FORCE_FULL_REBUILD: set GRAPH_FORCE_FULL_REBUILD=true
    to revert to the old evict+reload-from-Supabase behavior if the incremental
    patch path misbehaves in production, without needing a redeploy."""
    return os.getenv('GRAPH_FORCE_FULL_REBUILD', 'false').lower() == 'true'


def rebuild_graph_for_user(user_id, source_filename=None):
    """Refresh a user's in-memory graph after an upload.

    When source_filename is given (the normal /upload case), patch the live
    nx.DiGraph in place — sweep-remove edges belonging to that source_doc, then
    add the freshly (re-)extracted edges/nodes already committed to Supabase by
    extract_and_store_graph — instead of evicting and reloading the whole graph.
    Falls back to the old full evict+reload on error, when explicitly forced via
    GRAPH_FORCE_FULL_REBUILD, or when no source_filename is given.
    """
    org_id = get_or_create_org_for_user(user_id)
    if source_filename and not _graph_force_full_rebuild():
        try:
            _incremental_update_graph_for_user(org_id, source_filename)
            return
        except Exception as e:
            logging.warning(f"[graph] incremental update failed for org {org_id}, "
                             f"falling back to full rebuild: {e}")
    with _graph_lock:
        _graph_store.pop(org_id, None)
    get_graph_for_user(user_id)
    logging.info(f"[graph] cache refreshed for org {org_id}")


def _incremental_update_graph_for_user(org_id, source_filename):
    """Patch the cached graph with source_filename's current edges/nodes.

    Held under _graph_lock for the entire remove-then-add so concurrent mutation
    isn't interleaved (NetworkX mutation across multiple add_edge calls isn't atomic).
    Does nothing if the org's graph isn't cached yet — the next lazy
    get_graph_for_user() call will build it fresh from Supabase, which already
    reflects this upload since extract_and_store_graph committed it first.
    """
    with _graph_lock:
        if org_id not in _graph_store:
            return

    edges_res = supabase_admin.table('graph_edges')\
        .select('source_entity,relation,target_entity,chunk_id,source_doc,page_num')\
        .eq('org_id', org_id).eq('source_doc', source_filename).execute()
    nodes_res = supabase_admin.table('graph_nodes')\
        .select('entity_name,entity_type,aliases')\
        .eq('org_id', org_id).execute()
    edges = edges_res.data or []
    nodes = nodes_res.data or []

    with _graph_lock:
        G = _graph_store.get(org_id)
        if G is None:
            return
        stale_edges = [(u, v) for u, v, data in G.edges(data=True)
                       if data.get('source_doc') == source_filename]
        G.remove_edges_from(stale_edges)
        for node in nodes:
            G.add_node(node['entity_name'], entity_type=node.get('entity_type'),
                       aliases=node.get('aliases') or [])
        for edge in edges:
            G.add_edge(
                edge['source_entity'], edge['target_entity'],
                relation=edge['relation'],
                chunk_id=edge['chunk_id'],
                source_doc=edge['source_doc'],
                page_num=edge.get('page_num'),
            )
    logging.info(f"[graph] incrementally updated org {org_id} "
                 f"({len(edges)} edges for {source_filename!r})")


def graph_expand_candidates(query, user_id, existing_texts):
    """Return [(0.5, text), ...] for graph-neighbor chunks not in existing_texts.

    Uses 2-hop BFS from entities found by substring match in the query.
    Score 0.5 is a placeholder — all candidates pass through the reranker together.
    """
    G = get_graph_for_user(user_id)
    if not G or G.number_of_nodes() == 0:
        return []

    query_lower = query.lower()
    matched_nodes = [
        node for node, data in G.nodes(data=True)
        if node in query_lower
        or any(alias and alias.lower() in query_lower for alias in (data.get('aliases') or []))
    ]
    if not matched_nodes:
        return []

    neighbor_chunk_ids = set()
    for node in matched_nodes:
        try:
            subgraph = nx.ego_graph(G, node, radius=2)
        except Exception:
            continue
        for n in subgraph.nodes():
            for _, _, data in G.edges(n, data=True):
                cid = data.get('chunk_id')
                if cid:
                    neighbor_chunk_ids.add(cid)

    if not neighbor_chunk_ids:
        return []

    try:
        results = qdrant.retrieve(
            collection_name=COLLECTION,
            ids=list(neighbor_chunk_ids)[: MAX_GRAPH_NEIGHBORS * 2],
            with_payload=True,
        )
    except Exception as e:
        logging.warning(f"[graph] Qdrant retrieve failed: {e}")
        return []

    expanded = []
    for r in results:
        text = r.payload.get('text', '')
        if text and text not in existing_texts:
            expanded.append((0.5, text))
        if len(expanded) >= MAX_GRAPH_NEIGHBORS:
            break

    if expanded:
        logging.info(f"[graph] expanded {len(expanded)} neighbor chunks for query")
    return expanded


def find_relevant_chunks_with_graph(query, user_id, top_n=5, top_k=20):
    """Hybrid retrieval + graph expansion, reranked as one pool."""
    if get_collection_count(user_id) == 0:
        return []

    candidates = hybrid_search(query, user_id, top_k=top_k)
    if not candidates:
        return []

    existing_texts = {text for _, text in candidates}
    graph_candidates = graph_expand_candidates(query, user_id, existing_texts)
    graph_texts = {text for _, text in graph_candidates}

    all_candidates = candidates + [
        (f'graph_{i}', text) for i, (_, text) in enumerate(graph_candidates)
    ]
    texts = [text for _, text in all_candidates]

    scores = list(get_reranker_model().rerank(query, texts))
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    result = [(_sigmoid(float(score)), texts[idx]) for idx, score in ranked[:top_n]]

    graph_in_top = sum(1 for _, text in result if text in graph_texts)
    logging.info(
        f"[eval] retrieval pool={len(texts)} graph_expanded={len(graph_texts)} "
        f"graph_in_top{top_n}={graph_in_top}"
    )
    return result


# ---- LLM ----

_SYSTEM_PROMPT = (
    'You are a helpful assistant that answers questions based on '
    'provided document excerpts. Always cite page numbers when referencing '
    'information. If the answer is not found in the provided context, '
    'say so clearly rather than guessing.'
)

_DECOMPOSE_PROMPT = (
    'Split this question into independent sub-questions, each answerable from separate '
    'document excerpts. If the question is simple, return just the original.\n'
    'Rules: max 3 sub-questions, each self-contained.\n'
    'Return ONLY a JSON array of strings, no explanation.\n\n'
    'Question: {question}'
)

_GRADE_PROMPT = (
    'Determine if each chunk is relevant to the query.\n'
    'Return ONLY JSON: {{"relevant": [0, 2], "irrelevant": [1]}}\n'
    '(numbers are 0-based chunk indices)\n\n'
    'Query: {query}\nChunks:\n{chunks}'
)

_REFORMULATE_PROMPT = (
    'Rewrite this query using different terminology that might appear in technical documents. '
    'The original failed to retrieve relevant content.\n\n'
    'Original: {query}\n'
    'Rewritten query (return ONLY the query text):'
)

_GRAPH_EXTRACT_PROMPT = (
    'Extract entities and relationships from each document chunk below.\n\n'
    'Entity types: PERSON, ORG, CONCEPT, LOCATION, DATE, METRIC\n'
    'Relations — use ONLY these: acquired_by, owns, reports_to, caused_by, '
    'has_obligation, depends_on, related_to, part_of, occurred_at, '
    'measured_by, approved_by, responsible_for, defines, mentioned_with\n\n'
    'Rules:\n'
    '- Normalize entity names to canonical lowercase\n'
    '- If an entity has alternate names/abbreviations in the text (e.g. "MSFT" for '
    '"Microsoft"), list them lowercase under "aliases" (omit if none)\n'
    '- Max 5 entities and 8 triples per chunk\n'
    '- Only include triples where both subject and object are in your entities list\n'
    '- Return ONLY a JSON array, one object per chunk, in input order\n\n'
    'Format: [{{"chunk_index":0,"entities":[{{"name":"...","type":"...","aliases":["..."]}}],'
    '"triples":[{{"subject":"...","relation":"...","object":"..."}}]}}, ...]\n\n'
    'Chunks:\n{chunks}'
)


def _parse_llm_json(content):
    """Strip markdown code fences then parse JSON. Raises ValueError on failure."""
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    m = re.search(r'```(?:json)?\s*([\s\S]*?)```', content)
    if m:
        return json.loads(m.group(1).strip())
    m = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', content)
    if m:
        return json.loads(m.group(1))
    raise ValueError(f"No JSON found in LLM output: {content[:200]}")


def _call_openrouter_helper(user_content, max_tokens, temperature=0.0, model=GRAPH_MODEL):
    """Single-turn OpenRouter call. Falls back to Groq/Gemini if unconfigured or failing."""
    if openrouter_client is not None:
        try:
            resp = openrouter_client.chat.completions.create(
                model=model,
                messages=[{'role': 'user', 'content': user_content}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logging.warning(f"OpenRouter helper failed ({e}), falling back to Groq/Gemini")
    return _call_groq_helper(user_content, max_tokens, temperature=temperature)


def _call_groq_helper(user_content, max_tokens, temperature=0.0, model=None):
    """Single-turn Groq call with Gemini fallback. Returns raw text or raises."""
    try:
        resp = groq_client.chat.completions.create(
            model=model or GROQ_MODEL,
            messages=[{'role': 'user', 'content': user_content}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logging.warning(f"Groq helper failed ({e}), trying Gemini")
    if not _gemini_client:
        raise RuntimeError("Groq failed and no GEMINI_API_KEY configured")
    resp = _gemini_client.models.generate_content(model=GEMINI_MODEL, contents=user_content)
    return resp.text.strip()


def decompose_query(question):
    """Split complex question into sub-queries. Returns list of 1–3 strings.

    Skips decomposition for short questions (<= 10 words) to save a Groq call.
    """
    if len(question.split()) <= 10:
        return [question]
    try:
        raw = _call_groq_helper(_DECOMPOSE_PROMPT.format(question=question), max_tokens=200)
        sub_queries = _parse_llm_json(raw)
        if isinstance(sub_queries, list) and all(isinstance(q, str) for q in sub_queries):
            result = [q.strip() for q in sub_queries[:3] if q.strip()]
            return result or [question]
    except Exception as e:
        logging.warning(f"Query decomposition failed: {e}")
    return [question]


def grade_chunks(query, chunks):
    """Grade chunks for relevance. Returns (relevant_texts, irrelevant_texts).

    Raises GradingUnavailableError if the grading call itself fails, so callers
    can tell "grader is down" apart from "grader ran and found nothing relevant"
    instead of silently treating every chunk as relevant either way.
    """
    if not chunks:
        return [], []
    formatted = '\n'.join(f'[{i}] {text[:300]}' for i, text in enumerate(chunks))
    try:
        raw = _call_groq_helper(_GRADE_PROMPT.format(query=query, chunks=formatted), max_tokens=150)
        grades = _parse_llm_json(raw)
        relevant_indices = {int(i) for i in grades.get('relevant', [])}
        relevant = [chunks[i] for i in relevant_indices if i < len(chunks)]
        irrelevant = [chunks[i] for i in range(len(chunks)) if i not in relevant_indices]
        precision = len(relevant) / len(chunks) if chunks else 0
        logging.info(f"[eval] crag_precision={precision:.2f} relevant={len(relevant)}/{len(chunks)} query={query[:60]!r}")
        return relevant, irrelevant
    except Exception as e:
        logging.warning(f"Chunk grading failed: {e}")
        raise GradingUnavailableError(str(e)) from e


class GradingUnavailableError(Exception):
    """Raised when the grading LLM call itself failed — distinct from a genuine
    zero-relevant verdict. Callers should stop retrying rather than burn iteration
    budget against a grader that isn't working."""


MAX_CRAG_ITERATIONS = int(os.getenv('MAX_CRAG_ITERATIONS', '3'))
CRAG_WALL_CLOCK_BUDGET_S = float(os.getenv('CRAG_WALL_CLOCK_BUDGET_S', '12'))


def reformulate_query(query):
    """Rewrite query for better retrieval. Returns rewritten string."""
    try:
        return _call_groq_helper(_REFORMULATE_PROMPT.format(query=query), max_tokens=100, temperature=0.1)
    except Exception as e:
        logging.warning(f"Query reformulation failed: {e}")
        return query


def generate_text(prompt, conversation_history=None):
    messages = [{'role': 'system', 'content': _SYSTEM_PROMPT}]
    for turn in (conversation_history or []):
        messages.append({'role': 'user', 'content': turn['question']})
        messages.append({'role': 'assistant', 'content': turn['answer']})
    messages.append({'role': 'user', 'content': prompt})

    if openrouter_client is not None:
        try:
            response = openrouter_client.chat.completions.create(
                model=ANSWER_MODEL,
                messages=messages,
                max_tokens=750,
                temperature=0.2
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.warning(f"[eval] openrouter_fallback=true reason={e}")

    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            max_tokens=750,
            temperature=0.2
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.warning(f"[eval] groq_fallback=true reason={e}")

    if not _gemini_client:
        logging.error("Groq failed and no GEMINI_API_KEY configured")
        return None
    try:
        history = []
        for turn in (conversation_history or []):
            history.append(genai_types.Content(role='user', parts=[genai_types.Part(text=turn['question'])]))
            history.append(genai_types.Content(role='model', parts=[genai_types.Part(text=turn['answer'])]))
        chat = _gemini_client.chats.create(
            model=GEMINI_MODEL,
            history=history,
            config=genai_types.GenerateContentConfig(system_instruction=_SYSTEM_PROMPT),
        )
        response = chat.send_message(prompt)
        return response.text.strip()
    except Exception as e:
        logging.error(f"Gemini fallback also failed: {e}", exc_info=True)
        return None


_SOURCE_RE = re.compile(r'^\[Page (\d+), Source: ([^\]]+)\]\s*(.*)', re.DOTALL)
_FIGURE_MARKER_RE = re.compile(r'\[Figure: ([^\]]+)\]\s*')


def build_source(score, text):
    """Parse a display chunk into a source dict. Returns (source, prompt_text).

    Figure chunks carry their storage path in a [Figure: path] marker; it is
    surfaced as image_path and stripped from the text sent to the LLM.
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
    return source, clean


def ask_file(question, user_id, conversation_history=None):
    """Return (response_text, sources) for a specific user's documents."""
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
        source, prompt_text = build_source(score, text)
        prompt += f"{prompt_text}\n\n"
        sources.append(source)

    prompt += f"Question: {question}\nAnswer:"
    response = generate_text(prompt, conversation_history=conversation_history)
    if response is None:
        return None, []
    return response, sources


def ask_file_agentic(question, user_id, conversation_history=None):
    """Agentic RAG: query decomp + CRAG loop + synthesis. Falls back to ask_file() on error."""
    t_total = time.perf_counter()
    try:
        if get_collection_count(user_id) == 0:
            return "No documents have been indexed yet. Please upload a PDF first.", []

        sub_queries = _timed("decompose_query", decompose_query, question)
        logging.info(f"[agentic] decomposed into {len(sub_queries)} sub-queries: {sub_queries}")
        complexity = estimate_query_complexity(question, sub_query_count=len(sub_queries))

        # Anchor retrieval to prior turn so follow-ups ("what about its revenue?")
        # retrieve on full context, not just the bare pronoun.
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
                try:
                    relevant, _ = _timed(f"grade_chunks_iter{iteration}", grade_chunks, sub_q, texts)
                except GradingUnavailableError:
                    logging.warning(f"[agentic] grading unavailable, using retrieved chunks as-is for '{sub_q}'")
                    relevant = texts
                    break

                if relevant:
                    break

                logging.info(f"[agentic] no relevant chunks for '{retrieval_query}' (iter {iteration}), reformulating")
                retrieval_query = _timed("reformulate_query", reformulate_query, retrieval_query)
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
            source, prompt_text = build_source(score, text)
            prompt += f"{prompt_text}\n\n"
            sources.append(source)

        prompt += f"Question: {question}\nAnswer:"
        response = _timed("generate_text", generate_text, prompt, conversation_history=conversation_history)
        if response is None:
            return None, []

        logging.info(f"[perf] ask_file_agentic total: {(time.perf_counter() - t_total) * 1000:.1f}ms")
        return response, sources

    except Exception as e:
        logging.error(f"[agentic] pipeline error, falling back to ask_file: {e}", exc_info=True)
        return ask_file(question, user_id, conversation_history=conversation_history)


# ---- Semantic cache ----

def _cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


_kb_version_cache = {}
_kb_version_lock = threading.Lock()


def get_kb_version(user_id):
    """Org-shared version counter — any member's upload invalidates the whole
    org's cached answers, since the retrieval corpus behind them is shared."""
    org_id = get_or_create_org_for_user(user_id)
    with _kb_version_lock:
        if org_id in _kb_version_cache:
            return _kb_version_cache[org_id]
    version = 0
    if redis_client:
        try:
            raw = redis_client.get(f"kbversion:{org_id}")
            if raw is not None:
                version = int(raw)
        except Exception:
            version = 0
    with _kb_version_lock:
        _kb_version_cache[org_id] = version
    return version


def bump_kb_version(user_id):
    org_id = get_or_create_org_for_user(user_id)
    current = get_kb_version(user_id)
    new_version = current + 1
    with _kb_version_lock:
        _kb_version_cache[org_id] = new_version
    if redis_client:
        try:
            redis_client.set(f"kbversion:{org_id}", new_version)
        except Exception:
            pass
    return new_version


# Global-bounded hot subset (ponytail: flat OrderedDict as manual LRU, not per-user,
# so memory stays capped regardless of user count — upgrade to a real LRU cache lib
# only if profiling shows this is a bottleneck). Redis is the source of truth (off-box).
SEMANTIC_CACHE_THRESHOLD = float(os.getenv('SEMANTIC_CACHE_THRESHOLD', '0.93'))
SEMANTIC_CACHE_L1_MAX = int(os.getenv('SEMANTIC_CACHE_L1_MAX', '500'))
SEMANTIC_CACHE_PER_USER_MAX = int(os.getenv('SEMANTIC_CACHE_PER_USER_MAX', '50'))

_semantic_l1 = OrderedDict()
_semantic_l1_lock = threading.Lock()
_semantic_entry_ids = itertools.count()


def _semantic_redis_key(org_id):
    return f"semcache:{org_id}"


def _semantic_l1_put(org_id, entry):
    key = (org_id, next(_semantic_entry_ids))
    with _semantic_l1_lock:
        _semantic_l1[key] = entry
        _semantic_l1.move_to_end(key)
        while len(_semantic_l1) > SEMANTIC_CACHE_L1_MAX:
            _semantic_l1.popitem(last=False)


def _semantic_l1_user_entries(org_id):
    with _semantic_l1_lock:
        return [entry for (oid, _), entry in _semantic_l1.items() if oid == org_id]


def semantic_cache_store(user_id, question, query_vec, response, sources, model_version, kb_version):
    """Org-scoped so a member sees a teammate's cached answer to the same question."""
    org_id = get_or_create_org_for_user(user_id)
    entry = {
        'question': question,
        'query_vec': [float(x) for x in query_vec],
        'response': response,
        'sources': sources,
        'model_version': model_version,
        'kb_version': kb_version,
    }
    _semantic_l1_put(org_id, entry)
    if not redis_client:
        return
    try:
        key = _semantic_redis_key(org_id)
        raw = redis_client.get(key)
        entries = json.loads(raw) if raw else []
        entries.append(entry)
        if len(entries) > SEMANTIC_CACHE_PER_USER_MAX:
            entries = entries[-SEMANTIC_CACHE_PER_USER_MAX:]
        redis_client.set(key, json.dumps(entries))
    except Exception:
        pass


def semantic_cache_lookup(user_id, query_vec, model_version, kb_version):
    """Best cosine match for this user's org's cached entries above SEMANTIC_CACHE_THRESHOLD.

    Metadata (model_version, kb_version) must match exactly — a high similarity score
    alone is never enough, since a reindex or model change can invalidate old answers.
    """
    org_id = get_or_create_org_for_user(user_id)
    candidates = _semantic_l1_user_entries(org_id)
    if not candidates and redis_client:
        try:
            raw = redis_client.get(_semantic_redis_key(org_id))
            candidates = json.loads(raw) if raw else []
        except Exception:
            candidates = []

    best_entry = None
    best_score = -1.0
    for entry in candidates:
        if entry.get('model_version') != model_version or entry.get('kb_version') != kb_version:
            continue
        score = _cosine_similarity(query_vec, entry['query_vec'])
        if score > best_score:
            best_score = score
            best_entry = entry

    if best_entry is not None and best_score >= SEMANTIC_CACHE_THRESHOLD:
        logging.info(f"[semantic-cache] hit score={best_score:.3f}")
        return best_entry['response'], best_entry['sources']
    if best_entry is not None:
        logging.info(f"[semantic-cache] near-miss score={best_score:.3f} threshold={SEMANTIC_CACHE_THRESHOLD}")
    return None, None


# ---- Cache ----

def get_cached_response(cache_key):
    """Check L1 (in-memory) then L2 (Upstash Redis). Returns (response, sources) or (None, None)."""
    with _memory_cache_lock:
        hit = _memory_cache.get(cache_key)
    if hit is not None:
        logging.info("[cache] L1 hit")
        return hit
    if not redis_client:
        logging.info("[cache] miss (no redis)")
        return None, None
    try:
        cached = redis_client.get(cache_key)
        if not cached:
            logging.info("[cache] miss")
            return None, None
        data = json.loads(cached)
        result = (data.get('response'), data.get('sources', []))
        with _memory_cache_lock:
            _memory_cache[cache_key] = result
        logging.info("[cache] L2 hit")
        return result
    except Exception:
        return None, None


def cache_response(cache_key, response, sources, ttl=3600):
    with _memory_cache_lock:
        _memory_cache[cache_key] = (response, sources)
    if not redis_client:
        return
    try:
        redis_client.setex(cache_key, ttl, json.dumps({'response': response, 'sources': sources}))
    except Exception:
        pass


# ---- Startup ----

threading.Thread(target=init_index, daemon=True).start()


# ---- Routes ----

@app.route('/')
def index():
    return render_template(
        'index.html',
        supabase_url=os.getenv('SUPABASE_URL', ''),
        supabase_anon_key=os.getenv('SUPABASE_ANON_KEY', '')
    )


def _trigger_ingest_workflow(storage_path, user_id, filename, display_name):
    """Fire a repository_dispatch so a GitHub Actions runner ingests the PDF."""
    resp = requests.post(
        f'https://api.github.com/repos/{GITHUB_REPO}/dispatches',
        headers={
            'Authorization': f'Bearer {GITHUB_DISPATCH_TOKEN}',
            'Accept': 'application/vnd.github+json',
        },
        json={
            'event_type': 'ingest_pdf',
            'client_payload': {
                'storage_path': storage_path,
                'user_id': user_id,
                'filename': filename,
                'display_name': display_name,
            },
        },
        timeout=15,
    )
    resp.raise_for_status()


@app.route('/upload', methods=['POST'])
@require_auth
def upload_pdf():
    if 'pdf' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['pdf']
    if not file.filename or not file.filename.lower().endswith('.pdf'):
        return jsonify({'error': 'Please upload a valid PDF file'}), 400

    original_name = file.filename
    filename = secure_filename(file.filename)
    user_id = g.user_id
    storage_path = f"{user_id}/{filename}"

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    try:
        file.save(tmp.name)
        with open(tmp.name, 'rb') as f:
            file_bytes = f.read()

        # Upload to Supabase Storage (remove old version first if present)
        try:
            supabase_admin.storage.from_('pdfs').remove([storage_path])
        except Exception:
            pass
        supabase_admin.storage.from_('pdfs').upload(
            path=storage_path,
            file=file_bytes,
            file_options={'content-type': 'application/pdf'}
        )

        supabase_admin.table('documents').delete().eq('user_id', user_id).eq('filename', filename).execute()

        if GITHUB_DISPATCH_TOKEN and GITHUB_REPO:
            supabase_admin.table('documents').insert({
                'user_id': user_id,
                'filename': filename,
                'storage_path': storage_path,
                'chunk_count': 0,
                'status': 'processing'
            }).execute()
            try:
                _trigger_ingest_workflow(storage_path, user_id, filename, original_name)
            except Exception as e:
                logging.error(f"Ingest dispatch failed for {storage_path}: {e}", exc_info=True)
                (supabase_admin.table('documents').update({'status': 'failed'})
                 .eq('user_id', user_id).eq('filename', filename).execute())
                return jsonify({'error': 'Could not queue indexing, please try again'}), 502
            return jsonify({'message': f'{filename} queued for indexing', 'status': 'processing'}), 202

        # ponytail: no dispatch config (local dev) -> index inline, as before
        count = index_pdf(tmp.name, user_id, force=True, display_name=original_name)
        rebuild_bm25_for_user(user_id, source_filename=original_name)
        rebuild_graph_for_user(user_id, source_filename=original_name)
        bump_kb_version(user_id)
        check_qdrant_shard_threshold()

        supabase_admin.table('documents').insert({
            'user_id': user_id,
            'filename': filename,
            'storage_path': storage_path,
            'chunk_count': count,
            'status': 'ready'
        }).execute()

        return jsonify({'message': f'Successfully indexed {count} chunks from {filename}'})
    except Exception as e:
        logging.error(f"Upload error: {e}", exc_info=True)
        return jsonify({'error': 'Upload failed, please try again'}), 500
    finally:
        os.unlink(tmp.name)


@app.route('/documents', methods=['GET'])
@require_auth
def list_documents():
    try:
        res = (
            supabase_admin.table('documents')
            .select('filename,chunk_count,status')
            .eq('user_id', g.user_id)
            .order('created_at')
            .execute()
        )
        documents = [
            {
                'filename': row['filename'],
                'chunks': row['chunk_count'],
                'status': row.get('status') or 'ready',
            }
            for row in res.data
        ]
        return jsonify({'documents': documents})
    except Exception as e:
        logging.error(f"Error in /documents: {e}")
        return jsonify({'documents': []}), 500


@app.route('/figure-url', methods=['GET'])
@require_auth
def figure_url():
    """Mint a short-lived signed URL for a figure image owned by the caller."""
    path = request.args.get('path', '')
    if '..' in path or not path.startswith(f"{g.user_id}/figures/"):
        return jsonify({'error': 'Not found'}), 404
    try:
        res = supabase_admin.storage.from_('pdfs').create_signed_url(path, 3600)
        url = res.get('signedURL') or res.get('signedUrl') or res.get('signed_url')
        if not url:
            raise ValueError(f"No signed URL in storage response: {res}")
        return jsonify({'url': url})
    except Exception as e:
        logging.error(f"Error in /figure-url for {path}: {e}")
        return jsonify({'error': 'Could not generate figure URL'}), 500


@app.route('/history', methods=['GET'])
@require_auth
def history():
    try:
        res = (
            supabase_admin.table('query_history')
            .select('id,question,answer,sources,created_at,session_id')
            .eq('user_id', g.user_id)
            .order('created_at', desc=False)
            .limit(200)
            .execute()
        )
        # Group by session_id; legacy rows (no session_id) each become their own session
        sessions_map = {}
        session_order = []
        for row in res.data:
            sid = row.get('session_id') or row['id']
            if sid not in sessions_map:
                sessions_map[sid] = {'session_id': sid, 'started_at': row['created_at'], 'questions': []}
                session_order.append(sid)
            sessions_map[sid]['questions'].append(row)
        # Most recent session first
        sessions = [sessions_map[sid] for sid in reversed(session_order)]
        return jsonify({'sessions': sessions})
    except Exception as e:
        logging.error(f"Error in /history: {e}")
        return jsonify({'sessions': []}), 500


@app.route('/health')
def health():
    qdrant_ok = False
    try:
        qdrant.get_collections()
        qdrant_ok = True
    except Exception as e:
        logging.warning(f"Qdrant health check failed: {e}")
    status = {
        'status': 'ok' if (qdrant_ok and _bm25_ready) else 'starting',
        'qdrant_ok': qdrant_ok,
        'bm25_ready': _bm25_ready,
    }
    return jsonify(status), 200


@app.route('/ask', methods=['POST'])
@require_auth
def ask():
    try:
        question = request.form.get('question', '').strip()
        session_id = request.form.get('session_id', '').strip() or None
        if not question:
            return jsonify({'error': 'No question provided'}), 400

        user_id = g.user_id

        # Fetch prior turns for this session (for multi-turn context)
        conversation_history = []
        if session_id:
            try:
                prior = (
                    supabase_admin.table('query_history')
                    .select('question,answer')
                    .eq('user_id', user_id)
                    .eq('session_id', session_id)
                    .order('created_at', desc=False)
                    .limit(5)
                    .execute()
                )
                conversation_history = prior.data or []
            except Exception as e:
                logging.warning(f"Failed to fetch session history: {e}")

        # Only use Redis cache for single-turn (no prior context). Org-scoped so
        # an org member's identical question hits a teammate's cached answer.
        cache_key = f"{get_or_create_org_for_user(user_id)}:{question}" if not conversation_history else None
        response, sources = get_cached_response(cache_key) if cache_key else (None, None)
        from_cache = response is not None

        query_vec = None
        kb_version = None
        if response is None and cache_key:
            query_vec = list(get_embedding_model().embed([question]))[0].tolist()
            kb_version = get_kb_version(user_id)
            response, sources = semantic_cache_lookup(user_id, query_vec, GEMINI_MODEL, kb_version)
            from_cache = response is not None

        if response is None:
            response, sources = ask_file_agentic(question, user_id, conversation_history=conversation_history)

        if response is not None and 'No documents' not in response:
            if not from_cache and cache_key:
                cache_response(cache_key, response, sources)
                if query_vec is None:
                    query_vec = list(get_embedding_model().embed([question]))[0].tolist()
                    kb_version = get_kb_version(user_id)
                semantic_cache_store(user_id, question, query_vec, response, sources, GEMINI_MODEL, kb_version)
            # Always save to history when a session is active so subsequent
            # turns can fetch this turn as context.
            if session_id or not from_cache:
                try:
                    supabase_admin.table('query_history').insert({
                        'user_id': user_id,
                        'question': question,
                        'answer': response,
                        'sources': sources,
                        'session_id': session_id,
                    }).execute()
                except Exception as e:
                    logging.warning(f"Failed to save query history: {e}")

        if response is None:
            return jsonify({'error': 'Failed to generate a response, please try again'}), 500

        return jsonify({'response': response, 'sources': sources})

    except Exception as e:
        logging.error(f"Error in /ask: {e}", exc_info=True)
        return jsonify({'error': 'An error occurred, please try again'})


if __name__ == '__main__':
    app.run(debug=False)
