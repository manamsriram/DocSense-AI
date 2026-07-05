import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from functools import wraps
from unittest.mock import patch, MagicMock


# ---- Helpers ----

def _make_auth_decorator(user_id='test-user-id'):
    """require_auth stand-in that injects g.user_id without Supabase."""
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            from flask import g
            g.user_id = user_id
            return f(*args, **kwargs)
        return wrapper
    return decorator


def _supabase_chain(data=None):
    """MagicMock satisfying chained Supabase builder calls ending in .execute()."""
    mock = MagicMock()
    mock.table.return_value = mock
    mock.select.return_value = mock
    mock.eq.return_value = mock
    mock.order.return_value = mock
    mock.limit.return_value = mock
    mock.insert.return_value = mock
    mock.delete.return_value = mock
    mock.execute.return_value = MagicMock(data=data or [])
    return mock


# ---- Semantic cache: cosine similarity ----

def test_cosine_similarity_identical_vectors_is_one():
    from app import _cosine_similarity
    assert abs(_cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) - 1.0) < 1e-6


def test_cosine_similarity_orthogonal_vectors_is_zero():
    from app import _cosine_similarity
    assert abs(_cosine_similarity([1.0, 0.0], [0.0, 1.0])) < 1e-6


def test_cosine_similarity_opposite_vectors_is_negative_one():
    from app import _cosine_similarity
    assert abs(_cosine_similarity([1.0, 0.0], [-1.0, 0.0]) - (-1.0)) < 1e-6


# ---- Semantic cache: kb_version ----

def test_get_kb_version_defaults_to_zero_for_new_user():
    from app import get_kb_version
    with patch('app.redis_client', None):
        assert get_kb_version('kbv-user-fresh-1') == 0


def test_bump_kb_version_increments_and_persists_in_process():
    from app import get_kb_version, bump_kb_version
    user_id = 'kbv-user-bump-1'
    with patch('app.redis_client', None):
        assert get_kb_version(user_id) == 0
        new_version = bump_kb_version(user_id)
        assert new_version == 1
        assert get_kb_version(user_id) == 1
        assert bump_kb_version(user_id) == 2


def test_bump_kb_version_writes_to_redis_when_available():
    from app import bump_kb_version
    with patch('app.redis_client') as mock_redis:
        mock_redis.get.return_value = None
        bump_kb_version('kbv-user-redis-2')
        mock_redis.set.assert_called_once_with('kbversion:kbv-user-redis-2', 1)


# ---- Semantic cache: store/lookup ----

def test_semantic_cache_lookup_miss_when_empty():
    from app import semantic_cache_lookup
    with patch('app.redis_client', None):
        response, sources = semantic_cache_lookup('semc-user-1', [1.0, 0.0, 0.0], 'model-a', 0)
        assert response is None
        assert sources is None


def test_semantic_cache_hit_for_similar_vector():
    from app import semantic_cache_store, semantic_cache_lookup
    with patch('app.redis_client', None):
        semantic_cache_store('semc-user-2', 'How do I reset my password?', [1.0, 0.0, 0.0],
                              'Reset via settings.', [{'source': 'faq.pdf'}], 'model-a', 0)
        response, sources = semantic_cache_lookup('semc-user-2', [0.99, 0.01, 0.0], 'model-a', 0)
        assert response == 'Reset via settings.'
        assert sources == [{'source': 'faq.pdf'}]


def test_semantic_cache_miss_below_threshold():
    from app import semantic_cache_store, semantic_cache_lookup
    with patch('app.redis_client', None):
        semantic_cache_store('semc-user-3', 'How do I reset my password?', [1.0, 0.0, 0.0],
                              'Reset via settings.', [], 'model-a', 0)
        response, sources = semantic_cache_lookup('semc-user-3', [0.0, 1.0, 0.0], 'model-a', 0)
        assert response is None
        assert sources is None


def test_semantic_cache_miss_on_kb_version_mismatch():
    from app import semantic_cache_store, semantic_cache_lookup
    with patch('app.redis_client', None):
        semantic_cache_store('semc-user-4', 'How do I reset my password?', [1.0, 0.0, 0.0],
                              'Reset via settings.', [], 'model-a', 0)
        response, sources = semantic_cache_lookup('semc-user-4', [1.0, 0.0, 0.0], 'model-a', 1)
        assert response is None
        assert sources is None


def test_semantic_cache_miss_on_model_version_mismatch():
    from app import semantic_cache_store, semantic_cache_lookup
    with patch('app.redis_client', None):
        semantic_cache_store('semc-user-5', 'How do I reset my password?', [1.0, 0.0, 0.0],
                              'Reset via settings.', [], 'model-a', 0)
        response, sources = semantic_cache_lookup('semc-user-5', [1.0, 0.0, 0.0], 'model-b', 0)
        assert response is None
        assert sources is None


def test_semantic_cache_isolated_per_user():
    from app import semantic_cache_store, semantic_cache_lookup
    with patch('app.redis_client', None):
        semantic_cache_store('semc-user-6a', 'How do I reset my password?', [1.0, 0.0, 0.0],
                              'Reset via settings.', [], 'model-a', 0)
        response, sources = semantic_cache_lookup('semc-user-6b', [1.0, 0.0, 0.0], 'model-a', 0)
        assert response is None
        assert sources is None


# ---- Sigmoid tests ----

def test_sigmoid_midpoint():
    from app import _sigmoid
    assert abs(_sigmoid(0.0) - 0.5) < 1e-6


def test_sigmoid_positive():
    from app import _sigmoid
    assert _sigmoid(2.0) > 0.5
    assert _sigmoid(2.0) < 1.0


def test_sigmoid_negative():
    from app import _sigmoid
    assert _sigmoid(-2.0) < 0.5
    assert _sigmoid(-2.0) > 0.0


# ---- /ask tests ----

def test_ask_returns_response_and_sources():
    """Single-turn ask returns response text and parsed sources."""
    chunks = [
        (0.94, '[Page 4, Source: test.pdf] Some text here'),
        (0.81, '[Page 12, Source: other.pdf] More text'),
    ]
    with patch('app.require_auth', _make_auth_decorator()), \
         patch('app.supabase_admin', _supabase_chain()), \
         patch('app.get_collection_count', return_value=1), \
         patch('app.decompose_query', return_value=['test question']), \
         patch('app.grade_chunks', return_value=([c[1] for c in chunks], [])), \
         patch('app.find_relevant_chunks_with_graph', return_value=chunks), \
         patch('app.generate_text', return_value='Answer text'), \
         patch('app.get_cached_response', return_value=(None, None)), \
         patch('app.cache_response'):

        import app as flask_app
        flask_app.app.config['TESTING'] = True
        client = flask_app.app.test_client()
        res = client.post('/ask', data={'question': 'test question', 'session_id': ''})
        data = res.get_json()

        assert res.status_code == 200
        assert data['response'] == 'Answer text'
        assert isinstance(data['sources'], list)
        assert len(data['sources']) == 2
        assert data['sources'][0]['source'] == 'test.pdf'
        assert data['sources'][0]['page'] == 4


def test_ask_cached_response_returned_directly():
    """Cache hit returns cached answer without calling LLM."""
    with patch('app.require_auth', _make_auth_decorator()), \
         patch('app.supabase_admin', _supabase_chain()), \
         patch('app.get_cached_response', return_value=('Cached answer', [])), \
         patch('app.generate_text') as mock_gen:

        import app as flask_app
        flask_app.app.config['TESTING'] = True
        client = flask_app.app.test_client()
        res = client.post('/ask', data={'question': 'test question'})
        data = res.get_json()

        assert res.status_code == 200
        assert data['response'] == 'Cached answer'
        assert data['sources'] == []
        mock_gen.assert_not_called()


def _fake_embedding_model(vec=(1.0, 0.0, 0.0)):
    import numpy as np
    model = MagicMock()
    model.embed.return_value = [np.array(vec)]
    return model


def test_ask_semantic_cache_hit_skips_llm():
    """Semantic cache hit (after exact-match miss) returns cached answer, skips LLM."""
    with patch('app.require_auth', _make_auth_decorator()), \
         patch('app.supabase_admin', _supabase_chain()), \
         patch('app.get_cached_response', return_value=(None, None)), \
         patch('app.get_embedding_model', return_value=_fake_embedding_model()), \
         patch('app.get_kb_version', return_value=0), \
         patch('app.semantic_cache_lookup', return_value=('Semantic answer', [{'source': 'faq.pdf'}])), \
         patch('app.semantic_cache_store') as mock_store, \
         patch('app.generate_text') as mock_gen:

        import app as flask_app
        flask_app.app.config['TESTING'] = True
        client = flask_app.app.test_client()
        res = client.post('/ask', data={'question': 'How do I reset my password?'})
        data = res.get_json()

        assert res.status_code == 200
        assert data['response'] == 'Semantic answer'
        assert data['sources'] == [{'source': 'faq.pdf'}]
        mock_gen.assert_not_called()
        mock_store.assert_not_called()


def test_ask_semantic_cache_miss_stores_new_entry():
    """Semantic + exact cache miss: LLM runs, then semantic_cache_store is called."""
    chunk = (0.9, '[Page 1, Source: doc.pdf] Some context')
    with patch('app.require_auth', _make_auth_decorator()), \
         patch('app.supabase_admin', _supabase_chain()), \
         patch('app.get_collection_count', return_value=1), \
         patch('app.decompose_query', return_value=['test question']), \
         patch('app.grade_chunks', return_value=([chunk[1]], [])), \
         patch('app.find_relevant_chunks_with_graph', return_value=[chunk]), \
         patch('app.generate_text', return_value='Fresh answer'), \
         patch('app.get_cached_response', return_value=(None, None)), \
         patch('app.cache_response'), \
         patch('app.get_embedding_model', return_value=_fake_embedding_model()), \
         patch('app.get_kb_version', return_value=0), \
         patch('app.semantic_cache_lookup', return_value=(None, None)), \
         patch('app.semantic_cache_store') as mock_store:

        import app as flask_app
        flask_app.app.config['TESTING'] = True
        client = flask_app.app.test_client()
        res = client.post('/ask', data={'question': 'test question'})
        data = res.get_json()

        assert res.status_code == 200
        assert data['response'] == 'Fresh answer'
        mock_store.assert_called_once()
        call_args = mock_store.call_args.args
        assert call_args[0] == 'test-user-id'
        assert call_args[1] == 'test question'
        assert call_args[3] == 'Fresh answer'


def test_ask_multi_turn_skips_semantic_cache():
    """Conversation history present: semantic cache is never consulted (single-turn only)."""
    prior_turns = [{'question': 'What is X?', 'answer': 'X is a thing.'}]
    chunk = (0.9, '[Page 1, Source: doc.pdf] Some context')
    with patch('app.require_auth', _make_auth_decorator()), \
         patch('app.supabase_admin', _supabase_chain(data=prior_turns)), \
         patch('app.get_collection_count', return_value=1), \
         patch('app.decompose_query', return_value=['Can you elaborate?']), \
         patch('app.grade_chunks', return_value=([chunk[1]], [])), \
         patch('app.find_relevant_chunks_with_graph', return_value=[chunk]), \
         patch('app.generate_text', return_value='Follow-up answer'), \
         patch('app.get_cached_response', return_value=(None, None)), \
         patch('app.cache_response'), \
         patch('app.semantic_cache_lookup') as mock_lookup, \
         patch('app.semantic_cache_store') as mock_store:

        import app as flask_app
        flask_app.app.config['TESTING'] = True
        client = flask_app.app.test_client()
        res = client.post('/ask', data={
            'question': 'Can you elaborate?',
            'session_id': 'session-abc-123',
        })

        assert res.status_code == 200
        mock_lookup.assert_not_called()
        mock_store.assert_not_called()


def test_ask_missing_question_returns_400():
    """Blank question returns 400."""
    with patch('app.require_auth', _make_auth_decorator()):
        import app as flask_app
        flask_app.app.config['TESTING'] = True
        client = flask_app.app.test_client()
        res = client.post('/ask', data={'question': '   '})
        assert res.status_code == 400
        assert 'error' in res.get_json()


def test_ask_no_auth_returns_401():
    """No auth header → 401."""
    import app as flask_app
    flask_app.app.config['TESTING'] = True
    client = flask_app.app.test_client()
    res = client.post('/ask', data={'question': 'hello'})
    assert res.status_code == 401


# ---- /upload kb_version invalidation ----

def test_upload_bumps_kb_version():
    """A successful upload/reindex bumps kb_version so stale semantic-cache entries stop matching."""
    from io import BytesIO
    with patch('app.require_auth', _make_auth_decorator()), \
         patch('app.supabase_admin', _supabase_chain()), \
         patch('app.index_pdf', return_value=1), \
         patch('app.rebuild_bm25_for_user'), \
         patch('app.rebuild_graph_for_user'), \
         patch('app.bump_kb_version') as mock_bump:

        import app as flask_app
        flask_app.app.config['TESTING'] = True
        client = flask_app.app.test_client()
        res = client.post('/upload', data={'pdf': (BytesIO(b'%PDF-1.4 fake'), 'test.pdf')})

        assert res.status_code == 200
        mock_bump.assert_called_once_with('test-user-id')


def test_ask_multi_turn_passes_history_to_llm():
    """Prior session turns are fetched and forwarded to generate_text."""
    prior_turns = [{'question': 'What is X?', 'answer': 'X is a thing.'}]
    captured = {}

    def fake_generate_text(prompt, conversation_history=None):
        captured['history'] = conversation_history
        return 'Follow-up answer'

    chunk = (0.9, '[Page 1, Source: doc.pdf] Some context')
    with patch('app.require_auth', _make_auth_decorator()), \
         patch('app.supabase_admin', _supabase_chain(data=prior_turns)), \
         patch('app.get_collection_count', return_value=1), \
         patch('app.decompose_query', return_value=['Can you elaborate?']), \
         patch('app.grade_chunks', return_value=([chunk[1]], [])), \
         patch('app.find_relevant_chunks_with_graph', return_value=[chunk]), \
         patch('app.generate_text', side_effect=fake_generate_text), \
         patch('app.get_cached_response', return_value=(None, None)), \
         patch('app.cache_response'):

        import app as flask_app
        flask_app.app.config['TESTING'] = True
        client = flask_app.app.test_client()
        res = client.post('/ask', data={
            'question': 'Can you elaborate?',
            'session_id': 'session-abc-123',
        })

        assert res.status_code == 200
        assert captured.get('history') == prior_turns


def test_ask_saves_session_id_to_history():
    """New response is inserted to query_history with the session_id."""
    inserted = {}

    supabase_mock = _supabase_chain(data=[])

    original_table = supabase_mock.table.side_effect

    def capturing_table(name):
        m = MagicMock()
        m.select.return_value = m
        m.eq.return_value = m
        m.order.return_value = m
        m.limit.return_value = m
        m.execute.return_value = MagicMock(data=[])
        def capturing_insert(record):
            if name == 'query_history':
                inserted.update(record)
            r = MagicMock()
            r.execute.return_value = MagicMock()
            return r
        m.insert = capturing_insert
        return m

    supabase_mock.table = capturing_table

    chunk = (0.9, '[Page 1, Source: doc.pdf] Context')
    with patch('app.require_auth', _make_auth_decorator()), \
         patch('app.supabase_admin', supabase_mock), \
         patch('app.get_collection_count', return_value=1), \
         patch('app.decompose_query', return_value=['What is Y?']), \
         patch('app.grade_chunks', return_value=([chunk[1]], [])), \
         patch('app.find_relevant_chunks_with_graph', return_value=[chunk]), \
         patch('app.generate_text', return_value='Answer'), \
         patch('app.get_cached_response', return_value=(None, None)), \
         patch('app.cache_response'):

        import app as flask_app
        flask_app.app.config['TESTING'] = True
        client = flask_app.app.test_client()
        res = client.post('/ask', data={
            'question': 'What is Y?',
            'session_id': 'session-xyz',
        })

        assert res.status_code == 200
        assert inserted.get('session_id') == 'session-xyz'
        assert inserted.get('question') == 'What is Y?'


# ---- /history tests ----

def test_history_returns_sessions_grouped():
    """Rows grouped by session_id; most recent session first."""
    rows = [
        {'id': 'r1', 'question': 'Q1', 'answer': 'A1', 'sources': [], 'created_at': '2026-01-01T10:00:00', 'session_id': 'sess-1'},
        {'id': 'r2', 'question': 'Q2', 'answer': 'A2', 'sources': [], 'created_at': '2026-01-01T10:01:00', 'session_id': 'sess-1'},
        {'id': 'r3', 'question': 'Q3', 'answer': 'A3', 'sources': [], 'created_at': '2026-01-01T11:00:00', 'session_id': 'sess-2'},
    ]
    with patch('app.require_auth', _make_auth_decorator()), \
         patch('app.supabase_admin', _supabase_chain(data=rows)):

        import app as flask_app
        flask_app.app.config['TESTING'] = True
        client = flask_app.app.test_client()
        res = client.get('/history')
        data = res.get_json()

        assert res.status_code == 200
        assert 'sessions' in data
        sessions = data['sessions']
        assert len(sessions) == 2
        # Most recent session first
        assert sessions[0]['session_id'] == 'sess-2'
        assert sessions[1]['session_id'] == 'sess-1'
        assert len(sessions[1]['questions']) == 2


def test_history_legacy_rows_become_singleton_sessions():
    """Rows without session_id each form their own session keyed by row id."""
    rows = [
        {'id': 'leg-1', 'question': 'Q1', 'answer': 'A1', 'sources': [], 'created_at': '2026-01-01T09:00:00', 'session_id': None},
        {'id': 'leg-2', 'question': 'Q2', 'answer': 'A2', 'sources': [], 'created_at': '2026-01-01T09:01:00', 'session_id': None},
    ]
    with patch('app.require_auth', _make_auth_decorator()), \
         patch('app.supabase_admin', _supabase_chain(data=rows)):

        import app as flask_app
        flask_app.app.config['TESTING'] = True
        client = flask_app.app.test_client()
        res = client.get('/history')
        data = res.get_json()

        assert res.status_code == 200
        assert len(data['sessions']) == 2
        for session in data['sessions']:
            assert len(session['questions']) == 1


def test_history_no_auth_returns_401():
    """No auth header → 401."""
    import app as flask_app
    flask_app.app.config['TESTING'] = True
    client = flask_app.app.test_client()
    res = client.get('/history')
    assert res.status_code == 401


# ---- /documents tests ----

def test_documents_returns_list():
    """Documents endpoint returns filename and chunk count from Supabase."""
    rows = [
        {'filename': 'alpha.pdf', 'chunk_count': 5},
        {'filename': 'beta.pdf', 'chunk_count': 2},
    ]
    with patch('app.require_auth', _make_auth_decorator()), \
         patch('app.supabase_admin', _supabase_chain(data=rows)):

        import app as flask_app
        flask_app.app.config['TESTING'] = True
        client = flask_app.app.test_client()
        res = client.get('/documents')
        data = res.get_json()

        assert res.status_code == 200
        by_name = {d['filename']: d for d in data['documents']}
        assert by_name['alpha.pdf']['chunks'] == 5
        assert by_name['beta.pdf']['chunks'] == 2


def test_documents_empty_returns_empty_list():
    """No documents → empty list, not an error."""
    with patch('app.require_auth', _make_auth_decorator()), \
         patch('app.supabase_admin', _supabase_chain(data=[])):

        import app as flask_app
        flask_app.app.config['TESTING'] = True
        client = flask_app.app.test_client()
        res = client.get('/documents')
        data = res.get_json()

        assert res.status_code == 200
        assert data['documents'] == []


def test_documents_no_auth_returns_401():
    """No auth header → 401."""
    import app as flask_app
    flask_app.app.config['TESTING'] = True
    client = flask_app.app.test_client()
    res = client.get('/documents')
    assert res.status_code == 401


# ---- Multimodal extraction tests ----

def test_rows_to_markdown_basic():
    from app import _rows_to_markdown
    md = _rows_to_markdown([['Name', 'Value'], ['Revenue', '100'], [None, '200']])
    lines = md.split('\n')
    assert lines[0] == '| Name | Value |'
    assert lines[1] == '| --- | --- |'
    assert lines[2] == '| Revenue | 100 |'
    assert lines[3] == '|  | 200 |'


def test_find_caption_prefers_figure_pattern():
    import pymupdf
    from app import _find_caption
    img_rect = pymupdf.Rect(100, 100, 300, 300)
    blocks = [
        (100, 320, 300, 340, 'Some nearby body text', 0, 0),
        (100, 350, 300, 370, 'Figure 2: Quarterly revenue', 1, 0),
    ]
    assert _find_caption(blocks, img_rect) == 'Figure 2: Quarterly revenue'


def test_find_caption_none_when_no_nearby_text():
    import pymupdf
    from app import _find_caption
    img_rect = pymupdf.Rect(100, 100, 300, 300)
    blocks = [(100, 600, 300, 620, 'Distant text', 0, 0)]
    assert _find_caption(blocks, img_rect) is None


# ---- build_source tests ----

def test_build_source_plain_text_chunk():
    from app import build_source
    source, prompt_text = build_source(0.9, '[Page 3, Source: report.pdf] Some excerpt text')
    assert source['page'] == 3
    assert source['source'] == 'report.pdf'
    assert source['text'] == 'Some excerpt text'
    assert 'image_path' not in source
    assert prompt_text == '[Page 3, Source: report.pdf] Some excerpt text'


def test_build_source_figure_chunk_extracts_image_path():
    from app import build_source
    text = '[Page 2, Source: report.pdf] [Figure: user1/figures/report.pdf/p2_f0.png] Figure 1: Revenue'
    source, prompt_text = build_source(0.8, text)
    assert source['image_path'] == 'user1/figures/report.pdf/p2_f0.png'
    assert source['text'] == 'Figure 1: Revenue'
    assert '[Figure:' not in prompt_text


def test_build_source_unparseable_falls_back():
    from app import build_source
    source, _ = build_source(0.5, 'raw text without prefix')
    assert source['page'] == 0
    assert source['source'] == 'unknown'


# ---- /figure-url tests ----

def test_figure_url_no_auth_returns_401():
    import app as flask_app
    flask_app.app.config['TESTING'] = True
    client = flask_app.app.test_client()
    res = client.get('/figure-url?path=anything')
    assert res.status_code == 401


# ---- Groq vision fallback tests ----

def _vision_response(text):
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content=text))]
    return resp


def test_describe_image_with_groq_returns_description():
    from app import describe_image_with_groq, _FIGURE_CAPTION_PROMPT
    with patch('app.groq_client') as mock_client:
        mock_client.chat.completions.create.return_value = _vision_response(
            'Bar chart of quarterly revenue.'
        )
        out = describe_image_with_groq(b'fake-png', _FIGURE_CAPTION_PROMPT)
        assert out == 'Bar chart of quarterly revenue.'
        call = mock_client.chat.completions.create.call_args
        from app import GROQ_VISION_MODEL
        assert call.kwargs['model'] == GROQ_VISION_MODEL
        content = call.kwargs['messages'][0]['content']
        assert content[1]['image_url']['url'].startswith('data:image/png;base64,')


def test_describe_image_with_groq_api_failure_returns_none():
    from app import describe_image_with_groq, _FIGURE_CAPTION_PROMPT
    with patch('app.groq_client') as mock_client:
        mock_client.chat.completions.create.side_effect = RuntimeError('rate limited')
        assert describe_image_with_groq(b'fake-png', _FIGURE_CAPTION_PROMPT) is None


def test_describe_image_with_groq_oversized_image_returns_none():
    from app import describe_image_with_groq, _FIGURE_CAPTION_PROMPT, MAX_VISION_B64_BYTES
    with patch('app.groq_client') as mock_client:
        big = b'x' * MAX_VISION_B64_BYTES  # b64 expands ~4/3, exceeds limit
        assert describe_image_with_groq(big, _FIGURE_CAPTION_PROMPT) is None
        mock_client.chat.completions.create.assert_not_called()


def test_extract_page_figures_vision_fallback_for_captionless():
    """Captionless figure gets a Groq caption when budget allows; spends budget."""
    import pymupdf
    doc = pymupdf.open()
    page = doc.new_page()
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 100, 100), False)
    pix.set_rect(pix.irect, (80, 80, 200))
    page.insert_image(pymupdf.Rect(72, 100, 192, 220), pixmap=pix)
    # no caption text anywhere on the page

    from app import extract_page_figures
    with patch('app.describe_image_with_groq', return_value='Scatter plot of test data.') as mock_vis:
        budget = {'remaining': 2}
        figures = extract_page_figures(page, vision_budget=budget)
        assert len(figures) == 1
        assert figures[0][0] == 'Scatter plot of test data.'
        assert budget['remaining'] == 1
        mock_vis.assert_called_once()
    doc.close()


def test_extract_page_figures_captionless_skipped_without_budget():
    """No budget, no nearby text → figure skipped, no vision call."""
    import pymupdf
    doc = pymupdf.open()
    page = doc.new_page()
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 100, 100), False)
    pix.set_rect(pix.irect, (80, 80, 200))
    page.insert_image(pymupdf.Rect(72, 100, 192, 220), pixmap=pix)

    from app import extract_page_figures
    with patch('app.describe_image_with_groq') as mock_vis:
        assert extract_page_figures(page) == []
        assert extract_page_figures(page, vision_budget={'remaining': 0}) == []
        mock_vis.assert_not_called()
    doc.close()


# ---- Item E: configurable retrieval width ----

def test_estimate_query_complexity_simple_question_gets_base_width():
    from app import estimate_query_complexity
    result = estimate_query_complexity('What is the effective date in section 4.2?')
    assert result == {'top_k': 20, 'top_n': 5, 'synthesis_top_n': 6}


def test_estimate_query_complexity_multiple_subqueries_widens_pool():
    """Sub-query count from decompose_query is weighted over word count (a short but
    genuinely multi-part question should still widen, per the flagged review risk)."""
    from app import estimate_query_complexity
    result = estimate_query_complexity('Compare X and Y', sub_query_count=2)
    assert result['top_k'] > 20
    assert result['top_n'] > 5


def test_estimate_query_complexity_conjunction_hint_widens_pool_even_at_one_subquery():
    from app import estimate_query_complexity
    result = estimate_query_complexity('Compare X and Y', sub_query_count=1)
    assert result['top_k'] > 20


def test_estimate_query_complexity_long_simple_question_stays_base_width():
    """A long single-fact question shouldn't get a wider pool just for being long —
    word count alone is a weak signal."""
    from app import estimate_query_complexity
    long_simple = ('What is the exact effective date specified in section 4.2 of the '
                    'amended supplier agreement signed earlier this year')
    result = estimate_query_complexity(long_simple, sub_query_count=1)
    assert result == {'top_k': 20, 'top_n': 5, 'synthesis_top_n': 6}


# ---- Item A-BM25: incremental indexing ----

def _fake_qdrant_point(point_id, text):
    point = MagicMock()
    point.id = point_id
    point.payload = {'text': text}
    return point


def test_incremental_bm25_update_adds_new_document_chunks():
    from app import _incremental_update_bm25_for_user, bm25_corpora, bm25_indices

    new_records = [_fake_qdrant_point('id-1', '[Page 1, Source: new.pdf] hello world')]
    with patch('app.qdrant') as mock_qdrant:
        mock_qdrant.scroll.return_value = (new_records, None)
        bm25_corpora.pop('bm25-user-1', None)
        bm25_indices.pop('bm25-user-1', None)
        _incremental_update_bm25_for_user('bm25-user-1', 'new.pdf')

    corpus = bm25_corpora['bm25-user-1']
    assert corpus == [('id-1', '[Page 1, Source: new.pdf] hello world')]
    assert 'bm25-user-1' in bm25_indices


def test_incremental_bm25_update_on_reindex_replaces_stale_entries_not_duplicates():
    """Re-uploading the same filename must drop the old chunks for that source even
    when the new document has a different chunk count/ids (the edge case flagged in
    the plan review) — otherwise BM25 accumulates stale/duplicate entries."""
    from app import _incremental_update_bm25_for_user, bm25_corpora, bm25_indices

    bm25_corpora['bm25-user-2'] = [
        ('old-id-1', '[Page 1, Source: doc.pdf] stale first chunk'),
        ('old-id-2', '[Page 2, Source: doc.pdf] stale second chunk'),
        ('other-id', '[Page 1, Source: other.pdf] unrelated document'),
    ]
    new_records = [_fake_qdrant_point('new-id-1', '[Page 1, Source: doc.pdf] fresh reindexed chunk')]
    with patch('app.qdrant') as mock_qdrant:
        mock_qdrant.scroll.return_value = (new_records, None)
        _incremental_update_bm25_for_user('bm25-user-2', 'doc.pdf')

    corpus = bm25_corpora['bm25-user-2']
    ids = [cid for cid, _ in corpus]
    assert 'old-id-1' not in ids
    assert 'old-id-2' not in ids
    assert 'other-id' in ids          # unrelated source untouched
    assert 'new-id-1' in ids
    assert len(corpus) == 2

    del bm25_corpora['bm25-user-2']
    bm25_indices.pop('bm25-user-2', None)


def test_rebuild_bm25_for_user_falls_back_to_full_rebuild_without_source_filename():
    """No source_filename (e.g. startup init path) → full rebuild, unchanged behavior."""
    from app import rebuild_bm25_for_user
    with patch('app._full_rebuild_bm25_for_user') as mock_full, \
         patch('app._incremental_update_bm25_for_user') as mock_incremental:
        rebuild_bm25_for_user('some-user')
        mock_full.assert_called_once_with('some-user')
        mock_incremental.assert_not_called()


def test_rebuild_bm25_for_user_falls_back_to_full_rebuild_on_incremental_error():
    from app import rebuild_bm25_for_user
    with patch('app._incremental_update_bm25_for_user', side_effect=RuntimeError('boom')), \
         patch('app._full_rebuild_bm25_for_user') as mock_full:
        rebuild_bm25_for_user('some-user', source_filename='doc.pdf')
        mock_full.assert_called_once_with('some-user')


def test_rebuild_bm25_for_user_forced_full_rebuild_env_flag():
    """BM25_FORCE_FULL_REBUILD=true is the rollback escape hatch — must bypass
    the incremental path entirely even when a source_filename is given."""
    from app import rebuild_bm25_for_user
    with patch.dict('os.environ', {'BM25_FORCE_FULL_REBUILD': 'true'}), \
         patch('app._full_rebuild_bm25_for_user') as mock_full, \
         patch('app._incremental_update_bm25_for_user') as mock_incremental:
        rebuild_bm25_for_user('some-user', source_filename='doc.pdf')
        mock_full.assert_called_once_with('some-user')
        mock_incremental.assert_not_called()


# ---- Item A-Graph: incremental in-memory graph patching ----

def _graph_supabase_mock(edges_data=None, nodes_data=None):
    """MagicMock distinguishing graph_edges vs graph_nodes table queries."""
    mock = MagicMock()

    def table_side_effect(name):
        tbl = MagicMock()
        tbl.select.return_value = tbl
        tbl.eq.return_value = tbl
        tbl.execute.return_value = MagicMock(
            data=(edges_data or []) if name == 'graph_edges' else (nodes_data or [])
        )
        return tbl

    mock.table.side_effect = table_side_effect
    return mock


def test_incremental_graph_update_adds_new_edges_and_nodes():
    from app import _incremental_update_graph_for_user, _graph_store
    import networkx as nx

    _graph_store['graph-user-1'] = nx.DiGraph()
    edges = [{'source_entity': 'acme', 'relation': 'employs', 'target_entity': 'alice',
              'chunk_id': 'c1', 'source_doc': 'new.pdf', 'page_num': 1}]
    nodes = [{'entity_name': 'acme', 'entity_type': 'org'},
             {'entity_name': 'alice', 'entity_type': 'person'}]
    with patch('app.supabase_admin', _graph_supabase_mock(edges, nodes)):
        _incremental_update_graph_for_user('graph-user-1', 'new.pdf')

    G = _graph_store['graph-user-1']
    assert G.has_edge('acme', 'alice')
    assert G['acme']['alice']['source_doc'] == 'new.pdf'
    assert G.nodes['alice']['entity_type'] == 'person'
    del _graph_store['graph-user-1']


def test_incremental_graph_update_on_reindex_replaces_stale_edges_not_duplicates():
    """Re-uploading the same filename must drop old edges for that source_doc even
    if the entities/relations changed, without touching edges from other sources."""
    from app import _incremental_update_graph_for_user, _graph_store
    import networkx as nx

    G = nx.DiGraph()
    G.add_edge('acme', 'bob', relation='employs', chunk_id='old-c1', source_doc='doc.pdf', page_num=1)
    G.add_edge('acme', 'other-co', relation='partners_with', chunk_id='oc1', source_doc='other.pdf', page_num=1)
    _graph_store['graph-user-2'] = G

    edges = [{'source_entity': 'acme', 'relation': 'employs', 'target_entity': 'carol',
              'chunk_id': 'new-c1', 'source_doc': 'doc.pdf', 'page_num': 1}]
    nodes = [{'entity_name': 'acme', 'entity_type': 'org'}, {'entity_name': 'carol', 'entity_type': 'person'}]
    with patch('app.supabase_admin', _graph_supabase_mock(edges, nodes)):
        _incremental_update_graph_for_user('graph-user-2', 'doc.pdf')

    G = _graph_store['graph-user-2']
    assert not G.has_edge('acme', 'bob')          # stale edge for this source removed
    assert G.has_edge('acme', 'carol')            # fresh edge added
    assert G.has_edge('acme', 'other-co')         # unrelated source untouched
    del _graph_store['graph-user-2']


def test_incremental_graph_update_skips_when_graph_not_cached():
    """If the user's graph isn't in memory yet, do nothing — the next lazy
    get_graph_for_user() load will already reflect the latest Supabase state."""
    from app import _incremental_update_graph_for_user, _graph_store

    _graph_store.pop('graph-user-3', None)
    with patch('app.supabase_admin') as mock_supabase:
        _incremental_update_graph_for_user('graph-user-3', 'doc.pdf')
        mock_supabase.table.assert_not_called()
    assert 'graph-user-3' not in _graph_store


def test_rebuild_graph_for_user_falls_back_to_full_rebuild_without_source_filename():
    """No source_filename → full evict+reload, unchanged behavior."""
    from app import rebuild_graph_for_user, _graph_store

    _graph_store['some-graph-user'] = MagicMock()
    with patch('app._incremental_update_graph_for_user') as mock_incremental, \
         patch('app.get_graph_for_user') as mock_get:
        rebuild_graph_for_user('some-graph-user')
        mock_incremental.assert_not_called()
        mock_get.assert_called_once_with('some-graph-user')
    assert 'some-graph-user' not in _graph_store


def test_rebuild_graph_for_user_falls_back_to_full_rebuild_on_incremental_error():
    from app import rebuild_graph_for_user

    with patch('app._incremental_update_graph_for_user', side_effect=RuntimeError('boom')), \
         patch('app.get_graph_for_user') as mock_get:
        rebuild_graph_for_user('some-graph-user', source_filename='doc.pdf')
        mock_get.assert_called_once_with('some-graph-user')


def test_rebuild_graph_for_user_forced_full_rebuild_env_flag():
    """GRAPH_FORCE_FULL_REBUILD=true is the rollback escape hatch — must bypass
    the incremental path entirely even when a source_filename is given."""
    from app import rebuild_graph_for_user

    with patch.dict('os.environ', {'GRAPH_FORCE_FULL_REBUILD': 'true'}), \
         patch('app._incremental_update_graph_for_user') as mock_incremental, \
         patch('app.get_graph_for_user') as mock_get:
        rebuild_graph_for_user('some-graph-user', source_filename='doc.pdf')
        mock_incremental.assert_not_called()
        mock_get.assert_called_once_with('some-graph-user')


# ---- Item C: alias-based entity linking ----

def test_build_graph_from_supabase_loads_aliases_onto_nodes():
    from app import _build_graph_from_supabase

    nodes = [{'entity_name': 'microsoft', 'entity_type': 'org', 'aliases': ['msft']}]
    edges = []
    with patch('app.supabase_admin', _graph_supabase_mock_nodes_edges(nodes, edges)):
        G = _build_graph_from_supabase('alias-user-1')

    assert G.nodes['microsoft']['aliases'] == ['msft']


def _graph_supabase_mock_nodes_edges(nodes_data, edges_data):
    mock = MagicMock()

    def table_side_effect(name):
        tbl = MagicMock()
        tbl.select.return_value = tbl
        tbl.eq.return_value = tbl
        tbl.execute.return_value = MagicMock(
            data=nodes_data if name == 'graph_nodes' else edges_data
        )
        return tbl

    mock.table.side_effect = table_side_effect
    return mock


def test_graph_expand_candidates_matches_via_alias_when_no_substring_match():
    """'MSFT' should hit the 'microsoft' node via its alias even though 'microsoft'
    never appears in the query text — pure substring match would miss this."""
    from app import graph_expand_candidates
    import networkx as nx

    G = nx.DiGraph()
    G.add_node('microsoft', entity_type='org', aliases=['msft'])
    G.add_node('azure', entity_type='product', aliases=[])
    G.add_edge('microsoft', 'azure', relation='owns', chunk_id='c1', source_doc='doc.pdf', page_num=1)

    hit = MagicMock()
    hit.payload = {'text': '[Page 1, Source: doc.pdf] Azure is a cloud platform.'}
    with patch('app.get_graph_for_user', return_value=G), \
         patch('app.qdrant') as mock_qdrant:
        mock_qdrant.retrieve.return_value = [hit]
        expanded = graph_expand_candidates('What does MSFT own?', 'alias-user-2', existing_texts=set())

    assert len(expanded) == 1
    assert 'Azure' in expanded[0][1]


def test_extract_and_store_graph_merges_new_aliases_with_existing():
    """Re-extraction for an entity must union new aliases with previously stored
    ones, not overwrite them — otherwise a reindex silently drops known aliases."""
    from app import extract_and_store_graph
    import json as json_module

    llm_response = json_module.dumps([{
        'chunk_index': 0,
        'entities': [{'name': 'microsoft', 'type': 'org', 'aliases': ['MSFT']}],
        'triples': [],
    }])
    existing_nodes = [{'entity_name': 'microsoft', 'entity_type': 'org', 'aliases': ['ms']}]

    mock_supabase = _graph_supabase_mock_nodes_edges(existing_nodes, [])
    upserted = {}

    def capture_upsert(rows, on_conflict=None):
        upserted['rows'] = rows
        result = MagicMock()
        result.execute.return_value = MagicMock(data=[])
        return result

    mock_supabase.table.side_effect = (
        lambda name: MagicMock(
            select=MagicMock(return_value=MagicMock(
                eq=MagicMock(return_value=MagicMock(
                    execute=MagicMock(return_value=MagicMock(data=existing_nodes))
                ))
            )),
            upsert=capture_upsert,
        ) if name == 'graph_nodes' else MagicMock(
            insert=MagicMock(return_value=MagicMock(execute=MagicMock(return_value=MagicMock(data=[]))))
        )
    )

    with patch('app.supabase_admin', mock_supabase), \
         patch('app._call_groq_helper', return_value=llm_response):
        extract_and_store_graph([('c1', 1, 'Microsoft owns Azure.')], 'alias-user-3', 'doc.pdf')

    aliases = upserted['rows'][0]['aliases']
    assert set(aliases) == {'ms', 'msft'}


# ---- Item F: Qdrant scale instrumentation ----

def test_get_total_collection_count_returns_qdrant_count():
    from app import get_total_collection_count
    with patch('app.qdrant') as mock_qdrant:
        mock_qdrant.count.return_value = MagicMock(count=42)
        assert get_total_collection_count() == 42


def test_get_total_collection_count_returns_zero_on_qdrant_error():
    from app import get_total_collection_count
    with patch('app.qdrant') as mock_qdrant:
        mock_qdrant.count.side_effect = RuntimeError('down')
        assert get_total_collection_count() == 0


def test_check_qdrant_shard_threshold_warns_when_over_threshold():
    from app import check_qdrant_shard_threshold
    with patch('app.get_total_collection_count', return_value=1_500_000), \
         patch.dict('os.environ', {'QDRANT_SHARD_ALERT_THRESHOLD': '1000000'}), \
         patch('app.logging.warning') as mock_warn:
        check_qdrant_shard_threshold()
        mock_warn.assert_called_once()
        assert '1500000' in mock_warn.call_args[0][0] or '1,500,000' in mock_warn.call_args[0][0]


def test_check_qdrant_shard_threshold_silent_when_under_threshold():
    from app import check_qdrant_shard_threshold
    with patch('app.get_total_collection_count', return_value=100), \
         patch.dict('os.environ', {'QDRANT_SHARD_ALERT_THRESHOLD': '1000000'}), \
         patch('app.logging.warning') as mock_warn:
        check_qdrant_shard_threshold()
        mock_warn.assert_not_called()


# ---- Item D: bounded iterative CRAG loop ----

def test_ask_file_agentic_stops_iterating_once_relevant_chunks_found():
    from app import ask_file_agentic
    chunk = (0.9, '[Page 1, Source: doc.pdf] relevant content')
    with patch('app.get_collection_count', return_value=1), \
         patch('app.decompose_query', return_value=['test question']), \
         patch('app.find_relevant_chunks_with_graph', return_value=[chunk]) as mock_retrieve, \
         patch('app.grade_chunks', return_value=([chunk[1]], [])) as mock_grade, \
         patch('app.generate_text', return_value='Answer'):
        response, sources = ask_file_agentic('test question', 'user-1')

    assert response == 'Answer'
    assert mock_retrieve.call_count == 1   # found relevant on first iteration, no retries
    assert mock_grade.call_count == 1


def test_ask_file_agentic_bounded_by_max_iterations_when_nothing_ever_relevant():
    """Grading keeps returning zero-relevant — loop must stop at MAX_CRAG_ITERATIONS,
    not spin forever, and still produce an answer from best-effort chunks."""
    import app
    chunk = (0.5, '[Page 1, Source: doc.pdf] never graded relevant')
    with patch('app.get_collection_count', return_value=1), \
         patch('app.decompose_query', return_value=['test question']), \
         patch('app.find_relevant_chunks_with_graph', return_value=[chunk]) as mock_retrieve, \
         patch('app.grade_chunks', return_value=([], [chunk[1]])), \
         patch('app.reformulate_query', side_effect=lambda q: q), \
         patch('app.generate_text', return_value='Answer'):
        response, sources = app.ask_file_agentic('test question', 'user-1')

    assert mock_retrieve.call_count == app.MAX_CRAG_ITERATIONS
    # nothing was ever graded relevant, so no chunks reach the synthesis prompt
    assert sources == []
    assert response == "I couldn't find relevant information in your documents to answer this question."


def test_ask_file_agentic_grading_failure_uses_retrieved_chunks_without_burning_iterations():
    """GradingUnavailableError (grader itself broken) must stop retrying immediately
    and use what was retrieved, rather than spending the full iteration budget."""
    import app
    from app import GradingUnavailableError
    chunk = (0.7, '[Page 1, Source: doc.pdf] some content')
    with patch('app.get_collection_count', return_value=1), \
         patch('app.decompose_query', return_value=['test question']), \
         patch('app.find_relevant_chunks_with_graph', return_value=[chunk]) as mock_retrieve, \
         patch('app.grade_chunks', side_effect=GradingUnavailableError('grader down')), \
         patch('app.generate_text', return_value='Answer') as mock_gen:
        response, sources = app.ask_file_agentic('test question', 'user-1')

    assert mock_retrieve.call_count == 1
    assert response == 'Answer'
    assert len(sources) == 1
    mock_gen.assert_called_once()


def test_ask_file_agentic_wall_clock_budget_stops_further_iterations():
    """A slow environment must not blow past the wall-clock budget just because the
    iteration counter hasn't run out yet."""
    import app
    chunk = (0.5, '[Page 1, Source: doc.pdf] slow content')
    with patch('app.get_collection_count', return_value=1), \
         patch('app.decompose_query', return_value=['test question']), \
         patch('app.CRAG_WALL_CLOCK_BUDGET_S', 0), \
         patch('app.find_relevant_chunks_with_graph', return_value=[chunk]) as mock_retrieve, \
         patch('app.grade_chunks', return_value=([], [chunk[1]])), \
         patch('app.reformulate_query', side_effect=lambda q: q), \
         patch('app.generate_text', return_value='Answer'):
        app.ask_file_agentic('test question', 'user-1')

    # budget is already exhausted before the first iteration even starts
    assert mock_retrieve.call_count == 0
