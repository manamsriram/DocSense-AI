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
         patch('app.find_relevant_chunks', return_value=chunks), \
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
         patch('app.find_relevant_chunks', return_value=[chunk]), \
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
         patch('app.find_relevant_chunks', return_value=[chunk]), \
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
