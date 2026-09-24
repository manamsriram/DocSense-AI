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
    failures. Also clears the reverse-mapping cache for the same reason."""
    pseudonymize._mapping_cache.clear()
    pseudonymize._reverse_mapping_cache.clear()
    pseudonymize._entity_types_cache.clear()
    yield
    pseudonymize._mapping_cache.clear()
    pseudonymize._reverse_mapping_cache.clear()
    pseudonymize._entity_types_cache.clear()


def _mock_supabase_table(rows_by_table, range_pages=None):
    """Returns a MagicMock standing in for app.supabase_admin, where
    .table(name).select/insert/upsert(...).execute() chains return
    canned data from rows_by_table[name].

    range_pages: optional {table_name: [page1_rows, page2_rows, ...]} --
    when given, .select(...).eq(...).range(start, end).execute() returns
    successive pages on successive calls (for pagination tests), instead
    of the single rows_by_table[name] snapshot every call.
    """
    mock = MagicMock()
    tbl_cache = {}

    def table_side_effect(name):
        if name not in tbl_cache:
            tbl = MagicMock()
            result = MagicMock()
            result.data = rows_by_table.get(name, [])
            tbl.select.return_value.eq.return_value.execute.return_value = result
            tbl.select.return_value.eq.return_value.eq.return_value.execute.return_value = result
            tbl.insert.return_value.execute.return_value = result
            tbl.upsert.return_value.execute.return_value = result

            if range_pages and name in range_pages:
                pages = list(range_pages[name])

                def range_execute_side_effect(_pages=pages):
                    page_result = MagicMock()
                    page_result.data = _pages.pop(0) if _pages else []
                    return page_result

                range_mock = tbl.select.return_value.eq.return_value.order.return_value.range
                range_mock.return_value.execute.side_effect = range_execute_side_effect
            else:
                tbl.select.return_value.eq.return_value.order.return_value.range.return_value.execute.return_value = result

            tbl_cache[name] = tbl
        return tbl_cache[name]

    mock.table.side_effect = table_side_effect
    # Pre-populate cache and set return_value to match
    for table_name in rows_by_table.keys():
        table_side_effect(table_name)
    # Set return_value to pseudonym_mappings for test assertions
    if 'pseudonym_mappings' in tbl_cache:
        mock.table.return_value = tbl_cache['pseudonym_mappings']

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


def test_get_org_entity_types_falls_back_on_malformed_shape():
    """pseudonymize_entities is unconstrained jsonb -- a malformed value
    (e.g. a string instead of a list) must fall back to
    DEFAULT_ENTITY_TYPES with a clear error, not get handed straight to
    Presidio's entities= param where it would fail opaquely mid-NER."""
    fake_supabase = _mock_supabase_table(
        {'orgs': [{'pseudonymize_entities': 'PERSON'}]}  # string, not a list
    )
    with patch('pseudonymize.app.supabase_admin', fake_supabase):
        assert pseudonymize.get_org_entity_types('org-1') == DEFAULT_ENTITY_TYPES


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
        pseudonym = pseudonymize.get_or_create_pseudonym(
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
        mapping, pattern = pseudonymize._fetch_org_mapping('org-1')
    assert mapping == {'Jane Doe': 'PERSON_ab12', 'jane@example.com': 'EMAIL_ADDRESS_cd34'}
    assert pattern is not None


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
    # Verify the org-scoped entity list and score_threshold were passed to .analyze()
    mock_get_analyzer.return_value.analyze.assert_called_once_with(
        text='Jane Doe filed the report.',
        entities=['PERSON'],
        language='en',
        score_threshold=pseudonymize.NER_SCORE_THRESHOLD,
    )
    fake_supabase.table.return_value.upsert.assert_called_once()
    upserted = fake_supabase.table.return_value.upsert.call_args[0][0]
    assert upserted['real_value'] == 'Jane Doe'
    assert upserted['entity_type'] == 'PERSON'


def test_detect_and_register_entities_skips_too_short_detection():
    """Fix #6: a detection shorter than MIN_ENTITY_LENGTH after stripping
    whitespace must not be registered, even if Presidio scored it above
    score_threshold -- short tokens ("US", "Hi") are disproportionately
    false positives, and once registered they get substituted everywhere."""
    fake_supabase = _mock_supabase_table({
        'orgs': [{'pseudonymize_entities': ['LOCATION']}],
        'pseudonym_mappings': [],
    })
    # "US" is 2 chars -- below MIN_ENTITY_LENGTH (3)
    fake_analyzer_result = [MagicMock(entity_type='LOCATION', start=0, end=2)]
    with patch('pseudonymize.app.supabase_admin', fake_supabase), \
         patch('pseudonymize._get_analyzer') as mock_get_analyzer:
        mock_get_analyzer.return_value.analyze.return_value = fake_analyzer_result
        pseudonymize.detect_and_register_entities('US filed the report.', 'org-1')
    fake_supabase.table.return_value.upsert.assert_not_called()


def test_detect_and_register_query_entities_registers_structured_pii():
    """Critical fix: a question is never ingested, so pseudonymize_text alone
    leaves PII typed directly into it unchanged. This regex-only,
    NER-free counterpart must register structured PII (SSN, email, phone,
    card) found directly in question text so a subsequent pseudonymize_text
    call actually substitutes it."""
    fake_supabase = _mock_supabase_table({
        'orgs': [{'pseudonymize_entities': ['US_SSN', 'EMAIL_ADDRESS']}],
        'pseudonym_mappings': [],
    })
    with patch('pseudonymize.app.supabase_admin', fake_supabase):
        pseudonymize.detect_and_register_query_entities(
            'Is 123-45-6789 or jane@example.com in these files?', 'org-1'
        )
    upserted = [c[0][0] for c in fake_supabase.table.return_value.upsert.call_args_list]
    real_values = {row['real_value'] for row in upserted}
    assert real_values == {'123-45-6789', 'jane@example.com'}


def test_detect_and_register_query_entities_skips_types_not_in_org_config():
    """An entity type not in the org's configured pseudonymize_entities must
    not be registered, same scoping detect_and_register_entities honors."""
    fake_supabase = _mock_supabase_table({
        'orgs': [{'pseudonymize_entities': ['EMAIL_ADDRESS']}],  # SSN not included
        'pseudonym_mappings': [],
    })
    with patch('pseudonymize.app.supabase_admin', fake_supabase):
        pseudonymize.detect_and_register_query_entities('SSN is 123-45-6789.', 'org-1')
    fake_supabase.table.return_value.upsert.assert_not_called()


def test_detect_and_register_query_entities_skips_person_and_location():
    """PERSON/LOCATION need spaCy NER and are deliberately NOT covered by
    this regex-only query-time path (see module comment on
    _QUERY_TIME_PATTERNS) -- a name typed directly into a question is a
    known, accepted gap, not something this function should silently
    attempt and get wrong."""
    fake_supabase = _mock_supabase_table({
        'orgs': [{'pseudonymize_entities': pseudonymize.DEFAULT_ENTITY_TYPES}],
        'pseudonym_mappings': [],
    })
    with patch('pseudonymize.app.supabase_admin', fake_supabase):
        pseudonymize.detect_and_register_query_entities('Is Jane Doe in these files?', 'org-1')
    fake_supabase.table.return_value.upsert.assert_not_called()


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


def test_pseudonymize_text_word_boundaries_prevent_substring_corruption():
    """Verify that word boundaries prevent "John" from matching inside "Johnny"."""
    fake_supabase = _mock_supabase_table({'pseudonym_mappings': [
        {'real_value': 'John', 'pseudonym': 'PERSON_aaaa'},
    ]})
    with patch('pseudonymize.app.supabase_admin', fake_supabase):
        # "Johnny" should not be corrupted because "John" is not a complete word inside it
        result = pseudonymize.pseudonymize_text('Johnny and John both signed.', 'org-1')
    assert result == 'Johnny and PERSON_aaaa both signed.'


def test_pseudonymize_text_matches_punctuation_containing_real_value():
    """The exact case that motivated switching from \\b to (?<!\\w)/(?!\\w)
    (see _compile_pattern): a real_value like a phone number starts/ends
    with punctuation, so \\b -- a transition between \\w and \\W -- never
    matches there and the value would never be substituted. Round-trips
    both directions so a regression here (e.g. reverting to \\b, or a
    re.escape interaction with parentheses/dashes) is caught."""
    fake_supabase = _mock_supabase_table({'pseudonym_mappings': [
        {'real_value': '+1 (555) 123-4567', 'pseudonym': 'PHONE_NUMBER_ff00'},
        {'real_value': '123-45-6789', 'pseudonym': 'US_SSN_ee11'},
    ]})
    with patch('pseudonymize.app.supabase_admin', fake_supabase):
        forward = pseudonymize.pseudonymize_text(
            'Call +1 (555) 123-4567 or reference SSN 123-45-6789 for verification.', 'org-1'
        )
        assert forward == 'Call PHONE_NUMBER_ff00 or reference SSN US_SSN_ee11 for verification.'

        backward = pseudonymize.deanonymize_text(
            'Call PHONE_NUMBER_ff00 or reference SSN US_SSN_ee11 for verification.', 'org-1'
        )
        assert backward == 'Call +1 (555) 123-4567 or reference SSN 123-45-6789 for verification.'


def test_deanonymize_reverse_cache_is_atomic():
    """Verify that reverse mapping and compiled pattern come from the same
    cache entry, preventing stale-pattern-vs-fresh-mapping races."""
    fake_supabase = _mock_supabase_table({'pseudonym_mappings': [
        {'real_value': 'Jane Doe', 'pseudonym': 'PERSON_ab12'},
    ]})
    with patch('pseudonymize.app.supabase_admin', fake_supabase):
        # Call deanonymize_text to populate the reverse cache
        text = 'PERSON_ab12 signed.'
        result = pseudonymize.deanonymize_text(text, 'org-1')

    # Verify cache entry exists and contains both mapping (index 1) and pattern (index 2)
    with pseudonymize._reverse_mapping_cache_lock:
        cached = pseudonymize._reverse_mapping_cache.get('org-1')

    # Cache tuple should be (fetched_at, reverse_mapping, compiled_pattern)
    assert cached is not None, "Reverse cache entry should exist after deanonymize_text"
    assert len(cached) == 3, "Cache entry should have (fetched_at, reverse_mapping, pattern)"
    fetched_at, reverse_mapping, pattern = cached
    assert isinstance(reverse_mapping, dict), "Second element should be reverse mapping dict"
    assert pattern is not None, "Third element should be compiled pattern (not None)"
    assert 'PERSON_ab12' in reverse_mapping, "Reverse mapping should contain pseudonym"
    assert reverse_mapping['PERSON_ab12'] == 'Jane Doe', "Reverse mapping should map pseudonym to real value"
    # Verify the pattern works on the reverse mapping
    assert result == 'Jane Doe signed.', "Deanonymize should work with atomic cache"


def test_make_pseudonym_digest_is_16_hex_chars():
    """16 hex chars (64 bits) with no DB uniqueness constraint on
    (org_id, pseudonym) pushes the birthday-paradox 50%-collision point out
    to ~5 billion distinct values per org."""
    pseudonym = pseudonymize._make_pseudonym('PERSON', 'Jane Doe', 'org-1')
    prefix, _, digest = pseudonym.rpartition('_')
    assert prefix == 'PERSON'
    assert len(digest) == 16
    assert all(c in '0123456789abcdef' for c in digest)


def test_make_pseudonym_differs_across_orgs_for_same_real_value():
    """Fix #2: pseudonyms must be org-scoped -- otherwise a provider that
    sees the same pseudonym for 'Jane Doe' across two different orgs' LLM
    calls can link the same person cross-tenant."""
    pseudonym_org1 = pseudonymize._make_pseudonym('PERSON', 'Jane Doe', 'org-1')
    pseudonym_org2 = pseudonymize._make_pseudonym('PERSON', 'Jane Doe', 'org-2')
    assert pseudonym_org1 != pseudonym_org2


def test_make_pseudonym_is_not_plain_unsalted_sha256():
    """Fix #2: the old scheme (plain sha256(real_value)[:8]) is reversible
    offline by anyone who knows the scheme (it's public, in this repo) --
    brute-forcing structured values like SSNs is ~2^30 work. The new HMAC
    scheme must depend on PSEUDONYM_SECRET, so the digest differs from the
    unsalted hash and can't be reproduced without the secret."""
    import hashlib
    real_value = 'Jane Doe'
    pseudonym = pseudonymize._make_pseudonym('PERSON', real_value, 'org-1')
    _, _, digest = pseudonym.rpartition('_')
    unsalted_digest = hashlib.sha256(real_value.encode()).hexdigest()[:8]
    assert digest != unsalted_digest


def test_make_pseudonym_depends_on_secret():
    """Fix #2: changing PSEUDONYM_SECRET must change the pseudonym for the
    same (org_id, real_value) -- proves the digest is actually keyed by the
    secret, not just incidentally different from unsalted sha256."""
    pseudonym_before = pseudonymize._make_pseudonym('PERSON', 'Jane Doe', 'org-1')
    with patch('pseudonymize.PSEUDONYM_SECRET', b'a-different-secret'):
        pseudonym_after = pseudonymize._make_pseudonym('PERSON', 'Jane Doe', 'org-1')
    assert pseudonym_before != pseudonym_after


def test_reverse_cache_reflects_forward_refresh_without_waiting_out_own_ttl():
    """The reverse cache must never serve a mapping snapshot older than the
    forward cache's current one. Before this fix, _fetch_org_mapping_reverse
    tracked its own independent TTL clock, so within a single /ask request
    the forward cache could pick up a newly-registered entity (via a
    pseudonymize_text call elsewhere in the request) while the reverse
    cache -- still "fresh" by its own clock -- kept serving the old
    snapshot, leaving a literal pseudonym token in the deanonymized answer."""
    fake_supabase = _mock_supabase_table({'pseudonym_mappings': [
        {'real_value': 'Jane Doe', 'pseudonym': 'PERSON_ab12'},
    ]})
    with patch('pseudonymize.app.supabase_admin', fake_supabase):
        # Populate both the forward and reverse caches with the initial mapping.
        assert pseudonymize.deanonymize_text('PERSON_ab12 signed.', 'org-1') == 'Jane Doe signed.'

    # A new entity gets registered mid-request (e.g. detect_and_register_entities
    # during ingestion, or a concurrent request) -- Supabase now has it too.
    fake_supabase2 = _mock_supabase_table({'pseudonym_mappings': [
        {'real_value': 'Jane Doe', 'pseudonym': 'PERSON_ab12'},
        {'real_value': 'Acme Corp', 'pseudonym': 'ORG_cd34'},
    ]})

    # Force the forward cache to look expired (simulating TTL rollover)
    # without touching the reverse cache directly -- isolates the scenario
    # where the forward cache refreshes on its own schedule.
    with pseudonymize._mapping_cache_lock:
        entry = pseudonymize._mapping_cache['org-1']
        pseudonymize._mapping_cache['org-1'] = entry._replace(
            fetched_at=entry.fetched_at - pseudonymize._MAPPING_CACHE_TTL_S - 1,
        )

    with patch('pseudonymize.app.supabase_admin', fake_supabase2):
        # Something else in the request path refreshes the forward cache first.
        pseudonymize._fetch_org_mapping('org-1')
        # deanonymize_text must see the new entity immediately, not wait out
        # its own independent TTL window.
        result = pseudonymize.deanonymize_text('ORG_cd34 employs PERSON_ab12.', 'org-1')

    assert result == 'Acme Corp employs Jane Doe.'


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


def test_fetch_org_mapping_pages_past_1000_rows():
    """Fix #1 (critical): a single unpaginated .select() silently truncates
    at Supabase's default 1000-row cap, leaving every entity past #1000
    unprotected (sent raw to LLMs, no error). _fetch_org_mapping must page
    through with .range() until a page shorter than 1000 rows comes back,
    accumulating every row across multiple .execute() calls."""
    page1 = [
        {'real_value': f'Person {i}', 'pseudonym': f'PERSON_{i:08x}'}
        for i in range(1000)
    ]
    page2 = [
        {'real_value': f'Person {i}', 'pseudonym': f'PERSON_{i:08x}'}
        for i in range(1000, 1250)
    ]
    fake_supabase = _mock_supabase_table(
        {'pseudonym_mappings': []},  # unused fallback -- range_pages drives this test
        range_pages={'pseudonym_mappings': [page1, page2]},
    )
    with patch('pseudonymize.app.supabase_admin', fake_supabase):
        mapping, pattern = pseudonymize._fetch_org_mapping('org-1')

    assert len(mapping) == 1250
    assert mapping['Person 0'] == 'PERSON_00000000'
    assert mapping['Person 1249'] == f'PERSON_{1249:08x}'
    # Confirms two .range() pages were actually fetched (not a single call).
    range_mock = fake_supabase.table.return_value.select.return_value.eq.return_value.order.return_value.range
    assert range_mock.return_value.execute.call_count == 2


def test_pseudonymize_text_matches_pseudonym_token_case_insensitively():
    """Fix #3: the graph-extraction LLM prompt instructs the model to
    lowercase entity names, so a pseudonym token like 'PERSON_a1b2c3d4' can
    come back from the LLM as 'person_a1b2c3d4'. deanonymize_text's match
    must be case-insensitive but still substitute the CANONICAL stored
    value regardless of the case actually matched."""
    fake_supabase = _mock_supabase_table({'pseudonym_mappings': [
        {'real_value': 'Jane Doe', 'pseudonym': 'PERSON_a1b2c3d4'},
    ]})
    with patch('pseudonymize.app.supabase_admin', fake_supabase):
        # LLM echoed the pseudonym token back lowercased.
        result = pseudonymize.deanonymize_text('person_a1b2c3d4 signed the report.', 'org-1')
    assert result == 'Jane Doe signed the report.'


def test_pseudonymize_text_forward_matching_is_case_sensitive():
    """Forward (real_value -> pseudonym) matching is deliberately
    case-SENSITIVE (unlike the reverse direction above): real_value's case
    comes straight from NER on the original text, and matching it
    case-insensitively would also pseudonymize any lowercase word that only
    coincidentally shares a spelling with a registered value -- e.g. a
    spaCy PERSON false positive on "May"/"Will"/"Bill" would then also
    substitute every lowercase "may"/"will"/"bill" in unrelated prompts."""
    fake_supabase = _mock_supabase_table({'pseudonym_mappings': [
        {'real_value': 'Jane Doe', 'pseudonym': 'PERSON_ab12'},
    ]})
    with patch('pseudonymize.app.supabase_admin', fake_supabase):
        exact = pseudonymize.pseudonymize_text('Jane Doe signed the report.', 'org-1')
        different_case = pseudonymize.pseudonymize_text('JANE DOE signed the report.', 'org-1')
    assert exact == 'PERSON_ab12 signed the report.'
    assert different_case == 'JANE DOE signed the report.'


def _fake_module(name, **attrs):
    """Build a bare types.ModuleType and inject it into sys.modules under
    `name` via the returned context manager (patch.dict) -- lets tests
    exercise pseudonymize's lazy `from presidio_x import Y` statements
    without presidio actually being installed in this test environment
    (it isn't; it's only a runtime dependency of the Docker image / GH
    Actions runner per the module docstring)."""
    import types
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


def test_get_analyzer_configures_en_core_web_sm_nlp_engine():
    """Fix #5: Presidio's default NlpEngineProvider() loads en_core_web_lg
    (~400-560MB), but only en_core_web_sm is installed per the
    Dockerfile/ingest.yml (Task 8). _get_analyzer must build an explicit
    NlpEngineProvider(nlp_configuration=...) pinned to en_core_web_sm and
    pass that engine into AnalyzerEngine(nlp_engine=...), not rely on
    AnalyzerEngine()'s unconfigured default."""
    import sys
    pseudonymize._analyzer = None
    try:
        MockAnalyzerEngine = MagicMock()
        MockProvider = MagicMock()
        fake_engine = MagicMock()
        MockProvider.return_value.create_engine.return_value = fake_engine

        fake_presidio_analyzer = _fake_module('presidio_analyzer', AnalyzerEngine=MockAnalyzerEngine)
        fake_nlp_engine_mod = _fake_module('presidio_analyzer.nlp_engine', NlpEngineProvider=MockProvider)
        fake_presidio_analyzer.nlp_engine = fake_nlp_engine_mod

        with patch.dict(sys.modules, {
            'presidio_analyzer': fake_presidio_analyzer,
            'presidio_analyzer.nlp_engine': fake_nlp_engine_mod,
        }):
            pseudonymize._get_analyzer()

        MockProvider.assert_called_once_with(
            nlp_configuration={
                'nlp_engine_name': 'spacy',
                'models': [{'lang_code': 'en', 'model_name': 'en_core_web_sm'}],
            }
        )
        MockAnalyzerEngine.assert_called_once_with(nlp_engine=fake_engine)
    finally:
        pseudonymize._analyzer = None


def test_get_image_redactor_shares_analyzer_engine():
    """Fix #5: ImageRedactorEngine must reuse the SAME AnalyzerEngine
    instance built via _get_analyzer() (already configured for
    en_core_web_sm) rather than letting ImageRedactorEngine() construct
    its own separate, unconfigured AnalyzerEngine (which would default to
    en_core_web_lg the same way _get_analyzer() used to)."""
    import sys
    pseudonymize._analyzer = None
    pseudonymize._image_redactor = None
    try:
        fake_analyzer = MagicMock()
        MockImageAnalyzer = MagicMock()
        MockRedactorEngine = MagicMock()
        fake_presidio_image_redactor = _fake_module(
            'presidio_image_redactor',
            ImageAnalyzerEngine=MockImageAnalyzer,
            ImageRedactorEngine=MockRedactorEngine,
        )

        with patch('pseudonymize._get_analyzer', return_value=fake_analyzer), \
             patch.dict(sys.modules, {'presidio_image_redactor': fake_presidio_image_redactor}):
            pseudonymize._get_image_redactor()

        MockImageAnalyzer.assert_called_once_with(analyzer_engine=fake_analyzer)
        MockRedactorEngine.assert_called_once_with(
            image_analyzer_engine=MockImageAnalyzer.return_value
        )
    finally:
        pseudonymize._analyzer = None
        pseudonymize._image_redactor = None


def test_fetch_org_mapping_reverse_returns_pattern_without_separate_fetch():
    """Fix #8: _fetch_org_mapping_reverse must return (mapping, pattern)
    atomically from its own cache read, so callers never need a second,
    separate lock acquisition to re-fetch the pattern -- eliminating the
    stale-pattern-vs-fresh-mapping race that caused KeyError in
    _substitute."""
    fake_supabase = _mock_supabase_table({'pseudonym_mappings': [
        {'real_value': 'Jane Doe', 'pseudonym': 'PERSON_ab12'},
    ]})
    with patch('pseudonymize.app.supabase_admin', fake_supabase):
        reverse_mapping, pattern = pseudonymize._fetch_org_mapping_reverse('org-1')
    assert reverse_mapping == {'PERSON_ab12': 'Jane Doe'}
    assert pattern is not None
    assert pattern.search('PERSON_ab12') is not None


def test_fetch_org_mapping_reverse_retries_instead_of_keyerror_on_concurrent_invalidation():
    """Regression: a concurrent _invalidate_mapping_cache could pop the
    forward cache entry between _fetch_org_mapping_reverse's unlocked
    fetch and its lock re-acquisition -- this used to KeyError on a bare
    `_mapping_cache[org_id]` index instead of retrying."""
    fake_supabase = _mock_supabase_table({'pseudonym_mappings': [
        {'real_value': 'Jane Doe', 'pseudonym': 'PERSON_ab12'},
    ]})
    original_fetch = pseudonymize._fetch_org_mapping_with_retry
    call_count = {'n': 0}

    def flaky_fetch(org_id):
        call_count['n'] += 1
        result = original_fetch(org_id)
        if call_count['n'] == 1:
            # Simulate another thread's _invalidate_mapping_cache landing
            # right after this fetch repopulated the cache.
            with pseudonymize._mapping_cache_lock:
                pseudonymize._mapping_cache.pop(org_id, None)
        return result

    with patch('pseudonymize.app.supabase_admin', fake_supabase), \
         patch('pseudonymize._fetch_org_mapping_with_retry', side_effect=flaky_fetch):
        reverse_mapping, pattern = pseudonymize._fetch_org_mapping_reverse('org-1')

    assert reverse_mapping == {'PERSON_ab12': 'Jane Doe'}
    assert call_count['n'] >= 2  # had to retry after the simulated race


def test_lazy_app_proxy_defers_import_and_stays_patchable():
    """Fix #9: pseudonymize.app must not be a plain top-level `import app`
    (that double-initializes Flask/clients if app.py is ever run directly
    via `python app.py`, since the running script is registered in
    sys.modules as '__main__', not 'app', so pseudonymize's own `import
    app` would re-execute app.py as a second module). It must instead be a
    lazy proxy that only imports on first attribute access, while still
    being patchable via the existing `patch('pseudonymize.app.<attr>',
    ...)` convention used throughout this file."""
    assert isinstance(pseudonymize.app, pseudonymize._LazyApp)
    fake_supabase = _mock_supabase_table({'pseudonym_mappings': []})
    with patch('pseudonymize.app.supabase_admin', fake_supabase):
        assert pseudonymize.app.supabase_admin is fake_supabase
