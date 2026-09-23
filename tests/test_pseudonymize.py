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
    yield
    pseudonymize._mapping_cache.clear()
    pseudonymize._reverse_mapping_cache.clear()


def _mock_supabase_table(rows_by_table):
    """Returns a MagicMock standing in for app.supabase_admin, where
    .table(name).select/insert/upsert(...).execute() chains return
    canned data from rows_by_table[name]."""
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
        mapping = pseudonymize._fetch_org_mapping('org-1')
    assert mapping == {'Jane Doe': 'PERSON_ab12', 'jane@example.com': 'EMAIL_ADDRESS_cd34'}


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
    # Verify the org-scoped entity list was actually passed to .analyze()
    mock_get_analyzer.return_value.analyze.assert_called_once_with(
        text='Jane Doe filed the report.',
        entities=['PERSON'],
        language='en'
    )
    fake_supabase.table.return_value.upsert.assert_called_once()
    upserted = fake_supabase.table.return_value.upsert.call_args[0][0]
    assert upserted['real_value'] == 'Jane Doe'
    assert upserted['entity_type'] == 'PERSON'


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


def test_make_pseudonym_digest_is_8_hex_chars():
    """4 hex chars (16 bits) collides ~50% of the time (birthday paradox) once
    an org has ~300 distinct names of one entity type -- 8 hex chars (32 bits)
    pushes that threshold out to ~77,000 names."""
    pseudonym = pseudonymize._make_pseudonym('PERSON', 'Jane Doe')
    prefix, _, digest = pseudonym.rpartition('_')
    assert prefix == 'PERSON'
    assert len(digest) == 8
    assert all(c in '0123456789abcdef' for c in digest)


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
        fetched_at, mapping, pattern = pseudonymize._mapping_cache['org-1']
        pseudonymize._mapping_cache['org-1'] = (
            fetched_at - pseudonymize._MAPPING_CACHE_TTL_S - 1, mapping, pattern,
        )

    with patch('pseudonymize.app.supabase_admin', fake_supabase2):
        # Something else in the request path refreshes the forward cache first.
        pseudonymize._fetch_org_mapping('org-1')
        # deanonymize_text must see the new entity immediately, not wait out
        # its own independent TTL window.
        result = pseudonymize.deanonymize_text('ORG_cd34 employs PERSON_ab12.', 'org-1')

    assert result == 'Acme Corp employs Jane Doe.'
