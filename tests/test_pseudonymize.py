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
