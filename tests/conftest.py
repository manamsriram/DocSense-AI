import os
from dotenv import load_dotenv

# Load .env first so real credentials are available, then stub any that are
# still missing or empty (e.g. SUPABASE_SERVICE_KEY left blank in .env).
load_dotenv()

_REQUIRED_ENV_STUBS = {
    'SUPABASE_URL': 'https://test.supabase.co',
    'SUPABASE_ANON_KEY': 'test-anon-key',
    'SUPABASE_SERVICE_KEY': 'test-service-key',
    'GROQ_API_KEY': 'test-groq-key',
    'QDRANT_URL': 'https://test.qdrant.io',
    'QDRANT_API_KEY': 'test-qdrant-key',
}

for key, value in _REQUIRED_ENV_STUBS.items():
    if not os.environ.get(key):
        os.environ[key] = value
