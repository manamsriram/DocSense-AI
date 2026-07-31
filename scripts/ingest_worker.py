"""Ingest one PDF on a GitHub Actions runner (7GB RAM) instead of the 512MB Render dyno.

Invoked by .github/workflows/ingest.yml on a repository_dispatch of type `ingest_pdf`.
Reads the payload from argv, downloads the PDF from Supabase Storage, runs the same
ingestion pipeline /upload used to run inline, and flips the documents row to
status='ready' (or 'failed').

Usage: python scripts/ingest_worker.py <storage_path> <user_id> <filename> [display_name]
"""
import logging
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app  # noqa: E402  (env must be loaded before import; app.py does that itself)

logging.basicConfig(level=logging.INFO)


def main():
    if len(sys.argv) < 4:
        sys.exit('usage: ingest_worker.py <storage_path> <user_id> <filename> [display_name]')
    storage_path, user_id, filename = sys.argv[1:4]
    display_name = sys.argv[4] if len(sys.argv) > 4 else filename

    def set_status(fields):
        (app.supabase_admin.table('documents').update(fields)
         .eq('user_id', user_id).eq('filename', filename).execute())

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    try:
        pdf_bytes = app.supabase_admin.storage.from_('pdfs').download(storage_path)
        with open(tmp.name, 'wb') as f:
            f.write(pdf_bytes)

        count = app.index_pdf(tmp.name, user_id, force=True, display_name=display_name)
        app.rebuild_bm25_for_user(user_id, source_filename=display_name)
        app.rebuild_graph_for_user(user_id, source_filename=display_name)
        app.bump_kb_version(user_id)
        app.check_qdrant_shard_threshold()

        set_status({'chunk_count': count, 'status': 'ready'})
        logging.info(f"Indexed {count} chunks from {filename}")
    except Exception as e:
        logging.error(f"Ingestion failed for {storage_path}: {e}", exc_info=True)
        set_status({'status': 'failed'})
        raise
    finally:
        os.unlink(tmp.name)


if __name__ == '__main__':
    main()
