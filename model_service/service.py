import gc
import os
import threading
from functools import wraps

from flask import Flask, request, jsonify

app = Flask(__name__)

SHARED_SECRET = os.environ['MODEL_SERVICE_SECRET']
CACHE_DIR = os.getenv('FASTEMBED_CACHE_PATH', None)

# MODEL_ROLE picks which single model this instance loads — each model gets
# its own dyno/memory budget instead of sharing one, which OOM'd a 512MB
# instance even under normal rerank batch sizes when both were loaded together.
MODEL_ROLE = os.environ['MODEL_ROLE']
assert MODEL_ROLE in ('embed', 'rerank'), "MODEL_ROLE must be 'embed' or 'rerank'"

_embedding_model = None
_reranker_model = None

if MODEL_ROLE == 'embed':
    from fastembed import TextEmbedding
    _embedding_model = TextEmbedding(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        cache_dir=CACHE_DIR,
        threads=1
    )
else:
    from fastembed.rerank.cross_encoder import TextCrossEncoder
    _reranker_model = TextCrossEncoder(
        model_name='Xenova/ms-marco-MiniLM-L-6-v2',
        cache_dir=CACHE_DIR,
        threads=1
    )


def require_secret(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if request.headers.get('X-Service-Secret') != SHARED_SECRET:
            return jsonify({'error': 'Unauthorized'}), 401
        return f(*args, **kwargs)
    return decorated


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'role': MODEL_ROLE}), 200


@app.route('/embed', methods=['POST'])
@require_secret
def embed():
    if _embedding_model is None:
        return jsonify({'error': f'this instance is role={MODEL_ROLE}, not embed'}), 404
    texts = request.get_json(force=True)['texts']
    vectors = [v.tolist() for v in _embedding_model.embed(texts)]
    gc.collect()
    return jsonify({'vectors': vectors})


RERANK_BATCH_SIZE = 3  # bounds peak ONNX arena size per call on a 512MB dyno

# gthread lets a health-check thread answer while another thread reranks, but
# two concurrent /rerank calls would each hold their own ONNX arena at once
# and blow past 512MB — serialize actual inference, /health stays unguarded.
_rerank_lock = threading.Lock()


@app.route('/rerank', methods=['POST'])
@require_secret
def rerank():
    if _reranker_model is None:
        return jsonify({'error': f'this instance is role={MODEL_ROLE}, not rerank'}), 404
    body = request.get_json(force=True)
    query = body['query']
    documents = body['documents']
    scores = []
    with _rerank_lock:
        for batch_num, i in enumerate(range(0, len(documents), RERANK_BATCH_SIZE)):
            batch = documents[i:i + RERANK_BATCH_SIZE]
            scores.extend(float(s) for s in _reranker_model.rerank(query, batch))
            # Collecting every sub-batch (10x for a 30-doc rerank at batch
            # size 3) was pushing hard, multi-iteration questions past
            # Render's ~30s gateway timeout. Every other batch still bounds
            # arena growth without doubling gc overhead.
            if batch_num % 2 == 1:
                gc.collect()
        gc.collect()
    return jsonify({'scores': scores})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 10000)))
