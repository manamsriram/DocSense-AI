import gc
import os
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


@app.route('/rerank', methods=['POST'])
@require_secret
def rerank():
    if _reranker_model is None:
        return jsonify({'error': f'this instance is role={MODEL_ROLE}, not rerank'}), 404
    body = request.get_json(force=True)
    query = body['query']
    documents = body['documents']
    scores = [float(s) for s in _reranker_model.rerank(query, documents)]
    gc.collect()
    return jsonify({'scores': scores})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 10000)))
