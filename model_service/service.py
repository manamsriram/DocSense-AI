import os
from functools import wraps

from flask import Flask, request, jsonify
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder

app = Flask(__name__)

SHARED_SECRET = os.environ['MODEL_SERVICE_SECRET']
CACHE_DIR = os.getenv('FASTEMBED_CACHE_PATH', None)

_embedding_model = TextEmbedding(
    model_name='sentence-transformers/all-MiniLM-L6-v2',
    cache_dir=CACHE_DIR,
    threads=1
)
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
    return jsonify({'status': 'ok'}), 200


@app.route('/embed', methods=['POST'])
@require_secret
def embed():
    texts = request.get_json(force=True)['texts']
    vectors = [v.tolist() for v in _embedding_model.embed(texts)]
    return jsonify({'vectors': vectors})


@app.route('/rerank', methods=['POST'])
@require_secret
def rerank():
    body = request.get_json(force=True)
    query = body['query']
    documents = body['documents']
    scores = [float(s) for s in _reranker_model.rerank(query, documents)]
    return jsonify({'scores': scores})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 10000)))
