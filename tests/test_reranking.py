from fastembed.rerank.cross_encoder import TextCrossEncoder

reranker = TextCrossEncoder(
    model_name="BAAI/bge-reranker-base",
    cache_dir="./.fastembed_cache",
)

documents = [
    "Paris is the capital of France.",
    "Berlin is the capital of Germany.",
    "Lyon is a city in France."
]
query = "What is the capital of France?"
scores = reranker.rerank(query, documents)
reranked_results = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)

for doc, score in reranked_results:
    print(f"Score: {score:.4f} - {doc}")
