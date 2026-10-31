def reciprocal_rank_fusion(semantic_results: list[dict], word_results: list[dict], rrf_constant: int = 60):
    if rrf_constant <= 0:
        raise ValueError("rrf_constant must be > 0")
    
    rrf_scores = {}

    for rank, result in enumerate(semantic_results, start=1):
        chunk_id = result["chunk_id"]
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (rrf_constant + rank)

    for rank, result in enumerate(word_results, start=1):
        chunk_id = result["chunk_id"]
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (rrf_constant + rank)

    results_by_id = {
        result["chunk_id"]: result.copy() for result in semantic_results + word_results
    }

    for chunk_id, score in rrf_scores.items():
        results_by_id[chunk_id]["rrf"] = score

    return sorted(
            results_by_id.values(),
            key=lambda result: result["rrf"],
            reverse=True
    )

if __name__ == "__main__":
    semantic_results = [{"similarity": 0.5, "page": 1, "text": "Hello world", "chunk_id": 10}]
    word_results = [{"chunk_id": 30}, {"score": 0.4, "page": 2, "text": "Machine Learning", "chunk_id": 20}]
    print(reciprocal_rank_fusion(semantic_results, word_results))


"""
results:
'similarity': 0.5, 
'page': 1, 
'text': 'Hello world', 
'chunk_id': 1, 
'rrf': 0.01639344262295082
"""


"""
semantic
"similarity": value.item(),
"page": self.chunks[ind]["page"],
"text": self.chunks[ind]["text"],
"chunk_id": ind
"""


"""
word
"score": float(scores[ind]),
"page": self.chunks[ind]["page"],
"text": self.chunks[ind]["text"],
"chunk_id": int(ind)
"""