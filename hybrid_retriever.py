from fusion import reciprocal_rank_fusion

class HybridRetriever:
    def __init__(self, semantic_retriever, bm25_retriever):
        self.semantic_retriever = semantic_retriever
        self.bm25_retriever = bm25_retriever

    def search(self, query: str, query_embedding, k: int = 5, candidate_k: int = 20):
        if k <= 0 or candidate_k <= 0:
            raise ValueError("k and candidate_k must be > 0")
        if candidate_k < k:
            raise ValueError("candidate_k must be >= k")
        
        semantic_results = self.semantic_retriever.search(query_embedding, candidate_k)
        word_results = self.bm25_retriever.search(query, candidate_k)
        results = reciprocal_rank_fusion(semantic_results, word_results)
        
        return results[:k]