class CrossEncoderReranker:
    def __init__(self, model):
        self.model = model

    def rerank(self, query: str, results: list[dict], k: int = 5):
        if k <= 0:
            raise ValueError("k must be > 0")
        if results == []:
            return []
        
        pairs = [(query, result["text"]) for result in results]
        predictions = self.model.predict(pairs)
        scored_results = []
        for result, prediction in zip(results, predictions):
            new_result = result.copy()
            new_result["rerank_logit"] = float(prediction)
            scored_results.append(new_result)

        return sorted(
            scored_results,
            key=lambda result: result["rerank_logit"],
            reverse=True
        )[:k]