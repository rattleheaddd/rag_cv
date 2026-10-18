import torch

class InMemoryRetriever:
    def __init__(self, chunks, embeddings):
        if len(chunks) == 0:
            raise ValueError("len chunks must be > 0")
        if len(chunks) != len(embeddings):
            raise ValueError("len chunks must be equal to len embeddings")
        self.embeddings = embeddings
        self.chunks = chunks

    def search(self, query_embedding, k: int = 3):
        if k <= 0:
            raise ValueError("k must be > 0")
        scores = self.embeddings @ query_embedding
        k = min(k, len(self.chunks))
        values, indices = torch.topk(scores, k)
        results = []

        for value, ind in zip(values, indices):
            ind = ind.item()
            results.append({
                "similarity": value.item(),
                "page": self.chunks[ind]["page"],
                "text": self.chunks[ind]["text"],
                "chunk_id": ind
            })

        return results