import torch

class InMemoryRetriever:
    def __init__(self, chunks, embeddings, device="cuda"):
        if len(chunks) == 0:
            raise ValueError("len chunks must be > 0")
        if len(chunks) != len(embeddings):
            raise ValueError("len chunks must be equal to len embeddings")
        
        self.chunks = chunks
        self.device = device

        self.embeddings = embeddings.to(
            device=self.device,
            dtype=torch.float32
        )

    def search(self, query_embedding, k: int = 3):
        if k <= 0:
            raise ValueError("k must be > 0")

        query_embedding = query_embedding.to(
            device=self.device,
            dtype=self.embeddings.dtype
        )
        scores = self.embeddings @ query_embedding
        k = min(k, len(self.chunks))
        values, indices = torch.topk(scores, k)
        values = values.cpu().tolist()
        indices = indices.cpu().tolist()
        results = []

        for value, ind in zip(values, indices):
            results.append({
                "similarity": value,
                "page": self.chunks[ind]["page"],
                "text": self.chunks[ind]["text"],
                "chunk_id": ind
            })

        return results