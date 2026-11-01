import torch

def normalize_embedding(embedding):
    norms = torch.linalg.vector_norm(embedding, ord=2, dim=1, keepdim=True).clamp_min(1e-12)
    return embedding / norms

class E5Embedder:
    def __init__(self, model, tokenizer, batch_size: int = 16):
        if batch_size <= 0:
            raise ValueError("batch size must be > 0")
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.device = next(self.model.parameters()).device
    
    def mean_pooling(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_state = outputs["last_hidden_state"]
        attention_mask = inputs["attention_mask"]
        last_hidden_state = last_hidden_state * attention_mask.unsqueeze(-1)
        return last_hidden_state.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)

    def embed_chunks(self, chunks):
        if not chunks:
            raise ValueError("Cannot embed an empty list of chunks")
        
        with torch.inference_mode():
            batches = []
            for i in range(0, len(chunks), self.batch_size):
                batches.append(chunks[i:i + self.batch_size])
            embeddings = []
            self.model.eval()
            for batch in batches:
                batch_texts = ["passage: " + chunk["text"] for chunk in batch]
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).to(self.device)
                embedding = normalize_embedding(self.mean_pooling(inputs))
                embeddings.append(embedding.float())

            return torch.cat(embeddings, dim=0)

    def embed_query(self, query: str):
        self.model.eval()

        with torch.inference_mode():
            query = "query: " + query
            
            inputs = self.tokenizer(
                query,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)
            
            embedding = normalize_embedding(self.mean_pooling(inputs)).squeeze(0)

        return embedding.float()