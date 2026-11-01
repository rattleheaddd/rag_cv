class Qwen3Embedder:
    def __init__(self, model, batch_size=16):
        self.model = model
        self.batch_size = batch_size

    def embed_chunks(self, chunks):
        texts = [chunk["text"] for chunk in chunks]
        texts_embed = self.model.encode_document(
            texts, 
            convert_to_tensor=True, 
            normalize_embeddings=True, 
            batch_size=self.batch_size
        )
        
        return texts_embed.float()

    def embed_query(self, query):
        query_embed = self.model.encode_query(
            query, 
            convert_to_tensor=True, 
            normalize_embeddings=True
        )
        return query_embed.float()
