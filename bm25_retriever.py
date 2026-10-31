from rank_bm25 import BM25Okapi
import re
import numpy as np


def word_tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class BM25Retriever:
    def __init__(self, chunks):
        if len(chunks) == 0:
            raise ValueError("len chunks must be > 0")
        
        self.chunks = chunks
        self.bm25 = BM25Okapi([word_tokenize(chunk["text"]) for chunk in self.chunks])

    def search(self, query: str, k: int = 5):
        if k <= 0:
            raise ValueError("k must be > 0")
        
        k = min(k, len(self.chunks))
        scores = self.bm25.get_scores(word_tokenize(query))
        indices = np.argsort(scores)[::-1][:k]
        results = []

        for ind in indices:
            score = float(scores[ind])
            if score > 0:
                results.append({
                    "score": score,
                    "page": self.chunks[ind]["page"],
                    "text": self.chunks[ind]["text"],
                    "chunk_id": int(ind)
                })

        return results

# if __name__ == "__main__":
    # from textpreparer import TextPreparer
#     questions = [
#         "What is LayoutLM?",
#         "Which datasets are used for document layout analysis?",
#         "How are graph neural networks applied to document understanding?"
#     ]

#     for question in questions:
#         model = AutoModel.from_pretrained("intfloat/multilingual-e5-small", device_map="cuda", local_files_only=True)
#         tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-small", local_files_only=True)
#         prep = TextPreparer(tokenizer=tokenizer, chunk_size=128, overlap=32)
#         chunks = prep.prepare("docs/document.pdf")
#         embed = E5Embedder(model=model, tokenizer=tokenizer, batch_size=16)
#         chunk_embeddings = embed.embed_chunks(chunks=chunks)
#         retriever = InMemoryRetriever(chunks=chunks, embeddings=chunk_embeddings)
#         llm = LocalModel(model="qwen3:4b-instruct")