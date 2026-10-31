from textpreparer import TextPreparer
from e5_embedder import E5Embedder
from llm import LocalModel
from transformers import AutoTokenizer, AutoModel
from context import build_context
from semantic_retriever import InMemoryRetriever
from bm25_retriever import BM25Retriever
from evaluation import evaluate, recall_at_k, reciprocal_rank
from hybrid_retriever import HybridRetriever
from semantic_retriever import InMemoryRetriever

if __name__ == "__main__":
    questions = [
        # "What is LayoutLM and how does it work?",
        # "How are graph neural networks used in document understanding?",
        "Which datasets are used for document layout analysis?",
        # "What is the difference between LayoutLM and LayoutLMv2?",
        # "How is OCR used in document information extraction?"
    ]
    model = AutoModel.from_pretrained("intfloat/multilingual-e5-small", device_map="cuda", local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-small", local_files_only=True)
    prep = TextPreparer(tokenizer=tokenizer, chunk_size=128, overlap=32)
    chunks = prep.prepare("docs/document.pdf")
    embed = E5Embedder(model=model, tokenizer=tokenizer, batch_size=16)
    chunk_embeddings = embed.embed_chunks(chunks=chunks)
    bm25_retriever = BM25Retriever(chunks=chunks)
    semantic_retriever = InMemoryRetriever(chunks, chunk_embeddings)
    hybrid_retriever = HybridRetriever(semantic_retriever, bm25_retriever)
    relevant_ids = [83, 84, 24]

    for chunk_id in [23, 25, 82, 85]:
        print(f"\nCHUNK {chunk_id}")
        print(chunks[chunk_id]["text"])

    # for query in questions:
    #     query_embedding = embed.embed_query(query)
    #     results_e5 = semantic_retriever.search(query_embedding, k=10)
    #     results_bm25 = bm25_retriever.search(query, k=10)
    #     for result in results_e5:
    #         print(result["chunk_id"])
    #     print()
    #     for result in results_bm25:
    #         print(result["chunk_id"])
    #     print()
        # context = build_context(results)
        # llm = LocalModel(model="qwen3:4b-instruct")
        # print(llm.generate_answer(context, query))
    