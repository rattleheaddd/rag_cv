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
from qwen3embedder import Qwen3Embedder
from sentence_transformers import SentenceTransformer

if __name__ == "__main__":
    questions = [
        # Уже проверенные
        "What is LayoutLM and how does it work?",
        "How are graph neural networks used in document understanding?",
        "Which datasets are used for document layout analysis?",
        "What is the difference between LayoutLM and LayoutLMv2?",
        "How is OCR used in document information extraction?",

        # Простые фактологические
        "What are the two main subtasks of document layout analysis?",
        "What is visual information extraction in Document AI?",
        "What is document image classification?",
        "What is document visual question answering?",
        "What is the PubLayNet dataset and how was it constructed?",
        "How was the DocBank dataset created?",
        "What is the purpose of the DocVQA dataset?",
        "What is the difference between visual and semantic layout analysis?",

        # Технические вопросы
        "How does LayoutLM incorporate two-dimensional positional information?",
        "What is masked visual-language modeling?",
        "How does spatial-aware self-attention work in LayoutLMv2?",
        "How are CNNs applied to document layout analysis?",
        "How are OCR text blocks converted into a graph?",
        "How do graph convolutional networks improve information extraction?",
        "What role do Transformer architectures play in Document AI?",
        "How is multimodal information incorporated into document pre-training?",
        "What pre-training objectives are used in LayoutLMv2?",

        # Сравнения и вопросы на понимание
        "How does document visual question answering differ from traditional VQA?",
        "Why is document layout information important in addition to text?",
        "What are the limitations of traditional document information extraction methods?",
        "How do task-specific models differ from general-purpose pre-trained models?",
        "What challenges arise when processing visually rich documents?",
        "How do visual features and textual features complement each other?",

        # Проверка устойчивости RAG
        "What are the main limitations and open challenges in Document AI?",
        "What exact GPU hardware was used to train every model discussed in the paper?"
    ]
    #model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device="cuda", local_files_only=True)
    model = AutoModel.from_pretrained("intfloat/multilingual-e5-small", device_map="cuda", local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-small", local_files_only=True)
    prep = TextPreparer(tokenizer=tokenizer, chunk_size=128, overlap=32)
    chunks = prep.prepare("docs/document.pdf")
    embed = E5Embedder(model=model, tokenizer=tokenizer, batch_size=16)
    #embed = Qwen3Embedder(model)
    chunk_embeddings = embed.embed_chunks(chunks)
    bm25_retriever = BM25Retriever(chunks=chunks)
    semantic_retriever = InMemoryRetriever(chunks, chunk_embeddings)
    hybrid_retriever = HybridRetriever(semantic_retriever, bm25_retriever)
    relevant_ids = [24, 25, 83, 84]
    llm = LocalModel(model="qwen3:4b-instruct")

    import json

    test_results = []

    for question in questions:
        question_embedding = embed.embed_query(question)

        results = hybrid_retriever.search(
            question,
            question_embedding,
            k=10,
            candidate_k=20
        )

        context = build_context(results)
        answer = llm.generate_answer(context, question)

        test_results.append({
            "question": question,
            "retrieved_ids": [r["chunk_id"] for r in results],
            "answer": answer
        })

    with open("evaluation_e5_hybrid.json", "w", encoding="utf-8") as f:
        json.dump(test_results, f, ensure_ascii=False, indent=2)