from textpreparer import TextPreparer
from e5_embedder import E5Embedder
from llm import LocalModel
from transformers import AutoTokenizer, AutoModel
from context import build_context
from retriever import InMemoryRetriever
from evaluation import evaluate, recall_at_k, reciprocal_rank

if __name__ == "__main__":    
    model = AutoModel.from_pretrained("intfloat/multilingual-e5-small", device_map="cuda", local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-small", local_files_only=True)
    prep = TextPreparer(tokenizer=tokenizer, chunk_size=128, overlap=32)
    chunks = prep.prepare("docs/document.pdf")
    embed = E5Embedder(model=model, tokenizer=tokenizer, batch_size=16)
    chunk_embeddings = embed.embed_chunks(chunks=chunks)
    retriever = InMemoryRetriever(chunks=chunks, embeddings=chunk_embeddings)
    llm = LocalModel(model="qwen3:4b-instruct")

    # test_cases = [
    #     {
    #         "question": "Какие проекты компьютерного зрения реализовал кандидат?",
    #         "relevant_ids": [1, 2]
    #     },
    #     {
    #         "question": "Где кандидат проходил практику?",
    #         "relevant_ids": [0]
    #     },
    #     {
    #         "question": "Какими разговорными языками владеет кандидат?",
    #         "relevant_ids": [3]
    #     },
    #     {
    #         "question": "Какие технологии использовались в Sudoku Solver?",
    #         "relevant_ids": [2]
    #     }
    # ]

    # for id, chunk in enumerate(chunks):
    #     print(id, chunk["text"])

    # for test_case in test_cases:
    #     question = test_case["question"]
    #     query_embedding = embed.embed_query(question)
    #     results = retriever.search(query_embedding, k=3)
    #     test_case["retrieved_ids"] = [result["chunk_id"] for result in results]

    question = "Какие проекты с использованием компьютерного зрения реализовал кандидат? Подробно опиши каждый. Объясни почему добавил проект в ответ."
    query_embedding = embed.embed_query(question)
    results = retriever.search(query_embedding, k=9)
    context = build_context(results)
    print(llm.generate_answer(context, question))