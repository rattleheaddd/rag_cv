from textpreparer import TextPreparer
from e5_embedder import E5Embedder
from llm import LocalModel
from transformers import AutoTokenizer, AutoModel
from context import build_context
from retriever import InMemoryRetriever

if __name__ == "__main__":    
    model = AutoModel.from_pretrained("intfloat/multilingual-e5-small", device_map="cuda", local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-small", local_files_only=True)
    prep = TextPreparer(tokenizer=tokenizer, chunk_size=256, overlap=50)
    chunks = prep.prepare("docs/document.pdf")
    embed = E5Embedder(model=model, tokenizer=tokenizer, batch_size=16)
    chunk_embeddings = embed.embed_chunks(chunks=chunks)
    retriever = InMemoryRetriever(chunks=chunks, embeddings=chunk_embeddings)
    llm = LocalModel(model="qwen3:4b-instruct")
    while True:
        question = input("Введите вопрос (exit для выхода): ")
        if question.lower() == "exit":
            break
        query_embedding = embed.embed_query(question)
        results = retriever.search(query_embedding=query_embedding, k=3)
        context = build_context(results)
        print(llm.generate_answer(context, question))