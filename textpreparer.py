import pymupdf

class TextPreparer:
    def __init__(self, tokenizer, chunk_size: int = 256, overlap: int = 50):
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.overlap = overlap
        if self.chunk_size <= 0: 
            raise ValueError("chunk size must be > 0")
        if self.overlap >= self.chunk_size or self.overlap < 0: 
            raise ValueError("overlap must be < chunk_size and >= 0")
    
    def extract_text(self, pdf_path):
        with pymupdf.open(pdf_path) as doc:
            return [{"page": page.number + 1, "text": page.get_text()} for page in doc]

    def chunk_text_by_tokens(self, pages):
        chunks = []
        for page in pages:
            token_ids = self.tokenizer.encode(page["text"], add_special_tokens=False, truncation=False)
            begin = 0
            while begin < len(token_ids):
                chunk = {"page": page["page"],
                         "text": self.tokenizer.decode(token_ids[begin: begin + self.chunk_size])}
                chunks.append(chunk)
                if begin + self.chunk_size >= len(token_ids):
                    break
                begin += self.chunk_size - self.overlap

        return chunks

    def prepare(self, pdf_path: str) -> list[dict]:
        pages = self.extract_text(pdf_path)
        chunks = self.chunk_text_by_tokens(pages)
        return chunks