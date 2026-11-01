import pymupdf
import unicodedata
import re

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

    def clean_text(self, text: str) -> str:
        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r"-[ \t]*\n[ \t]*", "", text)     # убираем '-\n' и пробелы с обеих сторон
        text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)    # убираем пробелы вокруг '\n'
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)    # заменяем '\n' на пробелы
        text = re.sub(r"[^\S\n]+", " ", text)           # убираем лишние пробелы, оставляя только один
        
        return text.strip()

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

if __name__ == "__main__":
    # from transformers import AutoTokenizer

    # tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-small", local_files_only=True)
    # prep = TextPreparer(tokenizer)

    # pages = prep.extract_text("docs/document.pdf")
    
    import pymupdf

    with pymupdf.open("docs/document.pdf") as doc:
        page = doc[9]
        blocks = page.get_text("blocks")

        for i, block in enumerate(blocks):
            if block[6] != 0:
                continue

            print(f"\nBLOCK {i}")
            print(f"Coordinates: {block[:4]}")
            print(repr(block[4]))