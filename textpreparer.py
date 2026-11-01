import pymupdf
import unicodedata
import re
import pysbd
from lingua import LanguageDetectorBuilder

class TextPreparer:
    def __init__(self, tokenizer, chunk_size: int = 256, overlap: int = 50):
        if chunk_size <= 0: 
            raise ValueError("chunk size must be > 0")
        if overlap >= chunk_size or overlap < 0: 
            raise ValueError("overlap must be < chunk_size and >= 0")
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.language_detector = LanguageDetectorBuilder.from_all_languages().build()

    def detect_language(self, blocks: list[dict]) -> str:
        text = " ".join(block["text"] for block in blocks)
        language = self.language_detector.detect_language_of(text[:5000])

        if language is None:
            raise ValueError("Could not detect document language")

        return language.iso_code_639_1.name.lower()

    def clean_text(self, text: str) -> str:
        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r"-[ \t]*\n[ \t]*", "", text)     # убираем '-\n' и пробелы с обеих сторон
        text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)    # убираем пробелы вокруг '\n'
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)    # заменяем '\n' на пробелы
        text = re.sub(r"[^\S\n]+", " ", text)           # убираем лишние пробелы, оставляя только один

        return text.strip()
    
    def extract_blocks(self, pdf_path):
        with pymupdf.open(pdf_path) as doc:            
            results = []
            for page in doc:
                blocks = page.get_text("blocks")
                for block in blocks:
                    if block[6] != 0:
                        continue
                    text = self.clean_text(block[4])
                    if not text:
                        continue
                    results.append({
                        "page": page.number + 1,
                        "text": text,
                        "bbox": block[:4]
                    })

            return results

    def split_long_block(self, segmenter, block: dict) -> list[dict]:

        def add_to_results(results, current_block):
            results.append({
                "page": block["page"],
                "text": " ".join(current_block)
            })

        sentences = segmenter.segment(block["text"])
        results = []
        current_block = []
        count_tokens = 0
        for sentence in sentences:
            token_ids = self.tokenizer.encode(sentence, add_special_tokens=False)
            len_sentence = len(token_ids)
            if len_sentence > self.chunk_size:
                if current_block:
                    add_to_results(results, current_block)
                    current_block = []
                    count_tokens = 0
                for i in range(0, len_sentence, self.chunk_size):
                    part_ids = token_ids[i : i + self.chunk_size]
                    part_text = self.tokenizer.decode(part_ids)
                    add_to_results(results, [part_text])
            else:
                if count_tokens + len_sentence > self.chunk_size:
                    add_to_results(results, current_block)
                    current_block = [sentence]
                    count_tokens = len_sentence
                else:
                    current_block.append(sentence)
                    count_tokens += len_sentence
        if current_block:
            add_to_results(results, current_block)

        return results

    def chunk_blocks(self, segmenter, blocks: list[dict]) -> list[dict]:
        
        def add_to_results(results: list[dict], current_blocks: list[dict]):
            text = "\n\n".join(current_block["text"] for current_block in current_blocks)
            results.append({
                "page": current_blocks[0]["page"],
                "text": text
            })
        
        results = []
        current_blocks = []
        len_tokens = 0
        for block in blocks:
            len_block_tokens = len(self.tokenizer.encode(block["text"], add_special_tokens=False))
            if len_block_tokens > self.chunk_size:
                if current_blocks:
                    add_to_results(results, current_blocks)
                    current_blocks = []
                    len_tokens = 0
                block_results = self.split_long_block(segmenter, block)
                results.extend(block_results)
                continue
            if current_blocks and (block["page"] != current_blocks[0]["page"]):
                add_to_results(results, current_blocks)
                current_blocks = []
                len_tokens = len_block_tokens
            else:
                len_tokens += len_block_tokens
            if len_tokens <= self.chunk_size:
                current_blocks.append(block)
            else:
                if current_blocks:
                    add_to_results(results, current_blocks)
                current_blocks = []
                len_tokens = len_block_tokens
                current_blocks.append(block)
        if current_blocks:
            add_to_results(results, current_blocks)

        return results

    # def chunk_text_by_tokens(self, pages):
    #     chunks = []
    #     for page in pages:
    #         token_ids = self.tokenizer.encode(page["text"], add_special_tokens=False, truncation=False)
    #         begin = 0
    #         while begin < len(token_ids):
    #             chunk = {"page": page["page"],
    #                      "text": self.tokenizer.decode(token_ids[begin: begin + self.chunk_size])}
    #             chunks.append(chunk)
    #             if begin + self.chunk_size >= len(token_ids):
    #                 break
    #             begin += self.chunk_size - self.overlap

    #     return chunks

    def prepare(self, pdf_path: str) -> list[dict]:
        pages = self.extract_text(pdf_path)
        chunks = self.chunk_text_by_tokens(pages)
        return chunks

if __name__ == "__main__":
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-small", local_files_only=True)
    prep = TextPreparer(tokenizer)
    blocks = prep.extract_blocks("docs/document.pdf")
    language = prep.detect_language(blocks)
    segmenter = pysbd.Segmenter(language, clean=False)
    chunks = prep.chunk_blocks(segmenter, blocks)

    lengths = [
        len(prep.tokenizer.encode(
            chunk["text"],
            add_special_tokens=False
        ))
        for chunk in chunks
    ]

    print("Total chunks:", len(chunks))
    print("Max tokens:", max(lengths))
    print("Oversized chunks:", sum(n > prep.chunk_size for n in lengths))