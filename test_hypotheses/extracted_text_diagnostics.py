import fitz

with fitz.open("docs/document.pdf") as doc:
    for page_number in [7, 10]:
        page = doc[page_number - 1]

        original = page.get_text("text")
        sorted_text = page.get_text("text", sort=True)

        print(f"\n{'=' * 20} PAGE {page_number} {'=' * 20}")

        print("\nORIGINAL:")
        print(repr(original[:2500]))

        print("\nSORTED:")
        print(repr(sorted_text[:2500]))