from markitdown import MarkItDown

def process_docx(file_path) -> str:
    md = MarkItDown()
    result = md.convert(file_path)
    return result.text_content

if __name__ == "__main__":
    text = process_docx("fixtures/test.docx")
    print(text)
