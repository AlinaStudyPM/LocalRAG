from odfdo import Document

test_odt_path = "fixtures/test.odt"

def process_odt(file_path) -> str:
    document = Document(file_path)
    md_content = document.to_markdown()
    return md_content if md_content else ""

if __name__ == "__main__":
    text = process_odt(test_odt_path)
    print(text)
