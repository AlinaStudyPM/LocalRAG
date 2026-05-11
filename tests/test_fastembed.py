# tests/test_fastembed.py
from fastembed import TextEmbedding

def create_chunks(texts: list[str]) -> list[str]:
    model = TextEmbedding(model_name="BAAI/bge-base-en-v1.5")
    generator = model.embed(texts)
    chunks = list(generator)
    return chunks

if __name__ == "__main__":
    texts = [
            "epodcs;hljbkns,ad heui hdlaew",
            "qweadfsbtgrhwyaewfdfvgtbqefrf",
            "a3wefvr"
    ]
    print(create_chunks(texts))
