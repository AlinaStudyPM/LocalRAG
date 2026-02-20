#! tests/test_ollama_embeddings.py
"""
Тестовый скрипт для проверки эмбеддингов
    - qwen3-embedding (Ollama)
"""

import ollama
from typing import List

def get_embedding(text: str, model: str = "qwen3-embedding:0.6b") -> List[float]:
    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]
    
if __name__ == "__main__":
    text = "Трудности бытия!"
    print(f"Text: {text}")
    embedding = get_embedding(text)
    print(f"Embedding: {embedding}")
    print(f"Lenght: {len(embedding)}")

