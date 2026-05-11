# tests/test_batch_process.py

import os
import hashlib
from pathlib import Path
from typing import List, Generator
import sys

from pypdf import PdfReader
from fastembed import TextEmbedding
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

FILE_PATH = ""
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
BATCH_SIZE = 50
CHROMA_DB_PATH = ""
COLLECTION_NAME = "test_batch"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=tiktoken_len
        )
# 1. Чтение текста из файла
def _process_text(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    return text

# 2. Разбиение на чанки (желательно, без langchain)
def _split_to_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
    chunks = []
    separators = ["\n\n", "\n", ". ", " ", ""]
    curr_start = 0
    while curr_start < len(text):
        curr_end = min(curr_start + chunk_size, len(text))
        chunk = text[curr_start:curr_end]
        # Стараемся обрезать по пробелу
        if len(chunk) == chunk_size and curr_end < len(text):
            last_space = chunk.rfind(' ')
            if last_space > chunk_size // 2:
                chunk = chunk[:last_space]
                curr_end = curr_start + last_space 
        chunks.append(chunk)
        curr_start = curr_end - overlap
        if start <= 0:
            start = chunk_size - overlap
    return chunks


# 3. Создание эмбеддингов
def create_embeddings(texts: List[str]) -> Generator:
    embedding_model = TextEmbedding(model_name=EMBEDDING_MODEL)
    return embedding_model.embed(texts)

# 4. Загрузка в базу данных
def upload_to_chroma(collection_name: str, texts: List[str]) -> None:
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    coll = client.get_or_create_collection(collection_name)
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        embeddings_gen = create_embeddings(batch)

        ids = []
        documents = []
        embeddings = []

        for j, (text, emb) in enumerate(zip(batch, embeddings_gen)):
            doc_id = hashlib.md5(text.encode()).hexdigest()[:16]
            ids.append(doc_id)
            documents.append(text)
            embeddings.append(emb.tolist())
        
        # Пакетная запись в Chroma
        collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=[{"index": i + j} for j in range(len(batch))]
        )

if __name__ == "__main__":
    text = _process_text(FILE_PATH)
    chunks = _split_to_chunks(text, CHUNK_SIZE, CHUNK_OVERLAP)
    upload_to_chroma(chunks)
