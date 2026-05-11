
# src/ChromaAdapter.py
from typing import List, Dict, Any
import uuid

import chromadb
import ollama

from fastembed import TextEmbedding

from src.Config import Config

class ChromaAdapter:
    """
    Работа с Chroma: коллекции, эмбеддинги, поиск.
    """
    def __init__(self, config: Config):
        """
        Инициализирует адаптер ChromaDB.
        """
        self.config = config
        self._client = chromadb.PersistentClient(path=self.config.CHROMA_DB_DIR)        
        self.ollama_client = ollama.Client(host=self.config.OLLAMA_LOCAL_URL)
        self.embedding_model = TextEmbedding(
            model_name=config.EMBEDDING_MODEL,
            cache_dir="./.fastembed_cache"
        )


    def add_documents(self, collection_name: str, file_name: str, texts: List[str]) -> None:
        """
        Добавляет список файлов в указанную коллекцию Chroma.

        Генерирует уникальные UUID для каждого документа. Документы помечаются метаданными
        (например, источником — именем файла), что позволяет в дальнейшем фильтровать/удалять по нему.
        """
        coll = self._client.get_collection(name=collection_name)
        ids = [str(uuid.uuid4()) for _ in range(len(texts))]
        embeddings = self.generate_embeddings(texts)
        metadatas = [
            {
                "source": file_name,
                "serial_number": i + 1
            } for i in range(len(texts))
        ]
        coll.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=texts)

    # Реранкер?
    # Установить нижнюю границу похожести
    def search(self, collection_name: str, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Выполняет семантический поиск по коллекции.
        """
        coll = self._client.get_collection(name=collection_name)
        query_emb = self.generate_embeddings([query])[0]
        result = coll.query(query_embeddings=[query_emb], n_results=top_k)
        return result

    def list_files(self, collection_name: str) -> List[str]:
        """
        Возвращает отсортированный список уникальных имён файлов,
        присутствующих в коллекции.
        """
        coll = self._client.get_collection(name=collection_name)
        metadatas = coll.get(include=["metadatas"])["metadatas"]
        sources = {m.get("source") for m in metadatas if m.get("source")}
        return sorted(sources)

    def is_in_collection(self, collection_name: str, file_name: str) -> bool:
        """
        Проверяет, содержится ли файл в коллекции.
        """
        coll = self._client.get_collection(name=collection_name)
        hits = coll.get(where={"source": file_name}, limit=1)
        return len(hits["ids"]) > 0

    def delete(self, collection_name: str, file_name: str) -> None:
        """
        Удаляет все записи, связанные с указанным файлом, из коллекции.
        """
        coll = self._client.get_collection(name=collection_name)
        coll.delete(where={"source": file_name})

    def generate_embeddings(self, texts: List[str], batch_size: int = 8):
        """
        Генерирует эмбеддинги для списка текстов с помощью предобученной модели.
        """
        embedding_generator = self.embedding_model.embed(texts)
        embeddings = list(embedding_generator)
        return embeddings
    
    def create_collection(self, collection_name: str) -> None:
        """
        Создаёт коллекцию, если она ещё не существует.
        """
        try:
            self._client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        except ValueError as e:
            if "already exists" not in str(e):
                raise

