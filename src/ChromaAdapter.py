
# src/ChromaAdapter.py
from typing import List, Dict, Any
import uuid

import chromadb
import torch
from transformers import AutoTokenizer, AutoModel

from Config import Config

class ChromaAdapter:
    """
    Работа с Chroma: коллекции, эмбеддинги, поиск.
    """
    def __init__(self, config: Config):
        """
        Инициализирует адаптер ChromaDB.
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._client = chromadb.PersistentClient(path=self.config.CHROMA_DB_DIR)
        self._tokenizer = AutoTokenizer.from_pretrained(self.config.embeddings_model)
        self._model = AutoModel.from_pretrained(self.config.embeddings_model).to(self.device)

    def add_documents(self, collection_name: str, file_name: str, texts: List[str]) -> None:
        """
        Добавляет список файлов в указанную коллекцию Chroma.

        Генерирует уникальные UUID для каждого документа. Документы помечаются метаданными
        (например, источником — именем файла), что позволяет в дальнейшем фильтровать/удалять по нему.
        """
        # self.create_collection(collection_name)
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

    # Передавать список коллекций???
    # Какой тип возвращаемого значения?
    # Реранкер?
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
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = self._tokenizer(
                batch, 
                padding=True, 
                truncation=True, 
                return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                outputs = self._model(**inputs)
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.extend(batch_embeddings)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
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

