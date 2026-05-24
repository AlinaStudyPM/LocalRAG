# src/ChromaAdapter.py
from typing import List, Dict, Any, Optional
import uuid

import chromadb
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder

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
        self.embedding_model = TextEmbedding(
            model_name=config.EMBEDDING_MODEL,
            cache_dir="./.fastembed_cache",
        )
        self.reranker = TextCrossEncoder(
            model_name=config.RETRIEVAL_MODEL,
            cache_dir="./.fastembed_cache",
        )

    def add_documents(self, collection_name: str, file_name: str, texts: List[str]) -> None:
        """
        Добавляет список файлов в указанную коллекцию Chroma.
        Обрабатывает батчами по 32 документа, чтобы не перегружать CPU/RAM.
        """
        coll = self._client.get_collection(name=collection_name)
        batch_size = 32
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_ids = [str(uuid.uuid4()) for _ in range(len(batch_texts))]
            batch_embeddings = self.generate_embeddings(batch_texts)
            batch_metadatas = [
                {
                    "source": file_name,
                    "serial_number": i + j + 1
                } for j in range(len(batch_texts))
            ]
            coll.upsert(
                ids=batch_ids,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas,
                documents=batch_texts
            )

    def search(self, 
               collection_name: str, 
               query: str, 
               top_embed: Optional[int] = None,
               top_rerank: Optional[int] = None,
               use_rerank: bool = True
        )-> Dict[str, Any]:
        """
        Выполняет семантический поиск по коллекции.
        """
        top_embed = top_embed if top_embed is not None else self.config.top_embed
        top_rerank = top_rerank if top_rerank is not None else self.config.top_rerank

        coll = self._client.get_collection(name=collection_name)
        query_emb = self.generate_embeddings([query])[0]
        
        candidates = coll.query(
            query_embeddings=[query_emb], 
            n_results=top_embed
        )
        candidate_docs = candidates['documents'][0]
        if not candidate_docs:
            return {
                "ids": [[]],
                "documents": [[]],
                "metadatas": [[]],
                "scores": [],
                "query": query
            }

        if not use_rerank:
            return {
                'ids': candidates['ids'],
                'documents': candidates['documents'],
                'metadatas': candidates['metadatas'],
                'scores': [],
                'query': query
            }
        else:
            if top_rerank > top_embed:
                top_rerank = top_embed


            rerank_scores = self._rerank_in_batches(
                query=query,
                documents=candidate_docs,
                batch_size=3
            )

            scored_candidates = sorted(
                zip(rerank_scores, candidate_docs, candidates['ids'][0], candidates['metadatas'][0]),
                key=lambda x: x[0],
                reverse=True
            )
            top_n = scored_candidates[:top_rerank]
            return {
                'ids': [[item[2] for item in top_n]],
                'documents': [[item[1] for item in top_n]],
                'metadatas': [[item[3] for item in top_n]],
                'scores': [item[0] for item in top_n],
                'query': query
            }

    def _rerank_in_batches(self, query: str, documents: List[str], batch_size: int) -> List[float]:
        all_scores = []
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batch_scores = list(self.reranker.rerank(query, batch))
            all_scores.extend(batch_scores)
        return all_scores

    def list_files(self, collection_name: str) -> List[str]:
        """
        Возвращает отсортированный список уникальных имён файлов,
        присутствующих в коллекции.
        """
        try:
            coll = self._client.get_collection(name=collection_name)
            metadatas = coll.get(include=["metadatas"])["metadatas"]
            sources = {m.get("source") for m in metadatas if m.get("source")}
            return sorted(sources)
        except Exception:
            return [] 

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

    def generate_embeddings(self, texts: List[str], batch_size: int = 32):
        """
        Генерирует эмбеддинги для списка текстов батчами.
        """
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embedding_generator = self.embedding_model.embed(batch)
            batch_embeddings = list(embedding_generator)
            all_embeddings.extend(batch_embeddings)
        return all_embeddings
    
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
