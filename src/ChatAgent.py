import json
import chromadb    # TODO: убрать связь с Chroma!!!
import requests
from typing import List, Dict, Any

from Config import Config
from ChromaAdapter import ChromaAdapter

class ChatAgent:
    """
    Класс для обещения пользователя с LLM
    """
    def __init__(self, config: Config, chroma_adapter: ChromaAdapter):
        self.config = config
        self.model = config.OLLAMA_MODEL
        self.chroma_adapter = chroma_adapter

    def chroma_query(self, collection_name: str, searched_text: str, top_k: int = 10):
        query_embedding = self.chroma_adapter.generate_embeddings([searched_text])[0]
        db_client = chromadb.PersistentClient(self.config.CHROMA_DB_DIR)
        collection = db_client.get_or_create_collection(name=collection_name)
        results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
        return results
    
    """
    def format_results(self, data: dict, n: int) -> str:
        documents = data['documents'][0]
        metadatas = data['metadatas'][0]
        result = [f"Топ {n} подходящих абзацев:", ""]
        for i, (doc, meta) in enumerate(zip(documents, metadatas), 1):
            source = meta.get('source', 'Неизвестный источник')
            result.append(f"{i}) Источник: {source}")
            result.append(f"    Текст: {doc}")
            result.append("")
        return "\n".join(result)
    """
    def format_results(self, data: dict, n: int) -> str:
        documents = data['documents'][0]
        metadatas = data['metadatas'][0] if data.get('metadatas') else [{}] * len(documents)

        # Группируем по источникам
        sources_dict = {}
        for doc, metadata in zip(documents, metadatas):
            source = metadata.get('source', 'Разное')
            if source not in sources_dict:
                sources_dict[source] = []
            sources_dict[source].append(doc)

        # Строим структурированный контекст
        context_lines = ["\n=== РЕЛЕВАНТНАЯ ИНФОРМАЦИЯ ===\n"]
        
        for source, docs in sources_dict.items():
            context_lines.append(f"\n--- {source.upper()} ---")
            for i, doc in enumerate(docs[:3], 1):  # Берем до 3 документов из каждого источника
                # Обрезаем слишком длинные документы
                if len(doc) > 1000:
                    doc = doc[:997] + "..."
                context_lines.append(f"\nФрагмент {i}: {doc}")
        
        context_lines.append("\n=== КОНЕЦ КОНТЕКСТА ===\n")
        
        return "\n".join(context_lines)
    
    def ollama_query(
            self, 
            input_text: str, 
            history: List[Dict[str, str]], 
            context: str = None,  
            temperature: float = 0.7, 
            max_history: int = 2
        ) -> str:

        messages = [{"role": "system", "content": self.config.SYSTEM_PROMPT}]

        if context:
            messages.append({
                "role": "user", 
                "content": f"""ИСПОЛЬЗУЙ ЭТУ ИНФОРМАЦИЮ ДЛЯ ОТВЕТА:
{context}
---
ОТВЕЧАЙ НА ОСНОВЕ ВЫШЕПРИВЕДЕННОЙ ИНФОРМАЦИИ"""
            })

        recent_history = history[-max_history*2:]
        messages.extend(recent_history)
        # messages.append({"role": "user", "content": input_text})
        # for m in messages:
        #   print(m)
        # print("\n\n\n")
        try:
            response = requests.post(
                f"{self.config.OLLAMA_URL}/api/chat",
                json={
                    "model": self.model, 
                    "messages": messages,
                    "temperature": temperature,
                    "stream": False
                }
            )
            response.raise_for_status()
            return response.json()["message"]["content"]
        except Exception as e:
            return f"Ошибка при запросе к Ollama: {str(e)}"
    
    def answer_question(self, question: str, chat_history, collection_names: List[str], top_k: int = 5):
        raw_results: List[Dict[str, Any]] = []
        for name in collection_names:
            raw = self.chroma_query(name, question, top_k=top_k)
            if ["documents"][0]:          # защита от пустой выдачи
                raw_results.append(raw)
        combined = {
            "documents": [[]],
            "metadatas": [[]],
        }
        for r in raw_results:
            combined["documents"][0].extend(r["documents"][0])
            combined["metadatas"][0].extend(r["metadatas"][0])

        #raw_results.sort(key=lambda x: x[1])
        #est_chunks = [txt for txt, _ in raw_results[:top_k]]

        context = self.format_results(combined, top_k * len(collection_names))
        print(context)
        return self.ollama_query(question, chat_history, context)
    
    def list_models(self) -> List[str]:
        """
        Возвращает список моделей, установленных в Ollama.
        """
        try:
            r = requests.get(f"{self.config.OLLAMA_URL}/api/tags", timeout=3)
            r.raise_for_status()
            return [m["name"] for m in r.json().get("models", [])]
        except Exception:
            return []

    


