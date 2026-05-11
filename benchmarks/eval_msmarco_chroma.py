# benchmarks/eval_msmarco_chroma.py
"""
Оценка полноценного RAG через ChromaDB с загрузкой MS MARCO данных.
"""
import asyncio
import copy
import os
import shutil
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
from openai import AsyncOpenAI
from ragas import evaluate
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
from ragas.llms import llm_factory
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics._faithfulness import faithfulness
from ragas.metrics._context_precision import context_precision
from ragas.metrics._context_recall import context_recall
from ragas.metrics._answer_relevance import AnswerRelevancy
from ragas.metrics._answer_similarity import SemanticSimilarity
from ragas.run_config import RunConfig
from langchain_community.embeddings import OllamaEmbeddings
# Загружаем .env из папки benchmarks
load_dotenv("benchmarks/.env", override=True)
# Импорты после загрузки .env
from benchmarks.data_utils import load_msmarco_sample
from src.ChromaAdapter import ChromaAdapter
from src.Config import Config as AppConfig
def get_llm(model_name: str = None):
    """Создать LLM для оценки."""
    if model_name is None:
        model_name = os.getenv("EVAL_MODEL", "llama3.2:latest")
    
    client = AsyncOpenAI(
        api_key="ollama",
        base_url=f"{os.getenv('OLLAMA_LOCAL_URL', 'http://localhost:11434')}/v1"
    )
    return llm_factory(model_name, client=client, provider="openai")
def get_async_llm_client():
    """Получить AsyncOpenAI клиент для генерации ответов."""
    return AsyncOpenAI(
        api_key="ollama",
        base_url=f"{os.getenv('OLLAMA_LOCAL_URL', 'http://localhost:11434')}/v1"
    )
def get_embeddings():
    """Создать эмбеддинги для ChromaDB и RAGAS."""
    embedding_model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    ollama_url = os.getenv("OLLAMA_LOCAL_URL", "http://localhost:11434")
    
    return LangchainEmbeddingsWrapper(
        OllamaEmbeddings(
            model=embedding_model,
            base_url=ollama_url
        )
    )
def load_passages_to_chroma(chroma_adapter, collection_name: str, max_passages: int = 1000):
    """Загрузить passage-и из MS MARCO в ChromaDB."""
    print(f"Загрузка до {max_passages} passages из MS MARCO...")
    
    data = load_msmarco_sample(n=max_passages)
    
    all_passages = []
    for item in data:
        contexts = item.get("contexts", [])
        if isinstance(contexts, list):
            all_passages.extend(contexts)
        else:
            all_passages.append(contexts)
    
    # Уникальные passage
    unique_passages = list(set(all_passages))
    
    # Ограничиваем
    unique_passages = unique_passages[:max_passages]
    
    print(f"Уникальных passages: {len(unique_passages)}")
    
    # Проверяем существование коллекции
    existing_collections = [c.name for c in chroma_adapter._client.list_collections()]
    
    if collection_name in existing_collections:
        print(f"Коллекция {collection_name} уже существует")
        # Получаем количество документов
        coll = chroma_adapter._client.get_collection(collection_name)
        count = coll.count()
        print(f"Документов в коллекции: {count}")
        
        if count > 0:
            print("Используем существующую коллекцию")
            return count
    else:
        chroma_adapter.create_collection(collection_name)
        print(f"Создана новая коллекция: {collection_name}")
    
    # Добавляем документы
    print("Добавление документов в ChromaDB...")
    batch_size = 100
    for i in range(0, len(unique_passages), batch_size):
        batch = unique_passages[i:i+batch_size]
        
        # Добавляем батчами (генерируем ID)
        import uuid
        ids = [str(uuid.uuid4()) for _ in range(len(batch))]
        
        # Генерируем эмбеддинги
        embeddings = chroma_adapter.generate_embeddings(batch)
        
        # Метаданные
        metadatas = [{"source": "ms_marco", "index": i + j} for j in range(len(batch))]
        
        coll = chroma_adapter._client.get_collection(collection_name)
        coll.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=batch
        )
        
        print(f"  Добавлено {min(i+batch_size, len(unique_passages))}/{len(unique_passages)}")
    
    coll = chroma_adapter._client.get_collection(collection_name)
    print(f"Итого документов: {coll.count()}")
    
    return coll.count()
def search_chroma(chroma_adapter, collection_name: str, question: str, top_k: int = 5) -> list[str]:
    """Найти релевантные документы в ChromaDB."""
    try:
        coll = chroma_adapter._client.get_collection(collection_name)
        query_emb = chroma_adapter.generate_embeddings([question])[0]
        
        result = coll.query(
            query_embeddings=[query_emb],
            n_results=top_k
        )
        
        documents = result.get("documents", [[]])[0]
        return documents
        
    except Exception as e:
        print(f"Ошибка при поиске: {e}")
        return []
async def generate_answer(client, model: str, question: str, contexts: list[str]) -> str:
    """Сгенерировать ответ используя контексты."""
    if not contexts:
        return "Нет контекста для ответа."
    
    context_text = "\n\n".join([f"Документ {i+1}:\n{ctx}" for i, ctx in enumerate(contexts)])
    
    prompt = f"""Используя предоставленные документы, ответь на вопрос.
Документы:
{context_text}
Вопрос: {question}
Ответ:"""
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Ошибка при генерации: {e}")
        return f"Ошибка: {e}"
async def run_evaluation():
    """Основная функция оценки."""
    # Параметры из .env
    chroma_dir = os.getenv("CHROMA_DIR", "./benchmarks/data/chroma_benchmark")
    collection_name = os.getenv("CHROMA_COLLECTION", "ms_marco_passages")
    max_passages = int(os.getenv("MAX_PASSAGES", "1000"))
    max_questions = int(os.getenv("MAX_QUESTIONS", "10"))
    top_k = int(os.getenv("TOP_K", "5"))
    results_dir = os.getenv("RESULTS_DIR", "./benchmarks/data")
    generate_model = os.getenv("GENERATE_MODEL", "llama3.2:latest")
    eval_model = os.getenv("EVAL_MODEL", "llama3.2:latest")
    timeout = int(os.getenv("TIMEOUT", "600"))
    
    print("=" * 60)
    print("ОЦЕНКА RAG ЧЕРЕЗ CHROMADB")
    print("=" * 60)
    print(f"Модель для генерации: {generate_model}")
    print(f"Модель для оценки: {eval_model}")
    print(f"Max passages: {max_passages}")
    print(f"Max questions: {max_questions}")
    print(f"Top-K: {top_k}")
    print(f"Chroma dir: {chroma_dir}")
    print(f"Collection: {collection_name}")
    print()
    
    # Создаём директорию для Chroma
    Path(chroma_dir).mkdir(parents=True, exist_ok=True)
    
    # Инициализируем ChromaDB
    print("Инициализация ChromaDB...")
    app_config = AppConfig()
    # Подменяем путь к Chroma
    original_chroma_dir = app_config.CHROMA_DB_DIR
    app_config.CHROMA_DB_DIR = chroma_dir
    
    chroma_adapter = ChromaAdapter(app_config)
    
    # Загружаем passages в Chroma
    doc_count = load_passages_to_chroma(chroma_adapter, collection_name, max_passages)
    
    if doc_count == 0:
        print("ОШИБКА: Не удалось загрузить документы в ChromaDB")
        return None
    
    # Загружаем вопросы
    print(f"\nЗагрузка {max_questions} вопросов из MS MARCO...")
    questions_data = load_msmarco_sample(n=max_questions)
    print(f"Загружено {len(questions_data)} вопросов")
    
    # Клиент для генерации
    client = get_async_llm_client()
    
    # Выполняем RAG
    print(f"\nВыполнение RAG запросов...")
    eval_samples = []
    
    for i, item in enumerate(questions_data):
        question = item["question"]
        ground_truth = item["answer"]
        
        # Поиск в Chroma
        contexts = search_chroma(chroma_adapter, collection_name, question, top_k)
        
        if not contexts:
            print(f"  [{i+1}] Нет контекстов для: {question[:50]}...")
            continue
        
        # Генерация ответа
        generated_answer = await generate_answer(client, generate_model, question, contexts)
        
        print(f"  [{i+1}] {question[:50]}...")
        print(f"       Контекстов: {len(contexts)}")
        
        sample = SingleTurnSample(
            user_input=question,
            response=generated_answer,
            retrieved_contexts=contexts,
            reference=ground_truth,
        )
        eval_samples.append(sample)
    
    if not eval_samples:
        print("Нет данных для оценки!")
        return None
    
    dataset = EvaluationDataset(eval_samples)
    print(f"\nДатасет: {len(dataset)} сэмплов")
    
    # Создаём LLM и эмбеддинги для RAGAS
    print("\nПодготовка метрик RAGAS...")
    llm = get_llm(eval_model)
    embeddings = get_embeddings()
    
    metrics = [
        copy.deepcopy(faithfulness),
        copy.deepcopy(context_precision),
        copy.deepcopy(context_recall),
        AnswerRelevancy(llm=llm, embeddings=embeddings),
        SemanticSimilarity(embeddings=embeddings),
    ]
    
    # Присваиваем llm
    metrics[0].llm = llm
    metrics[1].llm = llm
    metrics[2].llm = llm
    
    print(f"Метрики: {[m.name for m in metrics]}")
    
    # Конфиг
    run_config = RunConfig(timeout=timeout, max_workers=1)
    
    print("\nЗапуск оценки RAGAS...")
    results = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        run_config=run_config,
        show_progress=True,
    )
    
    # Сохраняем результаты
    output_dir = Path(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"chroma_eval_{timestamp}.csv"
    
    results.to_pandas().to_csv(output_path, index=False)
    print(f"\nРезультаты сохранены: {output_path}")
    
    # Копия базы данных
    backup_path = output_dir / f"chroma_backup_{timestamp}"
    shutil.copytree(chroma_dir, backup_path)
    print(f"Копия базы: {backup_path}")
    
    return results
if __name__ == "__main__":
    asyncio.run(run_evaluation())
