# benchmarks/eval_msmarco_passages.py
"""
Оценка RAG на основе готовых релевантных текстов (passages).
"""
import asyncio
import copy
import os
from datetime import datetime
from pathlib import Path
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
from benchmarks import config
from benchmarks.data_utils import load_msmarco_sample
def get_llm():
    """Создать LLM для RAGAS на основе конфигурации."""
    provider = config.LLM_PROVIDER
    
    if provider == "ollama":
        client = AsyncOpenAI(
            api_key="ollama",
            base_url=f"{config.OLLAMA_URL}/v1"
        )
        return llm_factory(config.MODEL, client=client, provider="openai")
    else:
        raise ValueError(f"Unknown provider: {provider}")
def get_embeddings():
    """Создать эмбеддинги для RAGAS через Ollama."""
    embedding_model = os.getenv("EVAL_EMBEDDING_MODEL", "qwen3-embedding:0.6b")
    
    return LangchainEmbeddingsWrapper(
        OllamaEmbeddings(
            model=embedding_model,
            base_url=config.OLLAMA_URL
        )
    )
async def run_evaluation(n_samples: int = 10):
    print(f"Загрузка {n_samples} примеров из MS MARCO...")
    data = load_msmarco_sample(n=n_samples)
    print(f"Загружено {len(data)} примеров")
    
    # Создаём датасет
    eval_samples = []
    for item in data:
        contexts = item["contexts"] if isinstance(item["contexts"], list) else [item["contexts"]]
        
        sample = SingleTurnSample(
            user_input=item["question"],
            response=item["answer"],
            retrieved_contexts=contexts,
            reference=item["answer"],
        )
        eval_samples.append(sample)
    
    dataset = EvaluationDataset(eval_samples)
    print(f"Создан датасет с {len(dataset)} сэмплами")
    
    # Создаём LLM и эмбеддинги
    llm = get_llm()
    embeddings = get_embeddings()
    
    print(f"LLM: {config.MODEL}")
    print(f"Эмбеддинги: {os.getenv('EVAL_EMBEDDING_MODEL', 'qwen3-embedding:0.6b')}")
    
    # Создаём метрики - ВАЖНО: нужно копировать и присваивать llm/embeddings
    metrics = [
        copy.deepcopy(faithfulness),
        copy.deepcopy(context_precision),
        copy.deepcopy(context_recall),
        AnswerRelevancy(llm=llm, embeddings=embeddings),
        SemanticSimilarity(embeddings=embeddings),
    ]
    
    # Присваиваем llm для метрик которые его требуют
    metrics[0].llm = llm
    metrics[1].llm = llm
    metrics[2].llm = llm
    
    print(f"Метрики: {[m.name for m in metrics]}")
    
    # Конфиг с таймаутом
    run_config = RunConfig(timeout=600, max_workers=1)
    
    print("Запуск оценки RAGAS...")
    results = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        run_config=run_config,
        show_progress=True,
    )
    
    # Сохраняем результаты
    output_dir = Path(config.RESULTS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"passages_eval_{timestamp}.csv"
    
    results.to_pandas().to_csv(output_path, index=False)
    print(f"Результаты сохранены: {output_path}")
    
    return results
if __name__ == "__main__":
    asyncio.run(run_evaluation(n_samples=10))
