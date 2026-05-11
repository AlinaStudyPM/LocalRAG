# benchmarks/eval_msmarco_passages.py
"""
Оценка RAG на основе готовых релевантных текстов (passages).
Не используется ChromaDB.
"""
import asyncio
import os
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict

from ragas import evaluate, Dataset
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    AnswerCorrectness,
    SemanticSimilarity,
)
from openai import AsyncOpenAI
import pandas as pd

from benchmarks import config
from benchmarks.data_utils import load_msmarco_sample

def get_llm():
    """Создать LLM для RAGAS на основе конфигурации."""
    
    provider = config.LLM_PROVIDER
    
    if provider == "openai":
        client = AsyncOpenAI(
            api_key=config.OPENAI_API_KEY,
            base_url="https://api.comet.com/v1"
        )
        return llm_factory(config.MODEL, client=client, provider="openai")
    
    elif provider == "openrouter":
        client = AsyncOpenAI(
            api_key=config.OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1"
        )
        return llm_factory(config.MODEL, client=client, provider="openrouter")
    
    elif provider == "ollama":
        client = AsyncOpenAI(
            api_key="ollama",
            base_url=f"{config.OLLAMA_URL}/v1"
        )
        return llm_factory(config.MODEL, client=client, provider="openai")
    
    else:
        raise ValueError(f"Unknown provider: {provider}")

async def run_evaluation(n_samples: int = 100):
    print(f"Загрузка {n_samples} примеров из MS MARCO...")
    data = load_msmarco_sample(n=n_samples)
    print(f"Загружено {len(data)} примеров")
    
    eval_data = []
    for item in data:
        eval_data.append({
            "question": item["question"],
            "answer": item["answer"],
            "contexts": item["contexts"],
        })
    
    llm = get_llm()
        
    metrics = [
        Faithfulness(llm=llm),
        #AnswerRelevancy(llm=llm),
        ContextPrecision(llm=llm),
        ContextRecall(llm=llm),
        #AnswerCorrectness(llm=llm),
        #AnswerSimilarity(),
    ]
    
    print("Запуск оценки RAGAS...")
    df = pd.DataFrame(eval_data)
    eval_dataset = Dataset.from_pandas(df, name="eval", backend="inmemory")
    
    results = evaluate(
        dataset=eval_dataset,
        metrics=metrics,
        llm=llm,
    )
    
    output_dir = Path(config.RESULTS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"passages_eval_{timestamp}.csv"
    
    results.to_pandas().to_csv(output_path, index=False)
    print(f"Результаты сохранены: {output_path}")
    
    return results


if __name__ == "__main__":
    asyncio.run(run_evaluation(n_samples=100))
