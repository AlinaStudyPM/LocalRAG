# benchmarks/data_utils.py
from datasets import load_dataset, load_from_disk, Dataset
from typing import List, Dict, Any
from pathlib import Path
import json

def load_msmarco_sample(n: int = 100) -> List[Dict]:
    """
    Загрузчик датасета MS MARCO.
    Выбирает из датасета только записи, для которых есть релевантные
    тексты и ответы.

    Returns:
        Список словарей с полями:
        - question: str
        - answer: str
        - contexts: List[str]
    """
    cache_dir = "./benchmarks/data"
    cache_path = Path(cache_dir) / "ms_marco_train"
    if cache_path.exists():
        ds = load_from_disk(str(cache_path))
    else:
        ds = load_dataset("microsoft/ms_marco", "v2.1", split="train")
        ds.save_to_disk(str(cache_path))

    filtered = []
    for item in ds:
        if item.get("answers") and len(item["answers"]) > 0:
            passages_data = item.get("passages", {})
            is_selected = passages_data.get("is_selected", [])
            passage_texts = passages_data.get("passage_text", [])
            # Выбрать тексты, где is_selected == 1
            selected_passages = [
                passage_texts[i] 
                for i, selected in enumerate(is_selected) 
                if selected == 1
            ]
            if selected_passages:
                filtered.append({
                    "question": item["query"],
                    "answer": item["answers"][0],
                    "contexts": selected_passages,
                })
        
        if len(filtered) >= n:
            break
    
    return filtered

def save_prepared_data(data: List[Dict], filepath: str):
    """Сохранить подготовленные данные в JSONL"""
    with open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def load_prepared_data(filepath: str) -> List[Dict]:
    """Загрузить подготовленные данные из JSONL"""
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def prepare_for_ragas(data: List[Dict]) -> Dataset:
    """
    Подготовить данные в формат RAGAS
    
    RAGAS ожидает:
    - question: str
    - answer: str  
    - contexts: List[str]
    """
    from ragas import Dataset as RagasDataset
    
    return RagasDataset.from_list(data)
