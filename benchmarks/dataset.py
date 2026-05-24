# benchmarks/dataset.py
import os
import json
import random
from pathlib import Path
from typing import List, Dict, Optional

from datasets import load_dataset


# Репозитории на HuggingFace
HF_REPOS = {
    "public_texts":     "ai-forever/hist-rag-bench-public-texts",
    "public_questions": "ai-forever/hist-rag-bench-public-questions",
    "private_qa":       "ai-forever/hist-rag-bench-private-qa",
    "private_texts":   "ai-forever/hist-rag-bench-private-texts",
}

DATA_DIR = Path(__file__).parent / "data" / "dragon"

def load_dragon_data(limit: Optional[int] = None, seed: Optional[int] = None):
    """
    1. Проверяет наличие датасета в кэше и при необходимости скачивает.
    2. Возвращает набор текстов и заданное количество рандомных вопросов
    """
    _ensure_dragon_data()

    print("=" * 60)
    print("Загрузка DRAGON...")
    benchmark = _get_dragon_benchmark()
    corpus = benchmark["corpus"]
    qa = benchmark["qa"]
    print(f"  Корпус: {len(corpus)} документов")
    print(f"  QA:     {len(qa)} вопросов")
    if limit:
        print(f"  Лимит:  {limit}")
        if limit > len(qa):
            print(f"    Предупреждение: лимит {limit} больше доступных вопросов ({len(qa)})")
            limit = len(qa)
        if seed is not None:
            random.seed(seed)
        random.shuffle(qa)
        qa = qa[:limit]
    return corpus, qa

def _ensure_dragon_data() -> bool:
    has_files = DATA_DIR.exists() and any(DATA_DIR.glob("*.jsonl"))
    if has_files:
        print(f"Датасет найден: {DATA_DIR}")
        return True
    print("Датасет DRAGON не обнаружен. Скачиваем...")
    _download_dragon()
    return True


def _get_dragon_benchmark() -> Dict[str, List[Dict]]:
    return {
        "corpus": _load_corpus(),
        "qa": _load_qa_split(),
    }

def _download_dragon():
    """
    Скачивает датасеты DRAGON с HuggingFace и сохраняет в benchmarks/data/dragon/ как JSONL.
    """
    try:
        token = os.environ.get("HF_TOKEN")
    except:
        token = None
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for name, repo_id in HF_REPOS.items():
        print(f"Загрузка {name} из {repo_id} ...")
        ds = load_dataset(repo_id, token=token)

        for split_name, split_ds in ds.items():
            out_file = DATA_DIR / f"{name}_{split_name}.jsonl"
            print(f"  Сохранение {split_name} ({len(split_ds)} записей) -> {out_file}")
            with open(out_file, "w", encoding="utf-8") as f:
                for row in split_ds:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\n✅ Все данные сохранены в {DATA_DIR}")

def _load_corpus() -> List[Dict]:
    candidates = sorted(DATA_DIR.glob("public_texts_*.jsonl"))
    docs = []
    with open(candidates[0], "r", encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))
    return docs


def _load_qa_split() -> List[Dict]:
    candidates = sorted(DATA_DIR.glob("private_qa_*.jsonl"))
    if not candidates:
        candidates = sorted(DATA_DIR.glob("public_questions_*.jsonl"))
    rows = []
    with open(candidates[0], "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            rows.append({
                "q": item.get("question", ""),
                "gt": item.get("answer", ""),
                "public_id": item.get("public_id", ""),
            })
    return rows
"""
def save_results(
    scores: Dict,
    config: Dict,
    num_evaluated: int,
    output_path: Path,
):

    print("\n📊 Результаты RAGAS (DRAGON):")
    for k, v in scores.items():
        print(f"  {k:25s}: {v:.3f}")
    avg = sum(scores.values()) / len(scores)
    print(f"  {'average':25s}: {avg:.3f}")
    print("=" * 60)

    result = {
        "config": config,
        "scores": scores,
        "num_evaluated": num_evaluated,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Сохранено: {output_path}")
    """
