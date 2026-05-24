# benchmarks/eval_generation_dragon.py
"""
RQ1: RAG vs Baseline (чистая LLM) на датасете DRAGON.

Измеряет:
  — answer correctness, answer relevancy
  — faithfulness
  — latency (sec) и peak memory (MB) на один вопрос

График: 1×2 — левый субплот «Качество», правый — «Ресурсы».
"""
import argparse
import json
from pathlib import Path

import benchmarks.config_generation as cfg
from benchmarks.dataset import load_dragon_data
from benchmarks.embeddings import FastEmbedRagas
from benchmarks.judge import get_judge_genapi as get_judge
from benchmarks.metrics import MetricsManager
from benchmarks.rag_pipeline import BenchRAG
from benchmarks.result_utils import RunLogger, ResultGeneration
from benchmarks.benchmark_utils import benchmark_call


def main():
    parser = argparse.ArgumentParser(description="RQ1: RAG vs Baseline на DRAGON")
    parser.add_argument("--model", default=cfg.GENERATION_MODEL, help="Модель Ollama для генерации")
    parser.add_argument("--embedding", default=cfg.EMBEDDING_MODEL, help="Модель эмбеддингов")
    parser.add_argument("--prompt", default=cfg.DEFAULT_PROMPT_FILE, help="Файл промпта из benchmarks/prompts/")
    parser.add_argument("--top-embed", type=int, default=cfg.TOP_EMBED, help="Количество первичных retrieved contexts")
    parser.add_argument("--top-rerank", type=int, default=cfg.TOP_RERANK, help="Количество вторичных retrieved contexts")
    parser.add_argument("--temp", type=float, default=cfg.TEMPERATURE, help="Temperature генерации")
    parser.add_argument("--limit", type=int, default=cfg.DEFAULT_LIMIT, help="Лимит вопросов для отладки")
    parser.add_argument("--judge", default=cfg.JUDGE_MODEL, help="Модель-судья (Ollama)")
    parser.add_argument("--out-dir", default="./benchmarks/results/generation", help="Папка для результатов")
    args = parser.parse_args()

    # 1. Данные
    print("=" * 60)
    print("RQ1: RAG vs Baseline на DRAGON")
    print("=" * 60)
    corpus, qa = load_dragon_data(limit=args.limit)

    # 2. Пайплайн
    chroma_subdir = args.embedding.replace("/", "_")
    rag = BenchRAG(
        chroma_subdir=chroma_subdir,
        model_name=args.model,
        prompt_file=args.prompt,
        embedding_model=args.embedding,
    )

    # 3. Индексация
    print("\n[1/5] Индексация корпуса...")
    try:
        existing = rag.chroma.list_files("dragon")
        if existing:
            print(f"     Коллекция 'dragon' уже существует ({len(existing)} файлов), пропускаем индексацию")
        else:
            passages = [doc.get("text", doc.get("content", str(doc))) for doc in corpus]
            rag.index_passages(passages, collection_name="dragon")
            print(f"     Залито {len(passages)} абзацев")
    except Exception:
        passages = [doc.get("text", doc.get("content", str(doc))) for doc in corpus]
        rag.index_passages(passages, collection_name="dragon")
        print(f"     Залито {len(passages)} абзацев")

    # 4. Судья и метрики
    print("[2/5] Инициализация судьи и метрик...")
    judge = get_judge(model=args.judge, temperature=0.0)
    embeddings = FastEmbedRagas(model_name=args.embedding)
    mm = MetricsManager(judge, embeddings)

    # 5. Логгер (append-only JSONL + JSON config)
    logger = RunLogger(run_dir=Path(args.out_dir))
    logger.log_config({
        "experiment": "RQ1_generation",
        "model_gen": args.model,
        "embedding_model": args.embedding,
        "prompt_file": args.prompt,
        "top_embed": args.top_embed,
        "top_rerank": args.top_rerank,
        "temperature": args.temp,
        "judge": args.judge,
        "limit": args.limit,
    })

    # 6. Генерация ответов с замером latency / memory
    print("[3/5] Генерация ответов с измерением ресурсов...")
    qs, rag_answers, base_answers, contexts_list, gts = [], [], [], [], []
    rag_latencies, rag_memories = [], []
    base_latencies, base_memories = [], []

    for i, item in enumerate(qa, 1):
        q = item["q"]
        gt = item["gt"]

        # Полный RAG-цикл: retrieve → generate
        def _rag_pipeline(query):
            ctxs = rag.retrieve(query, "dragon", args.top_embed, args.top_rerank)
            ans = rag.generate_rag(query, ctxs, temperature=args.temp)
            return ans, ctxs

        (rag_ans, contexts), rag_lat, rag_mem = benchmark_call(_rag_pipeline, q)
        base_ans, base_lat, base_mem = benchmark_call(rag.generate_baseline, q, temperature=args.temp)

        qs.append(q)
        rag_answers.append(rag_ans)
        base_answers.append(base_ans)
        contexts_list.append(contexts)
        gts.append(gt)
        rag_latencies.append(rag_lat)
        rag_memories.append(rag_mem)
        base_latencies.append(base_lat)
        base_memories.append(base_mem)

        # Append-only: данные сохраняются сразу, даже если скрипт упадёт дальше
        logger.log_sample({
            "question": q,
            "ground_truth": gt,
            "contexts": contexts,
            "rag_answer": rag_ans,
            "baseline_answer": base_ans,
            "rag_latency_sec": round(rag_lat, 3),
            "rag_memory_mb": round(rag_mem, 1),
            "baseline_latency_sec": round(base_lat, 3),
            "baseline_memory_mb": round(base_mem, 1),
        })

        if i % 10 == 0 or i == len(qa):
            print(f"     Обработано {i}/{len(qa)}")

    # 7. Метрики RAGAS (batch)
    print("[4/5] Вычисление метрик RAGAS...")
    print("     Вычисляем faithfulness для RAG...")
    rag_f = [None] * len(qs)
    # rag_f = mm.faithfulness_batch(qs, rag_answers, contexts_list)
    print("     Вычисляем answer correctness для RAG...")
    rag_ac = mm.answer_correctness_batch(qs, rag_answers, gts)
    print("     Вычисляем answer relevancy для RAG...")
    rag_ar = mm.answer_relevancy_batch(qs, rag_answers)
    
    print("     Вычисляем answer correctness для baseline....")
    base_ac = mm.answer_correctness_batch(qs, base_answers, gts)
    print("     Вычисляем answer relevancy для baseline...")
    base_ar = mm.answer_relevancy_batch(qs, base_answers)

    # Перезаписываем JSONL с финальными скорами
    print("     Выводим результаты...")
    full_samples = []
    for i in range(len(qs)):
        full_samples.append({
            "question": qs[i],
            "ground_truth": gts[i],
            "contexts": contexts_list[i],
            "rag_answer": rag_answers[i],
            "baseline_answer": base_answers[i],
            "rag_latency_sec": rag_latencies[i],
            "rag_memory_mb": rag_memories[i],
            "baseline_latency_sec": base_latencies[i],
            "baseline_memory_mb": base_memories[i],
            "scores": {
                "rag": {
                    "faithfulness": rag_f[i],
                    "answer_correctness": rag_ac[i],
                    "answer_relevancy": rag_ar[i],
                },
                "baseline": {
                    "answer_correctness": base_ac[i],
                    "answer_relevancy": base_ar[i],
                },
            },
        })

    logger.samples_file.write_text(
        "".join(json.dumps(s, ensure_ascii=False) + "\n" for s in full_samples),
        encoding="utf-8"
    )

    # 8. Отчёт и визуализация
    print("[5/5] Формирование отчёта и визуализация...")
    agg = ResultGeneration(full_samples)
    summary = agg.compute_summary()

    md_path = logger.run_dir / f"{logger.run_id}_report.md"
    agg.to_markdown(md_path)
    print(f"\n📄 Markdown отчёт: {md_path}")

    agg.plot(logger.run_dir / f"{logger.run_id}_rq1_generation.png")

    # Итог в консоль
    print("\n" + "=" * 60)
    print("ИТОГОВЫЕ МЕТРИКИ")
    print("=" * 60)
    for k, v in summary["rag"].items():
        m = v["mean"]
        print(f"  RAG {k:20s}: {m:.3f}" if m is not None else f"  RAG {k:20s}: N/A")
    for k, v in summary["baseline"].items():
        m = v["mean"]
        print(f"  Base {k:19s}: {m:.3f}" if m is not None else f"  Base {k:19s}: N/A")
    res = summary.get("resources", {})
    for k, v in res.items():
        m = v["mean"]
        print(f"  {k:25s}: {m:.3f}" if m is not None else f"  {k:25s}: N/A")
    print("=" * 60)


if __name__ == "__main__":
    main()
