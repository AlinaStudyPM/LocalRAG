# benchmarks/eval_retrieval_dragon.py
"""
RQ2: Влияние глубины первичного поиска (top_embed).

Измеряет:
  — context_precision, context_recall
  — latency (sec) и peak memory (MB)

"""

import argparse
import json
from pathlib import Path

import benchmarks.config_retrieval as cfg
from benchmarks.dataset import load_dragon_data
from benchmarks.embeddings import FastEmbedRagas
from benchmarks.judge import get_judge_genapi as get_judge
from benchmarks.metrics import MetricsManager
from benchmarks.rag_pipeline import BenchRAG
from benchmarks.result_utils import RunLogger, ResultRetrieval
from benchmarks.benchmark_utils import benchmark_call


def main():
    parser = argparse.ArgumentParser(description="RQ2: Влияние количества фрагментов первичного поиска.")
    parser.add_argument("--model", default=cfg.GENERATION_MODEL, help="Модель Ollama для генерации")
    parser.add_argument("--embedding", default=cfg.EMBEDDING_MODEL, help="Модель эмбеддингов")
    parser.add_argument("--prompt", default=cfg.DEFAULT_PROMPT_FILE, help="Файл промпта")
    parser.add_argument("--top-embed-values", nargs="+", type=int, default=cfg.TOP_EMBED_VALUES,
                        help="Список значений top_embed (например: 5 10 15 20 30)")
    parser.add_argument("--temp", type=float, default=cfg.TEMPERATURE, help="Temperature генерации")
    parser.add_argument("--limit", type=int, default=cfg.DEFAULT_LIMIT, help="Лимит вопросов")
    parser.add_argument("--judge", default=cfg.JUDGE_MODEL, help="Модель-судья")
    parser.add_argument("--out-dir", default="./benchmarks/results/retrieval", help="Папка для результатов")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Данные
    # ------------------------------------------------------------------
    print("=" * 60)
    print("RQ2: Влияние top_embed на retrieval-метрики")
    print("=" * 60)
    corpus, qa = load_dragon_data(limit=args.limit)

    # ------------------------------------------------------------------
    # 2. Пайплайн (один экземпляр на все прогоны)
    # ------------------------------------------------------------------
    chroma_subdir = args.embedding.replace("/", "_")
    rag = BenchRAG(
        chroma_subdir=chroma_subdir,
        model_name=args.model,
        prompt_file=args.prompt,
        embedding_model=args.embedding,
    )

    # ------------------------------------------------------------------
    # 3. Индексация (один раз)
    # ------------------------------------------------------------------
    print("\n[1/4] Индексация корпуса...")
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

    # ------------------------------------------------------------------
    # 4. Судья и метрики
    # ------------------------------------------------------------------
    print("[2/4] Инициализация судьи и метрик...")
    judge = get_judge(model=args.judge, temperature=0.0)
    embeddings = FastEmbedRagas(model_name=args.embedding)
    mm = MetricsManager(judge, embeddings)

    # ------------------------------------------------------------------
    # 5. Логгер
    # ------------------------------------------------------------------
    logger = RunLogger(run_dir=Path(args.out_dir))
    logger.log_config({
        "experiment": "RQ2_retrieval",
        "model_gen": args.model,
        "embedding_model": args.embedding,
        "prompt_file": args.prompt,
        "top_embed_values": args.top_embed_values,
        "temperature": args.temp,
        "judge": args.judge,
        "limit": args.limit,
    })

    # ------------------------------------------------------------------
    # 6. Основной цикл: один проход на каждое значение top_embed
    # ------------------------------------------------------------------
    print("[3/4] Оценка конфигураций retrieval...")
    samples_by_embed = {}

    for top_embed in args.top_embed_values:
        print(f"\n>>> top_embed = {top_embed}")
        qs, contexts_list, rag_answers, gts = [], [], [], []
        latencies, memories = [], []

        for i, item in enumerate(qa, 1):
            q = item["q"]
            gt = item["gt"]

            def _rag_cycle(query):
                ctxs = rag.retrieve(query, "dragon", top_embed=top_embed, use_rerank=False)
                ans = rag.generate_rag(query, ctxs, temperature=args.temp)
                return ans, ctxs

            (rag_ans, ctxs), lat, mem = benchmark_call(_rag_cycle, q)

            qs.append(q)
            contexts_list.append(ctxs)
            rag_answers.append(rag_ans)
            gts.append(gt)
            latencies.append(lat)
            memories.append(mem)

            if i % 10 == 0 or i == len(qa):
                print(f"     Обработано {i}/{len(qa)}")

        # Batch-метрики RAGAS
        print(f"     Вычисление метрик для top_embed={top_embed}...")
        cp_scores = mm.context_precision_batch(qs, contexts_list, gts)
        cr_scores = mm.context_recall_batch(qs, contexts_list, gts)

        # Сохраняем сырые данные по этому top_embed
        samples = []
        for idx in range(len(qs)):
            samples.append({
                "question": qs[idx],
                "ground_truth": gts[idx],
                "contexts": contexts_list[idx],
                "rag_answer": rag_answers[idx],
                "latency_sec": round(latencies[idx], 3),
                "memory_mb": round(memories[idx], 1),
                "scores": {
                    "context_precision": cp_scores[idx],
                    "context_recall": cr_scores[idx],
                },
            })

        samples_by_embed[top_embed] = samples

        # Append-only лог (сразу пишем, чтобы не потерять при падении)
        for s in samples:
            logger.log_sample({"top_embed": top_embed, **s})

    # ------------------------------------------------------------------
    # 7. Перезапись JSONL финальными данными
    # ------------------------------------------------------------------
    print("[4/4] Формирование отчёта и визуализация...")
    all_samples = []
    for top_embed, samples in samples_by_embed.items():
        for s in samples:
            all_samples.append({"top_embed": top_embed, **s})

    logger.samples_file.write_text(
        "".join(json.dumps(s, ensure_ascii=False) + "\n" for s in all_samples),
        encoding="utf-8"
    )

    # ------------------------------------------------------------------
    # 8. Агрегация
    # ------------------------------------------------------------------
    agg = ResultRetrieval(samples_by_embed)
    summary = agg.compute_summary()

    md_path = logger.run_dir / f"{logger.run_id}_report.md"
    agg.to_markdown(md_path)
    print(f"\n📄 Markdown отчёт: {md_path}")

    # ------------------------------------------------------------------
    # 9. Визуализация
    # ------------------------------------------------------------------
    agg.plot(logger.run_dir / f"{logger.run_id}_plot.png")

    # ------------------------------------------------------------------
    # 10. Итог в консоль
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("ИТОГОВЫЕ МЕТРИКИ")
    print("=" * 60)
    for i, te in enumerate(summary["top_embeds"]):
        print(f"\ntop_embed = {te}")
        cp = summary["context_precision"]["mean"][i]
        cr = summary["context_recall"]["mean"][i]
        lat = summary["latency"]["mean"][i]
        mem = summary["memory"]["mean"][i]
        print(f"  Context Precision : {cp:.3f}" if cp is not None else "  Context Precision : N/A")
        print(f"  Context Recall    : {cr:.3f}" if cr is not None else "  Context Recall    : N/A")
        print(f"  Latency (sec)     : {lat:.3f}" if lat is not None else "  Latency (sec)     : N/A")
        print(f"  Memory (MB)       : {mem:.1f}" if mem is not None else "  Memory (MB)       : N/A")
    print("=" * 60)


if __name__ == "__main__":
    main()
