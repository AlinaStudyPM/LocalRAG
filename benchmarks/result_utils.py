# benchmarks/result_utils.py

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class RunLogger:
    """Логгер для сохранения конфигурации и промежуточных результатов эксперимента."""

    def __init__(self, run_dir: Path, run_id: Optional[str] = None):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.samples_file = self.run_dir / f"{self.run_id}_samples.jsonl"
        self.config_file = self.run_dir / f"{self.run_id}_config.json"

    def log_config(self, config: Dict[str, Any]) -> None:
        """Сохраняет конфигурацию эксперимента в JSON файл."""
        payload = {
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            **config,
        }
        with open(self.config_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def log_sample(self, sample: Dict[str, Any]) -> None:
        """Append-only запись промежуточных результатов (сохраняются даже при падении скрипта)."""
        with open(self.samples_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


class ResultGeneration:
    """Агрегация, отчёт и визуализация для RQ3 (generation: RAG vs Baseline)."""

    def __init__(self, samples: List[Dict[str, Any]]):
        self.samples = samples

    @classmethod
    def from_jsonl(cls, path: Path):
        """Загружает samples из JSONL файла."""
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        return cls(samples)

    @staticmethod
    def _mean(values: List[Any]) -> Optional[float]:
        clean = [v for v in values if v is not None]
        return sum(clean) / len(clean) if clean else None

    @staticmethod
    def _std(values: List[Any]) -> Optional[float]:
        clean = [v for v in values if v is not None]
        if len(clean) < 2:
            return 0.0
        m = sum(clean) / len(clean)
        return (sum((x - m) ** 2 for x in clean) / (len(clean) - 1)) ** 0.5

    def compute_summary(self) -> Dict[str, Any]:
        """Вычисляет средние и стандартные отклонения для всех метрик."""
        if not self.samples:
            return {}

        rag_scores = {"faithfulness": [], "answer_correctness": [], "answer_relevancy": []}
        base_scores = {"answer_correctness": [], "answer_relevancy": []}
        rag_lat, rag_mem = [], []
        base_lat, base_mem = [], []

        for s in self.samples:
            rag = s.get("scores", {}).get("rag", {})
            base = s.get("scores", {}).get("baseline", {})
            
            for k in rag_scores:
                if k in rag and rag[k] is not None:
                    rag_scores[k].append(rag[k])
            for k in base_scores:
                if k in base and base[k] is not None:
                    base_scores[k].append(base[k])

            if "rag_latency_sec" in s and s["rag_latency_sec"] is not None:
                rag_lat.append(s["rag_latency_sec"])
            if "rag_memory_mb" in s and s["rag_memory_mb"] is not None:
                rag_mem.append(s["rag_memory_mb"])
            if "baseline_latency_sec" in s and s["baseline_latency_sec"] is not None:
                base_lat.append(s["baseline_latency_sec"])
            if "baseline_memory_mb" in s and s["baseline_memory_mb"] is not None:
                base_mem.append(s["baseline_memory_mb"])

        return {
            "rag": {k: {"mean": self._mean(v), "std": self._std(v)} for k, v in rag_scores.items()},
            "baseline": {k: {"mean": self._mean(v), "std": self._std(v)} for k, v in base_scores.items()},
            "resources": {
                "rag_latency_sec": {"mean": self._mean(rag_lat), "std": self._std(rag_lat)},
                "rag_memory_mb": {"mean": self._mean(rag_mem), "std": self._std(rag_mem)},
                "baseline_latency_sec": {"mean": self._mean(base_lat), "std": self._std(base_lat)},
                "baseline_memory_mb": {"mean": self._mean(base_mem), "std": self._std(base_mem)},
            },
            "count": len(self.samples),
        }

    def to_markdown(self, path: Optional[Path] = None) -> str:
        """Генерирует Markdown отчёт со средними и стандартными отклонениями."""
        summary = self.compute_summary()
        lines = [
            "# Результаты оценки RAG (Dragon) — RQ3: Generation",
            "",
            f"**Количество примеров:** {summary['count']}",
            "",
            "## RAG",
            "",
            "| Метрика | Среднее | Std |",
            "|---------|---------|-----|",
        ]
        
        for k, v in summary["rag"].items():
            mean = f"{v['mean']:.3f}" if v["mean"] is not None else "N/A"
            std = f"{v['std']:.3f}" if v["std"] is not None else "N/A"
            lines.append(f"| {k} | {mean} | {std} |")

        lines.extend([
            "",
            "## Baseline (чистая LLM)",
            "",
            "| Метрика | Среднее | Std |",
            "|---------|---------|-----|",
        ])
        
        for k, v in summary["baseline"].items():
            mean = f"{v['mean']:.3f}" if v["mean"] is not None else "N/A"
            std = f"{v['std']:.3f}" if v["std"] is not None else "N/A"
            lines.append(f"| {k} | {mean} | {std} |")

        lines.extend([
            "",
            "## Ресурсы",
            "",
            "| Параметр | Среднее | Std |",
            "|----------|---------|-----|",
        ])
        
        for k, v in summary.get("resources", {}).items():
            mean = f"{v['mean']:.3f}" if v["mean"] is not None else "N/A"
            std = f"{v['std']:.3f}" if v["std"] is not None else "N/A"
            lines.append(f"| {k} | {mean} | {std} |")

        md = "\n".join(lines)
        if path:
            Path(path).write_text(md, encoding="utf-8")
        return md

    def plot(self, save_path: Path):
            """
            1×2 grouped-bar chart:
              - Left: Quality (answer_correctness, answer_relevancy) со значениями
              - Right: Resources (latency, memory) с разными штриховками, двойная ось Y,
                   значения над каждым столбцом
            """
            summary = self.compute_summary()
            rag = summary["rag"]
            base = summary["baseline"]
            res = summary.get("resources", {})
    
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            width = 0.35
    
            # ── Left subplot: Quality ──────────────────────────────────────────
            quality_metrics = ["answer_correctness", "answer_relevancy"]
            labels_q = ["Answer Correctness", "Answer Relevancy"]
            x_q = np.arange(len(labels_q))
    
            rag_vals_q = [rag[m]["mean"] if rag[m]["mean"] is not None else 0 for m in quality_metrics]
            base_vals_q = [base[m]["mean"] if base[m]["mean"] is not None else 0 for m in quality_metrics]
    
            bars_rag_q = axes[0].bar(
                x_q - width / 2, rag_vals_q, width, label="RAG", color="#4c78a8"
           )
            bars_base_q = axes[0].bar(
                x_q + width / 2, base_vals_q, width, label="Baseline", color="#f58518"
            )
    
            # Подписи над столбцами качества
            for bar in bars_rag_q:
                h = bar.get_height()
                axes[0].annotate(
                    f"{h:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color="#000"
                )
            for bar in bars_base_q:
                h = bar.get_height()
                axes[0].annotate(
                    f"{h:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color="#000"
                )
    
            axes[0].set_ylabel("Score")
            axes[0].set_title("Качество ответов")
            axes[0].set_xticks(x_q)
            axes[0].set_xticklabels(labels_q)
            axes[0].legend()
            axes[0].set_ylim(0, 1.1)

            # ── Right subplot: Resources ───────────────────────────────────────
            ax_res = axes[1]
            ax_res_mem = ax_res.twinx()
    
            rag_lat = res.get("rag_latency_sec", {}).get("mean", 0)
            base_lat = res.get("baseline_latency_sec", {}).get("mean", 0)
            rag_mem = res.get("rag_memory_mb", {}).get("mean", 0)
            base_mem = res.get("baseline_memory_mb", {}).get("mean", 0)

            x_r = np.arange(2)

            # Latency — основная ось (левая), со штриховкой "/"
            bars_lat_rag = ax_res.bar(
                x_r[0] - width / 2, rag_lat, width,
               color="#4c78a8", alpha=0.6, hatch="//", edgecolor="black", linewidth=0.5,
               label="RAG"
            )
            bars_lat_base = ax_res.bar(
                x_r[0] + width / 2, base_lat, width,
                color="#f58518", alpha=0.6, hatch="//", edgecolor="black", linewidth=0.5,
                label="Baseline"
            )
            ax_res.set_ylabel("Latency (sec)", color="#333")
            ax_res.tick_params(axis="y", labelcolor="#333")
    
            # Memory — вторая ось (правая), со штриховкой "\"
            bars_mem_rag = ax_res_mem.bar(
                x_r[1] - width / 2, rag_mem, width,
                color="#4c78a8", alpha=0.6, hatch="\\", edgecolor="black", linewidth=0.5
            )
            bars_mem_base = ax_res_mem.bar(
                x_r[1] + width / 2, base_mem, width,
                color="#f58518", alpha=0.6, hatch="\\", edgecolor="black", linewidth=0.5
            )
            ax_res_mem.set_ylabel("Peak Memory (MB)", color="#555")
            ax_res_mem.tick_params(axis="y", labelcolor="#555")

            # Подписи над столбцами Latency (основная ось)
            for bar, val in [(bars_lat_rag[0], rag_lat), (bars_lat_base[0], base_lat)]:
                ax_res.annotate(
                    f"{val:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color="#333"
                )

            # Подписи над столбцами Memory (вторая ось)
            for bar, val in [(bars_mem_rag[0], rag_mem), (bars_mem_base[0], base_mem)]:
                ax_res_mem.annotate(
                    f"{val:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color="#555"
                )

            ax_res.set_xticks(x_r)
            ax_res.set_xticklabels(["Latency (sec)", "Memory (MB)"])
            ax_res.set_title("Ресурсы")

            # Единая легенда с учётом штриховки
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor="#4c78a8", hatch="--", edgecolor="black", label="RAG"),
                Patch(facecolor="#f58518", hatch="--", edgecolor="black", label="Baseline"),
            ]
            ax_res.legend(handles=legend_elements, loc="upper left")

            fig.tight_layout()
            fig.savefig(save_path, dpi=150)
            plt.close(fig)
            print(f"📊 График RQ3 сохранён: {save_path}")

        
class ResultRetrieval:
    """Агрегация, отчёт и визуализация для RQ2 (retrieval: влияние top_embed)."""

    def __init__(self, samples_by_embed: Dict[int, List[Dict]]):
        self.samples_by_embed = samples_by_embed

    @staticmethod
    def _mean(values: List[Any]) -> Optional[float]:
        clean = [v for v in values if v is not None]
        return float(np.mean(clean)) if clean else None

    @staticmethod
    def _std(values: List[Any]) -> Optional[float]:
        clean = [v for v in values if v is not None]
        return float(np.std(clean)) if len(clean) > 1 else 0.0

    def compute_summary(self) -> Dict[str, Any]:
        """Вычисляет средние и std для каждого значения top_embed."""
        summary = {
            "top_embeds": [],
            "context_precision": {"mean": [], "std": []},
            "context_recall": {"mean": [], "std": []},
            "latency": {"mean": [], "std": []},
            "memory": {"mean": [], "std": []},
        }

        for top_embed in sorted(self.samples_by_embed.keys()):
            samples = self.samples_by_embed[top_embed]
            summary["top_embeds"].append(top_embed)

            cp_vals = [s["scores"]["context_precision"] for s in samples]
            summary["context_precision"]["mean"].append(self._mean(cp_vals))
            summary["context_precision"]["std"].append(self._std(cp_vals))

            cr_vals = [s["scores"]["context_recall"] for s in samples]
            summary["context_recall"]["mean"].append(self._mean(cr_vals))
            summary["context_recall"]["std"].append(self._std(cr_vals))

            lat_vals = [s["latency_sec"] for s in samples]
            summary["latency"]["mean"].append(self._mean(lat_vals))
            summary["latency"]["std"].append(self._std(lat_vals))

            mem_vals = [s["memory_mb"] for s in samples]
            summary["memory"]["mean"].append(self._mean(mem_vals))
            summary["memory"]["std"].append(self._std(mem_vals))

        return summary

    def to_markdown(self, path: Optional[Path] = None) -> str:
        """Генерирует Markdown отчёт со средними и стандартными отклонениями."""
        summary = self.compute_summary()
        lines = [
            "# RQ2: Влияние top_embed на retrieval-метрики\n",
            "",
            "| top_embed | Context Precision | Context Recall | Latency (sec) | Memory (MB) |",
            "|-----------|-------------------|----------------|---------------|-------------|",
        ]

        for i, te in enumerate(summary["top_embeds"]):
            cp_mean = summary["context_precision"]["mean"][i]
            cp_std = summary["context_precision"]["std"][i]
            cr_mean = summary["context_recall"]["mean"][i]
            cr_std = summary["context_recall"]["std"][i]
            lat_mean = summary["latency"]["mean"][i]
            lat_std = summary["latency"]["std"][i]
            mem_mean = summary["memory"]["mean"][i]
            mem_std = summary["memory"]["std"][i]

            def fmt(mean, std, fmt_s):
                if mean is None:
                    return "N/A"
                return f"{fmt_s.format(mean)} ± {fmt_s.format(std)}"

            lines.append(
                f"| {te} | {fmt(cp_mean, cp_std, '{:.3f}')} | {fmt(cr_mean, cr_std, '{:.3f}')} "
                f"| {fmt(lat_mean, lat_std, '{:.3f}')} | {fmt(mem_mean, mem_std, '{:.1f}')} |"
            )

        md = "\n".join(lines)
        if path:
            Path(path).write_text(md, encoding="utf-8")
        return md

    def plot(self, output_path: Path):
        """Рисует 2×2 grid bar-charts с подписями значений."""
        summary = self.compute_summary()
        top_embeds = summary["top_embeds"]
        
        metrics = [
            (summary["context_precision"]["mean"], "Context Precision", "Score"),
            (summary["context_recall"]["mean"], "Context Recall", "Score"),
            (summary["latency"]["mean"], "Latency (sec)", "Seconds"),
            (summary["memory"]["mean"], "Peak Memory (MB)", "MB"),
        ]

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(
            f"RQ2: Влияние top_embed (без rerank)\n({len(top_embeds)} конфигураций)",
            fontsize=14,
        )

        for ax, (values, title, ylabel) in zip(axes.flat, metrics):
            x = np.arange(len(top_embeds))
            display_vals = [v if v is not None else 0.0 for v in values]
            color = "steelblue" if "Score" in ylabel else "coral"
            bars = ax.bar(x, display_vals, color=color, width=0.5, edgecolor="black", linewidth=0.5)

            ax.set_xticks(x)
            ax.set_xticklabels(top_embeds)
            ax.set_xlabel("top_embed", fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.grid(axis="y", linestyle="--", alpha=0.4)

            for bar, val in zip(bars, values):
                height = bar.get_height()
                label = f"{val:.3f}" if val is not None else "N/A"
                ax.annotate(
                    label,
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                )

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"📊 График сохранён: {output_path}")


class ResultRerank:
    """Агрегация, отчёт и визуализация для RQ3 (rerank: влияние top_rerank)."""

    def __init__(self, samples_by_rerank: Dict[int, List[Dict]]):
        self.samples_by_rerank = samples_by_rerank

    @staticmethod
    def _mean(values: List[Any]) -> Optional[float]:
        clean = [v for v in values if v is not None]
        return float(np.mean(clean)) if clean else None

    @staticmethod
    def _std(values: List[Any]) -> Optional[float]:
        clean = [v for v in values if v is not None]
        return float(np.std(clean)) if len(clean) > 1 else 0.0

    def compute_summary(self) -> Dict[str, Any]:
        """Вычисляет средние и std для каждого значения top_rerank."""
        summary = {
            "top_reranks": [],
            "context_precision": {"mean": [], "std": []},
            "context_recall": {"mean": [], "std": []},
            "latency": {"mean": [], "std": []},
            "memory": {"mean": [], "std": []},
        }

        for top_rerank in sorted(self.samples_by_rerank.keys()):
            samples = self.samples_by_rerank[top_rerank]
            summary["top_reranks"].append(top_rerank)

            cp_vals = [s["scores"]["context_precision"] for s in samples]
            summary["context_precision"]["mean"].append(self._mean(cp_vals))
            summary["context_precision"]["std"].append(self._std(cp_vals))

            cr_vals = [s["scores"]["context_recall"] for s in samples]
            summary["context_recall"]["mean"].append(self._mean(cr_vals))
            summary["context_recall"]["std"].append(self._std(cr_vals))

            lat_vals = [s["latency_sec"] for s in samples]
            summary["latency"]["mean"].append(self._mean(lat_vals))
            summary["latency"]["std"].append(self._std(lat_vals))

            mem_vals = [s["memory_mb"] for s in samples]
            summary["memory"]["mean"].append(self._mean(mem_vals))
            summary["memory"]["std"].append(self._std(mem_vals))

        return summary

    def to_markdown(self, path: Optional[Path] = None, top_embed_fixed: int = 20) -> str:
        """Генерирует Markdown отчёт со средними и стандартными отклонениями."""
        summary = self.compute_summary()
        lines = [
            f"# RQ3: Влияние реранкера при фиксированном top_embed={top_embed_fixed}\n",
            "",
            "| top_rerank | Режим | Context Precision | Context Recall | Latency (sec) | Memory (MB) |",
            "|------------|-------|-------------------|----------------|---------------|-------------|",
        ]

        for i, tr in enumerate(summary["top_reranks"]):
            mode = "без rerank" if tr == top_embed_fixed else "rerank"
            cp_mean = summary["context_precision"]["mean"][i]
            cp_std = summary["context_precision"]["std"][i]
            cr_mean = summary["context_recall"]["mean"][i]
            cr_std = summary["context_recall"]["std"][i]
            lat_mean = summary["latency"]["mean"][i]
            lat_std = summary["latency"]["std"][i]
            mem_mean = summary["memory"]["mean"][i]
            mem_std = summary["memory"]["std"][i]

            def fmt(mean, std, fmt_s):
                if mean is None:
                    return "N/A"
                return f"{fmt_s.format(mean)} ± {fmt_s.format(std)}"

            lines.append(
                f"| {tr} | {mode} | {fmt(cp_mean, cp_std, '{:.3f}')} | {fmt(cr_mean, cr_std, '{:.3f}')} "
                f"| {fmt(lat_mean, lat_std, '{:.3f}')} | {fmt(mem_mean, mem_std, '{:.1f}')} |"
            )

        md = "\n".join(lines)
        if path:
            Path(path).write_text(md, encoding="utf-8")
        return md

    def plot(self, output_path: Path):
        """Рисует 2×2 grid bar-charts с подписями значений."""
        summary = self.compute_summary()
        top_reranks = summary["top_reranks"]
        
        metrics = [
            (summary["context_precision"]["mean"], "Context Precision", "Score"),
            (summary["context_recall"]["mean"], "Context Recall", "Score"),
            (summary["latency"]["mean"], "Latency (sec)", "Seconds"),
            (summary["memory"]["mean"], "Peak Memory (MB)", "MB"),
        ]

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(
            f"RQ3: Влияние top_rerank при фиксированном top_embed=20\n"
            f"({len(top_reranks)} конфигураций)",
            fontsize=14,
        )

        for ax, (values, title, ylabel) in zip(axes.flat, metrics):
            x = np.arange(len(top_reranks))
            display_vals = [v if v is not None else 0.0 for v in values]
            color = "steelblue" if "Score" in ylabel else "coral"
            bars = ax.bar(x, display_vals, color=color, width=0.5, edgecolor="black", linewidth=0.5)

            ax.set_xticks(x)
            ax.set_xticklabels(top_reranks)
            ax.set_xlabel("top_rerank", fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.grid(axis="y", linestyle="--", alpha=0.4)

            for bar, val in zip(bars, values):
                height = bar.get_height()
                label = f"{val:.3f}" if val is not None else "N/A"
                ax.annotate(
                    label,
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                )

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"📊 График сохранён: {output_path}")
