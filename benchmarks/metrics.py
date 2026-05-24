# benchmarks/metrics.py
from typing import List, Optional

from ragas.metrics.collections import (
    ContextPrecision,
    ContextRecall,
    Faithfulness,
    AnswerCorrectness,
    AnswerRelevancy,
)

from benchmarks.embeddings import FastEmbedRagas
from benchmarks.judge import get_judge_ollama as get_judge


class MetricsManager:
    
    def __init__(self, judge, embeddings):
        self.judge = judge
        self.embeddings = embeddings

        self._ctx_precision = ContextPrecision(llm=judge)
        self._ctx_recall = ContextRecall(llm=judge)
        self._faithfulness = Faithfulness(llm=judge)
        self._answer_correctness = AnswerCorrectness(llm=judge, embeddings=embeddings)
        self._answer_relevancy = AnswerRelevancy(llm=judge, embeddings=embeddings)

    @staticmethod
    def _get_float(result) -> Optional[float]:
        if result is None:
            return None
        val = getattr(result, "value", None)
        return float(val) if val is not None else None

    # -------------------------------------------------------------------------
    # Retrieval-метрики
    # -------------------------------------------------------------------------

    def context_precision(
        self,
        question: str,
        contexts: List[str],
        reference: str,
    ) -> Optional[float]:
        result = self._ctx_precision.score(
            user_input=question,
            retrieved_contexts=contexts,
            reference=reference,
        )
        return self._get_float(result)

    def context_recall(
        self,
        question: str,
        contexts: List[str],
        reference: str,
    ) -> Optional[float]:
        result = self._ctx_recall.score(
            user_input=question,
            retrieved_contexts=contexts,
            reference=reference,
        )
        return self._get_float(result)

    def context_precision_batch(
        self,
        questions: List[str],
        contexts_list: List[List[str]],
        references: List[str],
    ) -> List[Optional[float]]:
        inputs = [
            {
                "user_input": q,
                "retrieved_contexts": ctx,
                "reference": ref,
            }
            for q, ctx, ref in zip(questions, contexts_list, references)
        ]
        results = self._ctx_precision.batch_score(inputs)
        return [self._get_float(r) for r in results]

    def context_recall_batch(
        self,
        questions: List[str],
        contexts_list: List[List[str]],
        references: List[str],
    ) -> List[Optional[float]]:
        inputs = [
            {
                "user_input": q,
                "retrieved_contexts": ctx,
                "reference": ref,
            }
            for q, ctx, ref in zip(questions, contexts_list, references)
        ]
        results = self._ctx_recall.batch_score(inputs)
        return [self._get_float(r) for r in results]
    
    # -------------------------------------------------------------------------
    # Generation-метрики
    # -------------------------------------------------------------------------

    def faithfulness(
        self,
        question: str,
        answer: str,
        contexts: List[str],
    ) -> Optional[float]:
        result = self._faithfulness.score(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
        )
        return self._get_float(result)

    def answer_correctness(
        self,
        question: str,
        answer: str,
        ground_truth: str,
    ) -> Optional[float]:
        result = self._answer_correctness.score(
            user_input=question,
            response=answer,
            reference=ground_truth,
        )
        return self._get_float(result)

    def answer_relevancy(
        self,
        question: str,
        answer: str,
    ) -> Optional[float]:
        result = self._answer_relevancy.score(
            user_input=question,
            response=answer,
        )
        return self._get_float(result)

    def faithfulness_batch(
        self,
        questions: List[str],
        answers: List[str],
        contexts_list: List[List[str]],
    ) -> List[Optional[float]]:
        inputs = [
            {
                "user_input": q,
                "response": a,
                "retrieved_contexts": ctx,
            }
            for q, a, ctx in zip(questions, answers, contexts_list)
        ]
        results = self._faithfulness.batch_score(inputs)
        return [self._get_float(r) for r in results]

    def answer_correctness_batch(
        self,
        questions: List[str],
        answers: List[str],
        ground_truths: List[str],
    ) -> List[Optional[float]]:
        inputs = [
            {
                "user_input": q,
                "response": a,
                "reference": gt,
            }
            for q, a, gt in zip(questions, answers, ground_truths)
        ]
        results = self._answer_correctness.batch_score(inputs)
        return [self._get_float(r) for r in results]

    def answer_relevancy_batch(
        self,
        questions: List[str],
        answers: List[str],
    ) -> List[Optional[float]]:
        inputs = [
            {
                "user_input": q,
                "response": a,
            }
            for q, a in zip(questions, answers)
        ]
        results = self._answer_relevancy.batch_score(inputs)
        return [self._get_float(r) for r in results]


if __name__ == "__main__":
    test_rows = [
        {
            "q": "Какой сегодня день?",
            "ctx": ["Сегодня понедельник, 17 апреля 2026 года."],
            "ans": "Понедельник",
            "gt": "Понедельник, 17 апреля 2026 года.",
        },
        {
            "q": "Столица Франции?",
            "ctx": ["Париж — столица Франции."],
            "ans": "Париж",
            "gt": "Париж",
        },
    ]

    judge = get_judge(model="qwen2.5:7b-instruct")
    embeddings = FastEmbedRagas("BAAI/bge-small-en-v1.5")
    mm = MetricsManager(judge, embeddings)

    qs = [r["q"] for r in test_rows]
    ctxs = [r["ctx"] for r in test_rows]
    ans = [r["ans"] for r in test_rows]
    gt = [r["gt"] for r in test_rows]

    print("Все метрики:")
    print(f"  context_precision:   {mm.context_precision_batch(qs, ctxs, gt)}")
    print(f"  context_recall:      {mm.context_recall_batch(qs, ctxs, gt)}")
    print(f"  faithfulness:        {mm.faithfulness_batch(qs, ans, ctxs)}")
    print(f"  answer_correctness:  {mm.answer_correctness_batch(qs, ans, gt)}")
    print(f"  answer_relevancy:    {mm.answer_relevancy_batch(qs, ans)}")

    print("\nТолько generation-метрики (как для baseline):")
    print(f"  answer_correctness:  {mm.answer_correctness_batch(qs, ans, gt)}")
    print(f"  answer_relevancy:    {mm.answer_relevancy_batch(qs, ans)}")
