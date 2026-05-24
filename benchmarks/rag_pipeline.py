# benchmarks/rag_pipeline.py
import shutil
from pathlib import Path
from typing import List, Optional

from src.Config import Config
from src.ChromaAdapter import ChromaAdapter
from src.ChatAgent import ChatAgent


PROMPT_DIR = Path(__file__).parent / "prompts"


class BenchRAG:

    def __init__(
        self,
        chroma_subdir: str,
        model_name: str,
        prompt_file: str,
        embedding_model: str,
    ):
        self.cfg = Config()
        self.cfg.CHROMA_DB_DIR = str(Path("./benchmarks/data/chroma_db") / chroma_subdir)
        self.cfg.OLLAMA_MODEL = model_name
        self.cfg.EMBEDDING_MODEL = embedding_model

        prompt_path = PROMPT_DIR / prompt_file
        self.cfg.SYSTEM_PROMPT = prompt_path.read_text(encoding="utf-8")

        self.chroma = ChromaAdapter(self.cfg)
        self.agent = ChatAgent(self.cfg, self.chroma)

    def index_passages(
        self,
        passages: List[str],
        collection_name: str,
        source_name: str = "dragon",
    ) -> None:
        self.chroma.create_collection(collection_name)
        self.chroma.add_documents(
            collection_name=collection_name,
            file_name=source_name,
            texts=passages,
        )

    def retrieve(
        self,
        query: str,
        collection_name: str,
        top_embed: Optional[int] = None,
        top_rerank: Optional[int] = None,
        use_rerank: bool = True
    ) -> List[str]:
        raw = self.chroma.search(collection_name, query, top_embed, top_rerank, use_rerank)
        return raw["documents"][0] if raw.get("documents") else []

    def generate_rag(
        self,
        question: str,
        contexts: List[str],
        temperature: float = 0.7,
    ) -> str:
        data = {
            "documents": [contexts],
            "metadatas": [[{"source": "dragon"}] * len(contexts)],
        }
        context_block = self.agent.format_results(data, n=len(contexts))
        return self.agent.ollama_query(
            input_text=question,
            history=[],
            context=context_block,
            temperature=temperature,
        )

    def generate_baseline(
        self,
        question: str,
        temperature: float = 0.7,
    ) -> str:
        return self.agent.ollama_query(
            input_text=question,
            history=[],
            context=None,
            temperature=temperature,
        )

    def cleanup(self) -> None:
        db_path = Path(self.cfg.CHROMA_DB_DIR)
        if db_path.exists():
            shutil.rmtree(db_path)

    @staticmethod
    def _format_contexts(contexts: List[str]) -> str:
        lines = ["\n=== РЕЛЕВАНТНАЯ ИНФОРМАЦИЯ ===\n"]
        for i, ctx in enumerate(contexts, 1):
            text = ctx[:1000] + "..." if len(ctx) > 1000 else ctx
            lines.append(f"\nФрагмент {i}: {text}")
        lines.append("\n=== КОНЕЦ КОНТЕКСТА ===\n")
        return "\n".join(lines)


if __name__ == "__main__":
    print("Проверка BenchRAG")

    rag = BenchRAG(
        chroma_subdir="test_bge",
        model_name="llama3.2:latest",
        prompt_file="prompt_ru.txt",
        embedding_model="BAAI/bge-small-en-v1.5",
    )

    passages = [
        "Париж — столица Франции и самый густонаселённый город страны.",
        "В 2024 году Париж принимал Олимпийские игры.",
        "Эйфелева башня была построена в 1889 году.",
    ]

    print("\n[1] Индексация...")
    rag.index_passages(passages, collection_name="test")

    print("[2] Retrieval...")
    contexts = rag.retrieve("Когда построили Эйфелеву башню?", "test", top_k=2)
    for i, c in enumerate(contexts):
        print(f"   {i+1}. {c[:60]}...")

    print("[3] RAG-ответ...")
    rag_ans = rag.generate_rag("Когда построили Эйфелеву башню?", contexts)
    print(f"   {rag_ans[:100]}...")

    print("[4] Baseline...")
    base_ans = rag.generate_baseline("Когда построили Эйфелеву башню?")
    print(f"   {base_ans[:100]}...")

    print("[5] Очистка...")
    rag.cleanup()
    print("   OK")
