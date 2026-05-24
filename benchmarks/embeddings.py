# benchmarks/embeddings.py
import asyncio
from typing import List

from fastembed import TextEmbedding
from ragas.embeddings.base import BaseRagasEmbedding


class FastEmbedRagas(BaseRagasEmbedding):
    def __init__(
        self,
        model_name: str,
        cache_dir: str = "./.fastembed_cache",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    ):
        super().__init__()
        self.model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)

    def embed_text(self, text: str, **kwargs) -> List[float]:
        # fastembed.embed() возвращает итератор; берём первый элемент
        # return list(self.model.embed([text]))[0]
        return next(self.model.embed([text]))

    def embed_texts(self, texts: List[str], **kwargs) -> List[List[float]]:
        return list(self.model.embed(texts))

    async def aembed_text(self, text: str, **kwargs) -> List[float]:
        # fastembed синхронный — запускаем в пуле потоков
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_text, text)

    async def aembed_texts(self, texts: List[str], **kwargs) -> List[List[float]]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_texts, texts)

def ensure_embedding_cache(cache_dir: str = "./.fastembed_cache") -> bool:
    """
    Проверяет наличие локального кэша fastembed.
    Возвращает True если кэш есть, иначе печатает предупреждение.
    """
    cache_path = Path(cache_dir)
    if cache_path.exists():
        print(f"Кэш fastembed найден: {cache_path}")
        return True

    print(f"⚠️  Кэш fastembed не найден ({cache_dir}). Модели будут скачаны при первом использовании.")
    return False

if __name__ == "__main__":
    print("Проверка FastEmbedRagas:")
    emb = FastEmbedRagas("BAAI/bge-small-en-v1.5")

    vec = emb.embed_text("Привет, мир!")
    print(f"   embed_text:   len={len(vec)}, вектор: {vec}")

    batch = emb.embed_texts(["Кошка сидит на окне.", "Собака бегает во дворе."])
    print(f"   embed_texts: количество={len(batch)}, размер={len(batch[0])}")

    print("Проверка пройдена.")
