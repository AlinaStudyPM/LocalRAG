# Оценка работы RAG

Данный модуль оценивает работу приложения, используя подход LLM-as-Judge. 

Для реализации используется фреймворк RAGAS и датасет Dragon.

Сравниваются ответы локальной языковой модели и RAG-пайплайна.

### Retrieval-метрики

Базовая конфигурация эксперимента находится в файле `benchmarks/config_retrieval.py`.
Для запуска в корне проекта необходимо выполнить команду:
```bash
python benchmarks/eval_retrieval_dragon.py
```
Результаты сохраняются в подпапке benchmarks/results/retrieval/ с временной меткой прогона. Для каждого запуска генерируются:
| Файл                                     | Содержимое                                                   |
| ---------------------------------------- | ------------------------------------------------------------ |
| `YYYYMMDD_HHMMSS_config.json`            | Параметры эксперимента                                       |
| `YYYYMMDD_HHMMSS_samples.jsonl`          | Сырые данные по каждому вопросу: контексты, метрики, время   |
| `YYYYMMDD_HHMMSS_retrieval_summary.json` | Средние значения метрик по моделям и top-k                   |
| `YYYYMMDD_HHMMSS_models_comparison.png`  | Столбчатая диаграмма precision/recall для каждой модели      |
| `YYYYMMDD_HHMMSS_topk_comparison.png`    | Столбчатая диаграмма precision/recall для top-k 3/5/10       |
| `YYYYMMDD_HHMMSS_retrieval_report.md`    | Markdown-отчёт с таблицами и выводами                        |

### Generation-метрики

Базовая конфигурация эксперимента находится в файле `benchmarks/config_generation.py`.
Для запуска в корне проекта необходимо выполнить команду:
```python
python benchmarks/eval_generation_dragon.py
```
Результаты будут в папке `benchmarks/results/generation`.


### Архитектура

##### `embeddings.py`
Реализация интерфеса эмбеддингов на основе базового класса `BaseRagasEmbedding`. Содержит методы `embed_text()`, `embed_texts() и их асинхронные аналоги `aembed_text()` и `aembed_texts().
```python
from fastembed import TextEmbedding
from ragas.embeddings.base import BaseRagasEmbedding

class FastEmbedRagas(BaseRagasEmbedding):
    def __init__(self, model_name, cache_dir: str = "./.fastembed_cache"):
        super().__init__()
        self.model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)
```

##### `judge.py`
Утилиты для подготовки LLM-клиента, выполняющего роль судьи. Содержит методы `get_judge_ollama()` и `get_judge_openrouter()`.
```python
from openai import OpenAI, AsyncOpenAI
from ragas.llms import llm_factory

llm = llm_factory(
    model=model,
    client=client,
    adapter="instructor",
    temperature=temperature,
)
```

##### `metrics.py`
Менеджер метрик. Инициализирует классы метрик и для каждой метрики имеет соответсвующий метод, запускающий её.
```python
from ragas.metrics.collections import (
    ContextPrecision,
    ContextRecall,
    Faithfulness,
    AnswerCorrectness,
    AnswerRelevancy,
)

from benchmarks.embeddings import FastEmbedRagas
from benchmarks.judge import get_judge

class MetricsManager:
    def __init__(self, judge, embeddings):
        self.judge = judge
        self.embeddings = embeddings

        self._ctx_precision = ContextPrecision(llm=judge)
        self._ctx_recall = ContextRecall(llm=judge)
        self._faithfulness = Faithfulness(llm=judge)
        self._answer_correctness = AnswerCorrectness(llm=judge, embeddings=embeddings)
        self._answer_relevancy = AnswerRelevancy(llm=judge, embeddings=embeddings)

```

##### `rag_pipeline.py`
import Config
import ChromaAdapter - для создания отдельной коллекции
import ChatAgent - для общения с нейросетью

Подмена Config.CHROMA_DB_DIR на benchmarks/data/chroma_db/<embedding_model>.
Подмена cfg.SYSTEM_PROMPT перед созданием ChatAgent.


##### `datasets.py`
Утилиты для работы с датасетами.
Для DRAGON: download_dragon() скачивает с HuggingFace, load_dragon_data() загружает корпус и QA из локального кэша с опциональным лимитом.
```python
from datasets import load_dataset

DATA_DIR = Path(__file__).parent / "data"

def def download_dragon():
    pass
```

##### `results_utils.py
`log_config()` - сохраняет JSON с параметрами прогона
`log_sample()` - дописывает в JSONL один QA-пример: question, ground_truth, retrieved_contexts, rag_answer, baseline_answer, scores.
Содержит RunLogger для логирования прогонов, ResultsAggregator для агрегации метрик, ResultsVisualizer для графиков сравнения RAG и Baseline, и RetrievalResultsVisualizer для графиков retrieval-экспериментов.

##### `config_retrieval.py`
Содержит конфигурацию для измерения retrieval-метрик: context precision и context recall.
```python
BENCH_CHROMA_DIR = "./benchmarks/data/chroma_db"
BENCH_DATA_DIR   = "./benchmarks/data/dragon"

EMBEDDING_MODELS = [
    "intfloat/multilingual-e5-large",
    "nomic-ai/nomic-embed-text-v1.5",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
]
DEFAULT_TOP_K_FOR_MODELS = 5
TOP_K_VALUES = [3, 5, 10]

DEFAULT_LIMIT = 10

JUDGE_LOCAL_MODEL      = "qwen2.5:7b-instruct"
JUDGE_OPENROUTER_MODEL = "meta-llama/llama-3.3-70b-instruct:free"
OPENROUTER_URL         = "https://openrouter.ai/api/v1"
OLLAMA_URL             = "http://localhost:11434"
```

##### `config_generation.py`
Содержит конфигурацию для измерения generation-метрик: faithfulness, answer correctness и answer relevancy.
```python
BENCH_CHROMA_DIR = "./benchmarks/data/chroma_db"
BENCH_DATA_DIR   = "./benchmarks/data/dragon"
PROMPT_DIR       = "./benchmarks/prompts"

DEFAULT_PROMPT_FILE = "prompt_ru.txt"
EMBEDDING_MODEL     = "intfloat/multilingual-e5-large"
TOP_K               = 5
TEMPERATURE         = 0.7

JUDGE_LOCAL_MODEL      = "qwen2.5:7b-instruct"
JUDGE_OPENROUTER_MODEL = "meta-llama/llama-3.3-70b-instruct:free"
OPENROUTER_URL         = "https://openrouter.ai/api/v1"
OLLAMA_URL             = "http://localhost:11434"
```

##### `eval_retrieval_dragon.py`
Скрипт для запуска измерения retrieval-метрик (context precision и context recall) на датасете Dragon.

##### `eval_dragon.py`
Скрипт для запуска измерения generation-метрик (faithfulness, answer correctness и answer relevancy) на датасете Dragon.

