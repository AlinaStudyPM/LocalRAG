# benchmarks/config_rerank.py
"""
Конфигурация для RQ3: влияние реранкера (top_rerank) при фиксированном top_embed.
"""

BENCH_CHROMA_DIR = "./benchmarks/data/chroma_db"
BENCH_DATA_DIR   = "./benchmarks/data/dragon"
PROMPT_DIR       = "./benchmarks/prompts"

EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
TOP_EMBED_FIXED = 15
TOP_RERANK_VALUES = [1, 3, 5, 10, 15]

GENERATION_MODEL    = "deepseek-r1:latest"
DEFAULT_PROMPT_FILE = "prompt_ru.txt"
TEMPERATURE         = 0.0

JUDGE_MODEL     = "gpt-4o-mini"
DEFAULT_LIMIT = 10
