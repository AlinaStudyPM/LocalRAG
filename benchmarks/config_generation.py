# benchmarks/config.py

BENCH_CHROMA_DIR = "./benchmarks/data/chroma_db"
BENCH_DATA_DIR   = "./benchmarks/data/dragon"
PROMPT_DIR       = "./benchmarks/prompts"

GENERATION_MODEL    = "deepseek-r1:latest"
DEFAULT_PROMPT_FILE = "prompt_ru.txt"
EMBEDDING_MODEL     = "intfloat/multilingual-e5-large"
TOP_EMBED           = 15
TOP_RERANK          = 3
TEMPERATURE         = 0.0

JUDGE_MODEL     = "gpt-4o-mini"
OPENROUTER_URL  = "https://openrouter.ai/api/v1"
OLLAMA_URL      = "http://localhost:11434"

DEFAULT_LIMIT = 50
