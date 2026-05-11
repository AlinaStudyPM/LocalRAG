# benchmarks/config.py
"""Конфигурация для оценки RAG"""

import os

LLM_PROVIDER = os.getenv("EVAL_LLM_PROVIDER", "ollama")  # openai | openrouter | ollama
OPENAI_API_KEY = os.getenv("EVAL_OPENAI_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("EVAL_OPENROUTER_API_KEY", "")
MODEL = os.getenv("EVAL_MODEL", "llama3.2:latest")
OLLAMA_URL = os.getenv("OLLAMA_LOCAL_URL", "http://localhost:11434")
CHROMA_DIR = os.getenv("EVAL_CHROMA_DIR", "./data/chroma_benchmark")
RESULTS_DIR = os.getenv("EVAL_RESULTS_DIR", "./benchmarks/data")
EMBEDDINGS_MODEL = "BAAI/bge-small-en-v1.5"

DEFAULT_N_SAMPLES = 100
