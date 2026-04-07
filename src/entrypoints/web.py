# src/entrypoints/web.py
from src.RAGApp import RAGApp

def main():
    app = RAGApp()
    app.run_web()

