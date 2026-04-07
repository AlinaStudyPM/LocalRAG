import os
from dotenv import load_dotenv
from pathlib import Path
#import pytesseract

class Config:
    """
    Класс для задания всех конфигураций
    """
    def __init__(self):
        load_dotenv()

        self.SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "./data/users.db")
        self.CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "./data/chroma_db")
        self.UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./data/upload")
        
        self.OLLAMA_DOCKER_URL = os.getenv("OLLAMA_DOCKER_URL")
        self.OLLAMA_LOCAL_URL = os.getenv("OLLAMA_LOCAL_URL")
        self.OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:latest")
        self.OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "qwen3-embedding:0.6b")
        self.chunk_size = int(os.getenv("CHUNK_SIZE", 800))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 150))
        
        prompt_path = os.getenv("SYSTEM_PROMPT_FILE", "config/system_prompt.txt")
        self.SYSTEM_PROMPT = Path(prompt_path).read_text(encoding="utf-8")   
        
        # Tesseract и Poppler
        #self.poppler_path = r"C:\poppler\poppler-25.07.0\Library\bin"
        #self.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        #os.environ["PATH"] += os.pathsep + self.poppler_path
        #pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd
