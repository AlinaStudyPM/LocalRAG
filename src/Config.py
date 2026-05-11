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
        self.UPLOAD_DIR = os.getenv("UPLOAD_DIR", os.path.abspath("./data/upload"))
        extensions_str = os.getenv("FILE_EXTENSIONS", "pdf,txt,md")
        self.FILE_EXTENSIONS = [ext.strip() for ext in extensions_str.split(",")]

        self.OLLAMA_DOCKER_URL = os.getenv("OLLAMA_DOCKER_URL")
        self.OLLAMA_LOCAL_URL = os.getenv("OLLAMA_LOCAL_URL")
        self.OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:latest")
        self.EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
        self.chunk_size = int(os.getenv("CHUNK_SIZE", 800))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 150))
        
        prompt_path = os.getenv("SYSTEM_PROMPT_FILE", "config/system_prompt.txt")
        self.SYSTEM_PROMPT = Path(prompt_path).read_text(encoding="utf-8")   
        
        self.LOG_LEVEL = "DEBUG" # INFO, WARNING, ERROR
        self.LOG_TO_FILE = True
        self.LOG_TO_SESSION_FILE = True
        self.LOG_TO_CONSOLE = False
        self.LOG_FILE_PATH = "logs/app.log" 
        self.LOG_SESSION_FILE_PATH = "logs/session.log"

        # Tesseract и Poppler
        #self.poppler_path = r"C:\poppler\poppler-25.07.0\Library\bin"
        #self.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        #os.environ["PATH"] += os.pathsep + self.poppler_path
        #pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd
