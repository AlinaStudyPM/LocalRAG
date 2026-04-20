import re
from typing import List
from pathlib import Path

import pdfplumber
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.Config import Config

def tiktoken_len(text: str) -> int:
    tokenizer = tiktoken.get_encoding("cl100k_base")  # для most modern models (gpt-3.5/4, gemma, mistral)
                                                            # Для multilingual-e5 — можно использовать "gpt2", но "cl100k_base" тоже подойдёт
    return len(tokenizer.encode(text, disallowed_special=()))

class DocumentProcessor:

    def __init__(self, config: Config):
        self.config = config
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=tiktoken_len
        )
        self._processors = {
            ".pdf": self._process_pdf,
            ".txt": self._process_text,
            ".md": self._process_text,
            ".py": self._process_text,
            ".rs": self._process_text,
            ".cpp": self._process_text,
            ".c": self._process_text,
            ".h": self._process_text,
            ".js": self._process_text,
            ".html": self._process_text,
            ".css": self._process_text,
            ".json": self._process_text,
            ".yaml": self._process_text,
            ".yml": self._process_text,
            ".xml": self._process_text,
            ".png": self._process_image,
            ".jpg": self._process_image,
            ".jpeg": self._process_image,
            ".bmp": self._process_image
        }

    def process(self, file_path: str, quick: bool = True) -> List[str]:
        """
        Определяет расширение файла и вызывает соответсвующий метод
        для извлечения текста и разбиения на чанки.
        """
        ext = Path(file_path).suffix.lower()

        if ext not in self._processors:
            raise ValueError(f"Неподдерживаемый формат: {ext}")
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Файл не найден: {file_path}")

        if ext == ".pdf":
            return self._process_pdf(file_path, quick)
        else:
            return self._processors[ext](file_path)


    def _process_text(self, file_path: str) -> List[str]:
        """Обработка текстовых файлов."""
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        return self.splitter.split_text(self._clean_text(text))

    def _process_image(self, file_path: str) -> List[str]:
        """Обработка изображений через OCR."""
        image = Image.open(file_path)
        text = pytesseract.image_to_string(image, lang='eng+rus')
        return self.splitter.split_text(self._clean_text(text))

    def _process_pdf(self, pdf_file_path: str, quick=True) -> List[str]:
        """Основной пайплайн: текст → чанки."""
        if quick:
            text = self._quick_extract_from_pdf(pdf_file_path)
        else:
            text = self._extract_from_pdf(pdf_file_path)
        chunks = self.splitter.split_text(text) 
        return chunks
    
    def _quick_extract_from_pdf(self, pdf_file_path: str) -> str:
        """Извлекает текст из PDF без OCR — быстро"""
        with pdfplumber.open(pdf_file_path) as pdf:
            texts = []
            for page in pdf.pages:
                text = page.extract_text(x_tolerance=1, y_tolerance=1) or ""
                texts.append(text)
            full_text = "\n".join(texts)
            return self._clean_text(full_text)
        
    def _extract_from_pdf(self, pdf_file_path: str) -> str:
        """Извлекает текст с помощью Tesseract OCR — медленно, но работает для сканов."""
        images = convert_from_path(pdf_file_path)
        ocr_texts = []
        for image in images:
            text = pytesseract.image_to_string(image, lang='eng+rus')
            ocr_texts.append(text)
        full_text = "\n".join(ocr_texts)
        return self._clean_text(full_text)
    
    def _clean_text(self, text: str) -> str:
        text = re.sub(r'-\n\s*', '', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'([.!?])\n', r'\1\n\n', text)
        return text.strip()
