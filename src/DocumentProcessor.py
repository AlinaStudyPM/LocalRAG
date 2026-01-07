import re
from typing import List

import pdfplumber
from pdf2image import convert_from_path
import pytesseract
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from Config import Config

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

    def process_pdf(self, pdf_file_path: str, quick=True) -> List[str]:
        """Основной пайплайн: текст → чанки."""
        if quick:
            text = self._quick_extract_text(pdf_file_path)
        else:
            text = self._extract_text(pdf_file_path)
        chunks = self.splitter.split_text(text) 
        return chunks
    
    def _quick_extract_text(self, pdf_file_path: str) -> str:
        """Извлекает текст из PDF без OCR — быстро"""
        with pdfplumber.open(pdf_file_path) as pdf:
            texts = []
            for page in pdf.pages:
                text = page.extract_text(x_tolerance=1, y_tolerance=1) or ""
                texts.append(text)
            full_text = "\n".join(texts)
            return self._clean_text(full_text)
        
    def _extract_text(self, pdf_file_path: str) -> str:
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
