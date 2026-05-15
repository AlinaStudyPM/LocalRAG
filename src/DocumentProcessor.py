import re
import html
from typing import List
from pathlib import Path

import pdfplumber
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter
from odfdo import Document
from markitdown import MarkItDown
from charset_normalizer import detect

from src.Config import Config

class DocumentProcessor:

    def __init__(self, config: Config):
        self._config = config        
        self._tokenizer = tiktoken.get_encoding("cl100k_base")
        self._md = MarkItDown()
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._config.chunk_size,
            chunk_overlap=self._config.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=self._tiktoken_len
        )
        self._processors = {
            ".pdf": self._process_pdf,
            ".txt": self._process_text,
            ".md": self._process_text,
            ".html": self._process_html,
            ".htm": self._process_html,
            ".odt": self._process_odt,
            ".docx": self._process_docx,
            ".pptx": self._process_pptx,
            ".png": self._process_image,
            ".jpg": self._process_image,
            ".jpeg": self._process_image,
            ".bmp": self._process_image,
            ".py": self._process_code,
            ".rs": self._process_code,
            ".cpp": self._process_code,
            ".c": self._process_code,
            ".h": self._process_code,
            ".js": self._process_code,
            ".css": self._process_code,
            ".json": self._process_code,
            ".yaml": self._process_code,
            ".yml": self._process_code,
            ".xml": self._process_code,
        }

    def get_supported_extensions(self) -> List[str]:
        """Возвращает список поддерживаемых расширений файлов."""
        exts = list(self._processors.keys())
        return [ext.lstrip(".") for ext in exts]

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
        raw = Path(file_path).read_bytes()
        text = None
        for enc in ("utf-8", "utf-8-sig"):
            try:
                text = raw.decode(enc)
                break
            except UnicodeDecodeError:
                continue

        if text is None:
            result = detect(raw)
            encoding = result.get("encoding") or "cp1251"
            text = raw.decode(encoding, errors="replace")
        return self._splitter.split_text(self._clean_text(text))

    def _process_code(self, file_path: str) -> List[str]:
        """Обработка файлов с кодом."""
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        text = re.sub(r'[\ufeff\u200b\u200c\u200d]', '', text)
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        text = re.sub(r'[ \t]+$', '', text, flags=re.MULTILINE)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = text.strip()
        return self._splitter.split_text(text)

    def _process_image(self, file_path: str) -> List[str]:
        """Обработка изображений через OCR."""
        with Image.open(file_path) as image:
            text = pytesseract.image_to_string(image, lang='eng+rus')
        return self._splitter.split_text(self._clean_text(text))

    def _process_pdf(self, pdf_file_path: str, quick=True) -> List[str]:
        """Обработка PDF."""
        if quick:
            text = self._quick_extract_from_pdf(pdf_file_path)
        else:
            text = self._extract_from_pdf(pdf_file_path)
        chunks = self._splitter.split_text(text) 
        return chunks
    
    def _quick_extract_from_pdf(self, pdf_file_path: str) -> str:
        """Обработка PDF с помощью pdfplumber (быстро)."""
        with pdfplumber.open(pdf_file_path) as pdf:
            texts = []
            for page in pdf.pages:
                text = page.extract_text(x_tolerance=1, y_tolerance=1) or ""
                texts.append(text)
            full_text = "\n".join(texts)
            return self._clean_text(full_text)
        
    def _extract_from_pdf(self, pdf_file_path: str) -> str:
        """Обработка PDF с помощью Tesseract OCR."""
        ocr_texts = []
        with pdfplumber.open(pdf_file_path) as pdf:
            total_pages = len(pdf.pages)

        batch_size = 5
        for start in range(1, total_pages + 1, batch_size):
            end = min(start + batch_size - 1, total_pages)
            images = convert_from_path(pdf_file_path, first_page=start, last_page=end)
            for image in images:
                text = pytesseract.image_to_string(image, lang='eng+rus')
                ocr_texts.append(text)
        full_text = "\n".join(ocr_texts)
        return self._clean_text(full_text)

    def _process_html(self, file_path: str) -> List[str]:
        """Обработка html."""
        result = self._md.convert(file_path)
        text = result.text_content if result.text_content else ""
        return self._splitter.split_text(self._clean_text(text))
    
    def _process_odt(self, file_path: str) -> List[str]:
        """Обработка ODF: преобразование в Markdown через odfdo."""
        document = Document(file_path)
        md_content = document.to_markdown()
        text = md_content if md_content else ""
        return self._splitter.split_text(self._clean_text(text))

    def _process_docx(self, file_path: str) -> List[str]:
        """Обработка DOCX: преобразование в Markdown через markitdown."""
        result = self._md.convert(file_path)
        text = result.text_content if result.text_content else ""
        return self._splitter.split_text(self._clean_text(text))

    def _process_pptx(self, file_path: str) -> List[str]:
        """Обработка PPTX: преобразование в Markdown через markitdown."""
        result = self._md.convert(file_path)
        text = result.text_content if result.text_content else ""
        return self._splitter.split_text(self._clean_text(text))

    def _clean_text(self, text: str) -> str:
        """Очистка текста через регулярные выражения."""
        text = re.sub(r'\r\n?', '\n', text)
        text = html.unescape(text)
        text = re.sub(r'[\ufeff\u200b\u200c\u200d\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        text = re.sub(r'(\w+)-\n[ \t]*(\w+)', r'\1\2', text, flags=re.UNICODE)
        text = re.sub(r'(?m)^[=\-*_]{4,}\s*$', '', text)
        text = re.sub(r'(?:\n[ \t\u00a0]*){3,}', '\n\n', text)
        text = re.sub(r'[ \t\u00a0\u2000-\u200a]+', ' ', text)
        text = re.sub(r'([.!?])\n', r'\1\n\n', text)
        return text.strip() 

    def _tiktoken_len(self, text: str) -> int:
        return len(self._tokenizer.encode(text, disallowed_special=()))

