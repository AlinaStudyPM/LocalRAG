# tests/test_document_processor.py
import pytest
from pathlib import Path
import os

from src.Config import Config
from src.DocumentProcessor import DocumentProcessor

FIXTURES_DIR = Path(__file__).parent / "fixtures"
OUTPUT_DIR = Path(__file__).parent / "test_outputs"

def save_chunks_to_file(chunks: list, output_name: str):
    """Сохраняет чанки в файл."""
    output_path = OUTPUT_DIR / f"{output_name}.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(chunks))

class TestDocumentProcessor:
    def setup_method(self):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        self.config = Config()
        self.processor = DocumentProcessor(self.config)
    
    def test_process_unsupported_format_raises(self):
        """Ошибка при неподдерживаемом формате."""
        file_path = str(FIXTURES_DIR / "test_unsupported.xyz")
        with pytest.raises(ValueError, match="Неподдерживаемый формат"):
            self.processor.process(file_path)
        
    def test_process_file_not_found_raises(self):
        """Ошибка при отсутствии файла."""
        file_path = str(FIXTURES_DIR / "nonexistent.pdf")
        with pytest.raises(FileNotFoundError, match="Файл не найден"):
            self.processor.process(file_path)
    

    # === Тесты текстовых файлов ===

    def test_process_txt_file(self):
        """Успешная обработка .txt файла."""
        file_path = str(FIXTURES_DIR / "test.txt")
        chunks = self.processor.process(file_path)
        save_chunks_to_file(chunks, "test_txt_result")
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert all(isinstance(c, str) for c in chunks)

    def test_process_md_file(self):
        """Успешная обработка .md файла."""
        file_path = str(FIXTURES_DIR / "test.md")
        chunks = self.processor.process(file_path)
        save_chunks_to_file(chunks, "test_txt_result")
        assert isinstance(chunks, list)
        assert len(chunks) > 0

    def test_process_py_file(self):
        """Успешная обработка .py файла (код)."""
        file_path = str(FIXTURES_DIR / "test.py")
        chunks = self.processor.process(file_path)
        save_chunks_to_file(chunks, "test_py_result")
        assert isinstance(chunks, list)
        assert len(chunks) > 0


    # === Тесты PDF ===

    def test_process_pdf_quick(self):
        """Быстрая обработка PDF (pdfplumber)."""
        file_path = str(FIXTURES_DIR / "test.pdf")
        chunks = self.processor.process(file_path, quick=True)
        save_chunks_to_file(chunks, "test_pdf_quick_result")
        assert isinstance(chunks, list)
        assert len(chunks) > 0

    def test_process_pdf_full(self):
        """Полная обработка PDF (с OCR)."""
        file_path = str(FIXTURES_DIR / "test.pdf")
        chunks = self.processor.process(file_path, quick=False)
        save_chunks_to_file(chunks, "test_pdf_full_result")
        assert isinstance(chunks, list)
        assert len(chunks) > 0


    # === Тесты изображений ===

    def test_process_image(self):
        """Обработка изображения через OCR."""
        file_path = str(FIXTURES_DIR / "test.jpg")
        chunks = self.processor.process(file_path)
        save_chunks_to_file(chunks, "test_jpg_result")
        assert isinstance(chunks, list)
        # Результат может быть пустым для чистого изображения
