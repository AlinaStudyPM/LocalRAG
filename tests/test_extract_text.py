import os

from src.Config import Config
from src.DocumentProcessor import DocumentProcessor

if __name__ == "__main__":
    config = Config()
    doc_processor = DocumentProcessor(config)

    file_path = r"./data/files/Teoriya_avtomatov.pdf"
    result_quick_path = r"./data/files/output_quick.txt"
    result_ocr_path = r"./data/files/output_ocr.txt"

    if not os.path.isfile(file_path):
        print(f"❌ Файл не найден: {file_path}")
    else:
        print(f"📄 Обрабатываю: {file_path}")
        try:
            # Быстрое извлечение (без OCR)
            print("\n⚡ Быстрое извлечение (quick=True)...")
            text_quick = doc_processor._clean_text(doc_processor._quick_extract_text(file_path))
            with open(result_quick_path, 'w', encoding='utf-8') as f:
                f.write(text_quick)
            print(f"✅ Текстовый слой изdлечён. Результат в файле {result_quick_path}.")

            # Полное извлечение (с OCR, если нужен fallback)
            print("\n🔍 Полное извлечение (quick=False — с OCR, медленнее)...")
            text_ocr = doc_processor._clean_text(doc_processor._extract_text(file_path))
            with open(result_ocr_path, 'w', encoding='utf-8') as f:
                f.write(text_ocr)
            print(f"✅ Файл распознан OCR. Результат в {result_ocr_path}.")
            
            
        except Exception as e:
            print(f"❌ Ошибка при обработке PDF: {e}")
            import traceback
            traceback.print_exc()

