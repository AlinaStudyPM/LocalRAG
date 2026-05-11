import os

from src.Config import Config
from src.DocumentProcessor import DocumentProcessor

if __name__ == "__main__":
    config = Config()
    doc_processor = DocumentProcessor(config)

    file_path = r"./files/DS_Lection_5.pdf"

    if not os.path.isfile(file_path):
        print(f"❌ Файл не найден: {file_path}")
    else:
        print(f"📄 Обрабатываю: {file_path}")
        try:
            # Быстрое извлечение (без OCR)
            print("\n⚡ Быстрое извлечение (quick=True)...")
            chunks_quick = doc_processor.process_pdf(file_path, quick=True)
            print(f"✅ Получено {len(chunks_quick)} чанков (быстро). Пример двух соседних:")
            print("-" * 50)
            print((chunks_quick[9]))
            print("-" * 50)
            print((chunks_quick[10]))
            print("-" * 50)

            # Полное извлечение (с OCR, если нужен fallback)
            print("\n🔍 Полное извлечение (quick=False — с OCR, медленнее)...")
            chunks_full = doc_processor.process_pdf(file_path, quick=False)
            print(f"✅ Получено {len(chunks_full)} чанков (с OCR). Пример двух соседних:")
            print("-" * 50)
            print((chunks_full[9]))
            print("-" * 50)
            print((chunks_full[10]))
            print("-" * 50)
        except Exception as e:
            print(f"❌ Ошибка при обработке PDF: {e}")
            import traceback
            traceback.print_exc()
    
