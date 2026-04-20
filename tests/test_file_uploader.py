# tests/test_file_uploader.py
import pytest
import os
from pathlib import Path

from src.FileUploader import (
    UploadedFile,
    FileUploaderBase,
    FileUploaderWeb,
    FileUploaderDesktop,
    FileUploaderConsole
)
from src.Config import Config
from src.ChromaAdapter import ChromaAdapter

FIXTURES_DIR = Path(__file__).parent / "fixtures"

@pytest.fixture
def temp_config():
    """Создаёт Config с временной папкой для ChromaDB."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        config = Config()
        config.CHROMA_DB_DIR = tmpdir
        yield config

@pytest.fixture
def created_collection(temp_config):
    """Создаёт Config с коллекцией для тестов загрузки файлов."""
    chroma_adapter = ChromaAdapter(temp_config)
    chroma_adapter.create_collection("test_collection")
    return temp_config

class TestFileUploaderConsole:
    def setup_method(self):
        self.config = Config()
        self.uploader = FileUploaderConsole(self.config)
    @pytest.mark.asyncio
    async def test_upload_pdf(self, created_collection):
        """Успешная загрузка PDF."""
        uploader = FileUploaderConsole(created_collection)
        file_path = str(FIXTURES_DIR / "test.pdf")
        
        result = await uploader.upload_and_process(
            collection_name="test_collection",
            file_paths=[file_path]
        )
        
        assert len(result) == 1
        assert result[0].name == "test.pdf"
        assert result[0].path == file_path
        assert result[0].size > 0
    @pytest.mark.asyncio
    async def test_upload_txt(self, created_collection):
        """Успешная загрузка текстового файла."""
        uploader = FileUploaderConsole(created_collection)
        file_path = str(FIXTURES_DIR / "test.txt")
        
        result = await uploader.upload_and_process(
            collection_name="test_collection",
            file_paths=[file_path]
        )
        
        assert len(result) == 1
        assert result[0].name == "test.txt"
    @pytest.mark.asyncio
    async def test_upload_nonexistent_file(self, created_collection):
        """Ошибка при загрузке несуществующего файла."""
        uploader = FileUploaderConsole(created_collection)
        
        with pytest.raises(FileNotFoundError):
            await uploader.upload_and_process(
                collection_name="test_collection",
                file_paths=["/nonexistent/test.pdf"]
            )
    @pytest.mark.asyncio
    async def test_upload_unsupported_extension(self, created_collection):
        """Ошибка при загрузке файла с неподдерживаемым расширением."""
        uploader = FileUploaderConsole(created_collection)
        file_path = str(FIXTURES_DIR / "test.py")
        
        with pytest.raises(ValueError, match="Неподдерживаемый формат"):
            await uploader.upload_and_process(
                collection_name="test_collection",
                file_paths=[file_path]
            )
    @pytest.mark.asyncio
    async def test_upload_multiple_files(self, created_collection):
        """Успешная загрузка нескольких файлов."""
        uploader = FileUploaderConsole(created_collection)
        
        result = await uploader.upload_and_process(
            collection_name="test_collection",
            file_paths=[
                str(FIXTURES_DIR / "test.pdf"),
                str(FIXTURES_DIR / "test.txt")
            ]
        )
        
        assert len(result) == 2
        names = [r.name for r in result]
        assert "test.pdf" in names
        assert "test.txt" in names

class TestFileUploaderWeb:
    def setup_method(self):
        self.config = Config()
    @pytest.mark.asyncio
    async def test_pick_files_stores_selected(self):
        """pick_files() сохраняет выбранные файлы."""
        from unittest.mock import Mock, patch
        
        mock_page = Mock()
        uploader = FileUploaderWeb(mock_page, self.config)
        
        mock_file = Mock()
        mock_file.name = "test.pdf"
        mock_file.size = 1024
        
        with patch.object(uploader.file_picker, 'pick_files', 
                         return_value=[mock_file]):
            await uploader.pick_files()
        
        assert len(uploader.get_selected_files()) == 1
        assert uploader.get_selected_files()[0].name == "test.pdf"
    @pytest.mark.asyncio
    async def test_upload_and_process_returns_result(self):
        """upload_and_process() возвращает корректный результат."""
        from unittest.mock import Mock, patch, AsyncMock
        import tempfile
        
        mock_page = Mock()
        uploader = FileUploaderWeb(mock_page, self.config)
        
        mock_file = Mock()
        mock_file.name = "test.pdf"
        mock_file.size = 1024
        uploader._selected_files = [mock_file]
        
        # Создаём временный файл для имитации загрузки
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            saved_path = Path(tmp.name)
        
        try:
            with patch.object(uploader, '_upload_file', 
                           return_value=saved_path):
                with patch.object(uploader.doc_processor, 'process',
                               return_value=["chunk1", "chunk2"]):
                    with patch.object(uploader.chroma_adapter, 'add_documents'):
                        result = await uploader.upload_and_process("test_collection")
            
            assert len(result) == 1
            assert result[0].name == "test.pdf"
            assert result[0].path is None  # веб-версия не хранит путь
            assert result[0].size == 1024
        finally:
            saved_path.unlink(missing_ok=True)
    @pytest.mark.asyncio
    async def test_file_deleted_after_upload(self):
        """После загрузки файл удаляется с диска."""
        from unittest.mock import Mock, patch
        import tempfile
        
        mock_page = Mock()
        uploader = FileUploaderWeb(mock_page, self.config)
        
        mock_file = Mock()
        mock_file.name = "test.pdf"
        mock_file.size = 1024
        uploader._selected_files = [mock_file]
        
        # Создаём временный файл
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            saved_path = Path(tmp.name)
            tmp.write(b"test content")
        
        with patch.object(uploader, '_upload_file', return_value=saved_path):
            with patch.object(uploader.doc_processor, 'process', return_value=[]):
                with patch.object(uploader.chroma_adapter, 'add_documents'):
                    await uploader.upload_and_process("test_collection")
        
        assert not saved_path.exists()  # файл удалён

class TestFileUploaderDesktop:
    def setup_method(self):
        self.config = Config()
    @pytest.mark.asyncio
    async def test_pick_files_stores_selected(self):
        """pick_files() сохраняет выбранные файлы."""
        from unittest.mock import Mock, patch
        
        mock_page = Mock()
        uploader = FileUploaderDesktop(mock_page, self.config)
        
        mock_file = Mock()
        mock_file.name = "test.pdf"
        mock_file.path = "/path/test.pdf"
        mock_file.size = 1024
        
        with patch.object(uploader.file_picker, 'pick_files',
                         return_value=[mock_file]):
            await uploader.pick_files()
        
        assert len(uploader.get_selected_files()) == 1
    @pytest.mark.asyncio
    async def test_upload_and_process_keeps_path(self):
        """upload_and_process() сохраняет путь к файлу."""
        from unittest.mock import Mock, patch
        import tempfile
        
        mock_page = Mock()
        uploader = FileUploaderDesktop(mock_page, self.config)
        
        # Создаём реальный файл
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            file_path = tmp.name
            tmp.write(b"test content")
        
        mock_file = Mock()
        mock_file.name = "test.txt"
        mock_file.path = file_path
        mock_file.size = 12
        uploader._selected_files = [mock_file]
        
        try:
            with patch.object(uploader.chroma_adapter, 'add_documents'):
                result = await uploader.upload_and_process("test_collection")
            
            assert len(result) == 1
            assert result[0].name == "test.txt"
            assert result[0].path == file_path  # десктоп сохраняет путь!
        finally:
            Path(file_path).unlink()


