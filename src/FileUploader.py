# src/FileUploader.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import asyncio

from src.Config import Config
from src.DocumentProcessor import DocumentProcessor
from src.ChromaAdapter import ChromaAdapter

@dataclass
class UploadedFile:
    name: str
    path: Optional[str] 
    size: int

class FileUploaderBase(ABC):
    def __init__(self, config: Config):
        self.config = config
        self.doc_processor = DocumentProcessor(config)
        self.chroma_adapter = ChromaAdapter(config)
        self._selected_files: List = []

    @abstractmethod
    async def pick_files(self) -> None: ...
    
    @abstractmethod
    async def upload_and_process(
        self,
        collection_name: str,
        file_paths: Optional[List[str]] = None
    ) -> List[UploadedFile]: ...
    
    def get_selected_files(self) -> List:
        return self._selected_files



import os
import flet as ft


class FileUploaderWeb(FileUploaderBase):
    """
    Загрузчик файлов для веб-версии.
    Файлы загружаются на сервер, обрабатываются, а потом удаляются.
    """
    def __init__(self, config: Config):
        super().__init__(config)
        self.upload_dir = config.UPLOAD_DIR
        os.makedirs(self.upload_dir, exist_ok=True)
        self._selected_files: List[ft.FilePickerFile] = []
        self._upload_status: Dict[str, asyncio.Event] = {}

    def set_page(self, page: ft.Page, process_label):
        self.page = page
        self.file_picker = ft.FilePicker()
        self.process_label = process_label

        def on_upload_progress(e: ft.FilePickerUploadEvent):
            if e.progress == 1.0:
                self._upload_status[e.file_name].set()
            elif e.error:
                self._upload_status[e.file_name].set()
        self.file_picker.on_upload = on_upload_progress

    async def pick_files(self) -> None:
        """
        Открывает диалоговое окно для выбора файлов
        и возвращает список файлов.
        """
        files = await self.file_picker.pick_files(
            allow_multiple=True,
            dialog_title="Выберите файл(-ы)",
            allowed_extensions=self.config.FILE_EXTENSIONS,
            file_type=ft.FilePickerFileType.CUSTOM
        )
        if files:
            self._selected_files = list(files)
        
    async def upload_and_process(
            self, 
            collection_name: str
        ) -> List[UploadedFile]:
        """
        1. Загружает выбранные файлы временно на сервер. 
        2. Обрабатывает и индексирует в ChromaDB.
        3. Удаляет временные файлы.
        4. Возвращает список с информацией о файлах.
        """
        uploaded = []
        for file in self._selected_files:
            self.process_label.value = "Файлы загружаются..."
            self.page.update()
            saved_path = await self._upload_file(file)
            self.process_label.value = "Файлы распознаются..."
            self.page.update()
            await asyncio.sleep(0)
            chunks = self.doc_processor.process(str(saved_path))
            self.chroma_adapter.add_documents(
                collection_name=collection_name,
                file_name=file.name,
                texts=chunks
            )
            saved_path.unlink(missing_ok=True)
            uploaded.append(UploadedFile(
                name=file.name,
                path=None,
                size=file.size
            ))
            self.process_label.value = ""
            self.page.update()
        self._selected_files = []
        return uploaded

    async def _upload_file(self, file: ft.FilePickerFile) -> Path:
        """
        Получает информацию о выбранном файле и загружает его на сервер.
        """
        filename = os.path.basename(file.name)
        upload_url = self.page.get_upload_url(
            filename,
            60  # URL действителен 60 секунд
        )
        upload_file = ft.FilePickerUploadFile(
            name=filename,
            upload_url=upload_url
        )
        self._upload_status[filename] = asyncio.Event()
        await self.file_picker.upload([upload_file])
        await self._upload_status[filename].wait()
        return Path(self.upload_dir).resolve() / filename


class FileUploaderDesktop(FileUploaderBase):
    """
    Загрузчик файлов для десктоп-версии.
    Получает прямой путь к файлу на устройстве и обрабатывает его.
    """
    def __init__(self, config: Config):
        super().__init__(config)

    def set_page(self, page: ft.Page, process_label):
        self.page = page
        self.file_picker = ft.FilePicker()
        self.process_label = process_label

    async def pick_files(self) -> None:
        """Открывает диалоговое окно для выбора файлов."""
        files = await self.file_picker.pick_files(
            allow_multiple=True,
            dialog_title="Выберите файл(-ы)",
            allowed_extensions=self.config.FILE_EXTENSIONS,
            file_type=ft.FilePickerFileType.CUSTOM
        )
        if files:
            self._selected_files = list(files)

    async def upload_and_process(
        self,
        collection_name: str
    ) -> List[UploadedFile]:
        """
        1. Обрабатывает выбранные файлы.
        2. Индексирует в ChromaDB.
        3. Возвращает список с информацией о файлах.
        """
        uploaded = []
        for file in self._selected_files:
            self.process_label.value = "Файлы распознаются..."
            self.page.update()
            await asyncio.sleep(0)
            chunks = self.doc_processor.process(file.path)
            self.process_label.value = "Обновляется база данных..."
            self.page.update()
            await asyncio.sleep(0)
            self.chroma_adapter.add_documents(
                collection_name=collection_name,
                file_name=file.name,
                texts=chunks
            )
            uploaded.append(UploadedFile(
                name=file.name,
                path=file.path,
                size=file.size
            ))
        self._selected_files = []
        self.process_label.value = ""
        self.page.update()
        return uploaded


class FileUploaderConsole(FileUploaderBase):
    """
    Загрузчик файлов для консольной версии.
    Файлы передаются как пути в метод upload_and_process().
    """
    def __init__(self, config: Config):
        super().__init__(config)
        self._selected_files: List[str] = []

    async def pick_files(self) -> None:
        """
        Для консоли не используется.
        """
        pass

    async def upload_and_process(
        self,
        collection_name: str,
        file_paths: List[str]   
    ) -> List[UploadedFile]:
        """
        1. Валидирует пути к файлам.
        2. Обрабатывает и индексирует в ChromaDB.
        3. Возвращает список с информацией о файлах.
        """
        uploaded = []
        for file_path in file_paths:
            path = Path(file_path)
            
            if not path.exists():
                raise FileNotFoundError(f"Файл не найден: {file_path}")
            
            ext = path.suffix.lower().lstrip(".")
            if ext not in self.config.FILE_EXTENSIONS:
                raise ValueError(
                    f"Неподдерживаемый формат '{ext}'. "
                    f"Допустимые: {', '.join(self.config.FILE_EXTENSIONS)}"
                )
            
            # Обработка
            chunks = self.doc_processor.process(str(path))
            self.chroma_adapter.add_documents(
                collection_name=collection_name,
                file_name=path.name,
                texts=chunks
            )
            
            uploaded.append(UploadedFile(
                name=path.name,
                path=str(path),
                size=path.stat().st_size
            ))
        
        return uploaded
