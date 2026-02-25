# src/FileUploader.py
import os
import tempfile
from pathlib import Path
from typing import Optional, List, Callable
import flet as ft

import logging
logger = logging.getLogger(__name__)

class FileUploader:
    """
    Модуль для загрузки файлов
    """
    def __init__(self, page: ft.Page, upload_dir: Path):
        self.page = page
        self.file_picker = ft.FilePicker()
        self.file_extensions = ["pdf", "txt", "md", "doc", "docx"]
        self.upload_dir = upload_dir
        os.makedirs(upload_dir, exist_ok=True)

        self.selected_files = []

    async def pick_files(self) -> None:
        """
        Открывает диалоговое окно для выбора файлов
        и возвращает список файлов.
        """
        logger.info("Open dialog box for selecting files...")
        try:
            files = await self.file_picker.pick_files(
                allow_multiple=True,
                dialog_title="Выберите файл(-ы)",
                allowed_extensions=self.file_extensions,
                file_type=ft.FilePickerFileType.CUSTOM
            )
            if files:
                self.selected_files.clear()
                for file in files:
                    self.selected_files.append(file)
                logger.info("Success! Files are selected.")
        except Exception as ex:
            logger.exception("Error during selection files: {ex}.")

    def get_files(self) -> List[ft.FilePickerFile]:
        return self.selected_files

    async def upload_files(self):
        """
        Отображает процесс загрузки выбранных файлов.
        """
        logger.info("Start upload_files...")
        upload_files = []
        for file in self.selected_files:
            filename = os.path.basename(file.name)
            upload_url = self.page.get_upload_url(
                filename,
                60  # URL действителен 60 секунд
            )
            upload_files.append(
                ft.FilePickerUploadFile(
                    name=file.name,
                    upload_url=upload_url,
                )
            )
        try:
            await self.file_picker.upload(upload_files)
            logger.info("Successs! Files uploaded.")
        except Exception as ex:
            logger.exception("Error during upload files: {ex}.")



