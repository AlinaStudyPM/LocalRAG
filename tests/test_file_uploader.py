# tests/test_file_uploader.py
import os
import flet as ft
import logging
from typing import List
from pathlib import Path
from src.FileUploader import FileUploader

UPLOAD_DIR_STR: str = "./upload"
UPLOAD_DIR_PATH: Path = Path(UPLOAD_DIR_STR)

logging.basicConfig(
    level=logging.INFO,
    format='%(name)s - %(levelname)s - %(message)s'
)

async def main(page: ft.Page):
    os.environ["FLET_SECRET_KEY"] = "my-secret-key-24022026"

    file_uploader = FileUploader(page, UPLOAD_DIR_STR)

    async def on_click1(e: ft.ControlEvent):
        await file_uploader.pick_files()
        selected_files: List[ft.FilePickerFile] = file_uploader.get_files()
        for flet_file in selected_files:
            print(flet_file.name)



    btn = ft.Button("Выбрать файлы", on_click=on_click1)
    btn2 = ft.Button("Загрузить файлы", on_click=file_uploader.upload_files)
    sf_view = ft.ListView(expand=True, spacing=5, height=50)
    page.controls.append(btn)
    page.controls.append(btn2)
    page.controls.append(sf_view)
    page.update()
    

    
if __name__ == "__main__":
    ft.run(
        main,
        view=ft.AppView.WEB_BROWSER,
        upload_dir=UPLOAD_DIR_STR,
        port=8550,
    )

