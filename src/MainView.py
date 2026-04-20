# src/MainView.py
import uuid
import tempfile
#import shutil
from pathlib import Path
import flet as ft
import asyncio

from src.CoreApp import CoreApp
from src.FileUploader import FileUploaderBase, FileUploaderWeb, FileUploaderDesktop

AppColors = {
    "user_message": ft.Colors.BLUE_ACCENT,
    "assistant_message": ft.Colors.GREY_300,
    "chat_list": ft.Colors.LIGHT_BLUE_50,
    "hovered_chat": ft.Colors.LIGHT_BLUE_100
}

class MainView(ft.View):
    def __init__(self, on_logout):
        super().__init__(route="/main")
        self.on_logout = on_logout
        self._active_chat_button = None

        # --- Header ---
        self.header = ft.Row(
            [
                ft.Text("NeuroKnowledge", size=24, weight="bold"),
                ft.Container(expand=True),
                ft.ElevatedButton("Выйти", on_click=self.handle_logout)
            ],
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
            height=50
        )

        # --- Dialog list ---
        self.button_new_chat = ft.ElevatedButton("+ Новый чат", height=40, on_click=self.create_new_chat)
        self.dialog_list = ft.Container(
            content=ft.Column([self.button_new_chat], spacing=5, horizontal_alignment=ft.CrossAxisAlignment.STRETCH),
            width=250,
            bgcolor=AppColors["chat_list"],
            padding=10
        )

        # ---- Message List ------
        self.message_list = ft.ListView([], expand=True, spacing=5, auto_scroll=True)

        # ---- Message Line ----
        self.models_dropdown = ft.Dropdown(
            label="Модель",
            dense=True,
            width=180
        )
        self.models_dropdown.on_change = self.on_model_change
        self.message_line = ft.TextField(expand=True, hint_text="Введите сообщение")
        self.message_button = ft.IconButton(ft.Icons.SEND, on_click=self.send_message)

        self.dialog_area = ft.Column(
            [
                self.message_list,
                ft.Row(
                    [
                        self.models_dropdown,
                        self.message_line,
                        self.message_button
                    ]
                )
            ],
            expand=True
        )

        # ----- Collections Select ------
        self.collection_title = ft.Text("Коллекции", size=18, weight="bold")
        self.list_collections = ft.ListView()
        self.button_new_coll = ft.ElevatedButton("Создать", icon=ft.Icons.ADD, on_click=self.on_create_collection)
        self.button_download = ft.ElevatedButton("Загрузить", icon=ft.Icons.UPLOAD_FILE, on_click=self.on_upload_file)
        #self.button_all_upload = ft.ElevatedButton("Загрузить тестовые файлы", on_click=self.test_upload_files)
        self.text_list_files = ft.Column(spacing=2)
        self.collections_box = ft.Container(
            content=ft.Column([
                self.collection_title,
                self.list_collections,
                self.button_new_coll,
                self.button_download,
                #self.button_all_upload,
                #self.text_list_files
            ], spacing=5, horizontal_alignment=ft.CrossAxisAlignment.STRETCH),
            width=200,
            padding=10
        )

        self.collection_name_field = ft.TextField(
            label="Название коллекции",
            # hint_text="Например: Отчёты 2025",
            width=300
        )
        self.create_collection_dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text("Создать коллекцию"),
            content=self.collection_name_field,
            actions=[
                ft.TextButton("Отмена", on_click=self.close_dialog),
                ft.ElevatedButton("Создать", on_click=self.do_create_collection),
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )

        self.upload_collection_dd = ft.Dropdown(
            label="Выберите коллекцию",
            dense=True,
            width=300
        )
        self.pick_files_btn = ft.ElevatedButton(
            "Выбрать файл(ы)…",
            icon=ft.Icons.FOLDER_OPEN,
            on_click=self.open_pickfiles
        )
        self.files_label = ft.Text("Файлы не выбраны", size=12, italic=True)
        self.button_filepick_cancel = ft.TextButton("Отмена", on_click=self.close_upload_dialog)
        self.button_filepick_upload = ft.ElevatedButton("Загрузить", on_click=self.do_upload_files)
        self.upload_dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text("Загрузить файл в коллекцию"),
            content=ft.Column([
                self.upload_collection_dd,
                self.pick_files_btn,
                self.files_label
            ], tight=True, spacing=10),
            actions=[
                self.button_filepick_cancel,
                self.button_filepick_upload
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )

        # --- Main part of Page ----
        self.body = ft.Row(
            [
                self.dialog_list,
                self.dialog_area,
                self.collections_box
            ],
            expand=True,
            spacing=10
        )

        # --- Page -------
        self.controls = [
            ft.Column([self.header, self.body], expand=True)
        ]

    def start(self, core: CoreApp, uploader: FileUploaderBase, current_user):
        self.core = core
        self.file_uploader = uploader
        self.file_uploader.set_page(self.page)
        # current_user.load_chats()
        self.current_user_id = current_user.user_id
        self.current_chat_id = None
        
        self.message_line.disabled = True
        self.message_button.disabled = True
        self.update()

        self.load_user_chats()
        self.load_models()
        self.load_collections()

        self.page.overlay.append(self.create_collection_dialog)
        self.page.overlay.append(self.upload_dialog)

        self.message_line.disabled = False
        self.message_button.disabled = False
        self.update()

    # Выход из аккаунта
    def handle_logout(self, e):
        self.core.logout_user(self.current_user_id)
        self.on_logout()

    # Загрузка списка чатов
    def load_user_chats(self, current_chat = None):
        user_chats = self.core.list_user_chats(self.current_user_id)

        if len(user_chats) == 0:
            new_id = f"chat_{uuid.uuid4().hex[:8]}"
            if self.core.create_chat_for_user(self.current_user_id, chat_id=new_id):
                user_chats = self.core.list_user_chats(self.current_user_id)
        self.current_chat_id = user_chats[-1]

        self.dialog_list.content.controls.clear()
        self.dialog_list.content.controls.append(self.button_new_chat)

        for chat_id in user_chats:
            btn = self._display_chat_list_item(chat_id)
            if chat_id == self.current_chat_id:
                btn.bgcolor = AppColors["hovered_chat"]
                self._active_chat_button = btn

            self.dialog_list.content.controls.append(btn)

        self.dialog_list.update()
        self._load_chat_history()

    # Выбор чата
    def on_chat_click(self, e: ft.ControlEvent):
        chat_id = e.control.content
        self.current_chat_id = chat_id

        # Меняем подсветку
        if self._active_chat_button is not None:
            self._active_chat_button.style = self._btn_style(False)
            self._active_chat_button.update()
        e.control.style = self._btn_style(True)
        e.control.update()
        self._active_chat_button = e.control

        self._load_chat_history()

    def _load_chat_history(self):
        history = self.core.get_chat_history(self.current_user_id, self.current_chat_id)

        self.message_list.controls.clear()
        for msg in history:
            self.message_list.controls.append(
                self._display_message(msg["role"], msg["content"])
            )
        asyncio.create_task(self.message_list.scroll_to(delta=-1))
        self.message_list.update()
        
    def load_models(self):
        available_models = self.core.list_available_models()
        self.models_dropdown.options = [
            ft.dropdown.Option(name) for name in available_models
        ]
        current_model = self.core.get_current_model()
        if current_model in available_models:
            self.models_dropdown.value = current_model
        else:
            self.models_dropdown.value = available_models[0] if available_models else None
            self.core.set_current_model(self.models_dropdown.value)
        self.models_dropdown.update()

    def on_model_change(self, e):
        new_model = e.control.value
        if new_model:
            self.core.set_current_model(new_model)

    def load_collections(self):
        
        collections = self.core.list_user_collections(self.current_user_id)
        self.list_collections.controls.clear()
        for col in collections:
            cb = ft.Checkbox(label=col, value=False)
            self.list_collections.controls.append(cb)
        
        """
        collections = self.core.list_user_files(self.current_user_id)
        self.list_collections.controls.clear()
        self.text_list_files.controls.clear()
        self.text_list_files.controls.append(ft.Text("*Список файлов:*", size=14))
        for coll_name in collections.keys():
            cb = ft.Checkbox(label=coll_name, value=False)
            self.list_collections.controls.append(cb)
            self.text_list_files.controls.append(ft.Text(f"{coll_name}:", size=14))
            for file in collections[coll_name]:
                self.text_list_files.controls.append(ft.Text(f"-{file}", size=14))"""
        
        """
        collections = self.core.list_user_files(self.current_user_id)
        self.list_collections.controls.clear()
        for coll_name in collections.keys():
            cb = ft.Button(label=coll_name, value=False)"""

    async def open_pickfiles(self, e: ft.ControlEvent):
        """Открывает окно файлового менеджера для загрузки файлов"""
        await self.file_uploader.pick_files()
        selected_files: List[ft.FilePickerFile] = self.file_uploader.get_selected_files()
        if len(selected_files) > 0:
            names = [f.name for f in selected_files]
            self.files_label.value = f"Выбрано: {', '.join(names)}"
        else:
            self.files_label.value = "Файлы не выбраны"
        self.upload_dialog.update()


    def on_file_picked(self, e: ft.FilePickerUploadEvent):
        """Обрабатывает выбранные файлы."""
        if e.files:
            # paths = [f.path for f in e.files]
            names = [f.name for f in e.files]
            self.files_label.value = f"Выбрано: {', '.join(names)}"
        else:
            self.files_label.value = "Файлы не выбраны"
        self.upload_dialog.update()

    # 3 метода для окна Создания коллекции
    def on_create_collection(self, e):
        """Открывает диалог создания коллекции."""
        self.collection_name_field.value = ""
        self.collection_name_field.error_text = None
        self.create_collection_dialog.open = True
        self.page.update()

    def close_dialog(self, e):
        """Закрывает текущий диалог."""
        self.create_collection_dialog.open = False
        self.page.update()

    def do_create_collection(self, e):
        """Создаёт коллекцию и закрывает диалоговое окно."""
        collection_name = self.collection_name_field.value
        if collection_name is not None:
            self.core.create_collection_for_user(self.current_user_id, collection_name)
            self.create_collection_dialog.open = False
            self.load_collections()
            self.page.update()

    def on_upload_file(self, e):
        """Открыть диалог загрузки файла."""
        # заполняем выпадающий список коллекций
        self.upload_collection_dd.options = [
            ft.dropdown.Option(name) for name in
            self.core.list_user_collections(self.current_user_id)
        ]
        if self.upload_collection_dd.options:
            self.upload_collection_dd.value = self.upload_collection_dd.options[0].key

        self.files_label.value = "Файлы не выбраны"
        self.upload_dialog.open = True
        self.page.update()

    def close_upload_dialog(self, e):
        self.upload_dialog.open = False
        self.page.update()

    """
    def test_upload_files(self, e):
        self.button_all_upload.disabled = True
        self.page.update()
        self.core.upload_directory_for_user(self.current_user_id, 'test')
        self.button_all_upload.disabled = False
        self.page.update()
    """

    async def do_upload_files(self, e):
        self.button_filepick_cancel.disabled = True
        self.button_filepick_upload.disabled = True
        self.page.update()

        collection_name = self.upload_collection_dd.value
        await self.file_uploader.upload_and_process(collection_name)

        self.upload_dialog.open = False
        self.button_filepick_cancel.disabled = False
        self.button_filepick_upload.disabled = False
        self.page.update()
        return

    def _save_upload(self, file_bytes: bytes, suffix: str) -> Path:
        """Сохраняет байты во временный файл и возвращает путь."""
        fd, path = tempfile.mkstemp(suffix=suffix)
        with open(fd, "wb") as f:
            f.write(file_bytes)
        return Path(path)


    

    


    # Создание нового чата
    def create_new_chat(self, e):
        new_id = f"chat_{uuid.uuid4().hex[:8]}"
        if self.core.create_chat_for_user(self.current_user_id, new_id):
            self.current_chat_id = new_id
            self.load_user_chats()
            self._load_chat_history()

    # Отправка сообещния
    def send_message(self, e) -> None:
        message = self.message_line.value
        if not message:
            return
        
        # Блокируем поле и кнопку
        self.message_line.disabled = True
        self.message_button.disabled = True
        self.message_line.value = ""
        self.update()

        # Сохраняем информацию о текущем сообщении
        self.core.add_message_to_chat(
            self.current_user_id, self.current_chat_id, "user", message
        )
        text_user_message = self._display_message("user", message)
        self.message_list.controls.append(text_user_message)
        self.message_list.update()

        # 3. Определяем, какие коллекции выбраны
        selected_collections = [
            cb.label for cb in self.list_collections.controls
            if isinstance(cb, ft.Checkbox) and cb.value
        ]

        # Генерируем ответное сообщение
        answer = self.core.generate_response(
            user_id=self.current_user_id,
            chat_id=self.current_chat_id,
            query=message,
            collection_names=selected_collections
        )
        self.core.add_message_to_chat(
            self.current_user_id, self.current_chat_id, "assistant", answer
        )
        text_assistant_message = self._display_message("assistant", answer)
        self.message_list.controls.append(text_assistant_message)

        # Разблокировываем
        self.message_line.disabled = False
        self.message_button.disabled = False
        self.update()


    # Дизайн

    def _display_message(self, role: str, text: str):
        """
        Красиво отображает сообщение в текущем чате
        """
        is_user = role == "user"
        bg_color = AppColors["user_message"] if is_user else AppColors["assistant_message"]
        alignment = ft.MainAxisAlignment.END if is_user else ft.MainAxisAlignment.START

        def width():
            max_width = min(700, int(self.dialog_area.width * 0.85)) if self.dialog_area.width else 700
            min_width = len(text) * 10
            return min(max_width, min_width)

        text_message = ft.Text(
            text,
            selectable=True,
            width=width(),
            no_wrap=False,
            overflow=ft.TextOverflow.VISIBLE,
            size=14
        )

        message_bubble = ft.Container(
            content=text_message,
            bgcolor=bg_color,
            padding=ft.padding.symmetric(horizontal=14, vertical=10),
            border_radius=ft.border_radius.only(
                top_left=18 if is_user else 4,
                top_right=4 if is_user else 18,
                bottom_left=18,
                bottom_right=18
            ),
            margin=ft.margin.only(
                left=40 if is_user else 10,
                right=10 if is_user else 40,
                top=4,
                bottom=4
            ),
            width=None,
            alignment=ft.Alignment.CENTER_LEFT if not is_user else ft.Alignment.CENTER_RIGHT,
            expand=False
        )
        return ft.Row([message_bubble], alignment=alignment)
        
    def _display_chat_list_item(self, chat_id):
        return ft.TextButton(
                    content=chat_id, 
                    height=40,
                    on_click=self.on_chat_click,
                    style=self._btn_style(False)
                )
    
    def _btn_style(self, selected: bool = False):
        """Возвращает стиль кнопки-чата."""
        return ft.ButtonStyle(
            alignment=ft.Alignment(-1, 0),
            padding=ft.padding.all(10),
            shape=ft.RoundedRectangleBorder(radius=5),
            bgcolor={
                "hovered": AppColors["hovered_chat"],
                "": AppColors["hovered_chat"] if selected else AppColors["chat_list"]
            }
        )
    

