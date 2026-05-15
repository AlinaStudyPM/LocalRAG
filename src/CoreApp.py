# src/CoreApp.py
import os
from typing import List, Optional, Dict

import chromadb

from src.Config import Config
from src.ChromaAdapter import ChromaAdapter
from src.DocumentProcessor import DocumentProcessor
from src.UserManager import UserManager, User
from src.ChatAgent import ChatAgent

class CoreApp:
    """
    Ядро приложения.
    """
    def __init__(self):
        self.config = Config()

        self.chroma_adapter = ChromaAdapter(self.config)
        self.doc_processor = DocumentProcessor(self.config)
        self.user_manager = UserManager(self.config)
        self.chat_agent = ChatAgent(self.config, self.chroma_adapter)

    #  === USER MANAGEMENT ===
    def register_user(self, username: str, password: str) -> Optional[User]:
        """Регистрирует нового пользователя."""
        return self.user_manager.register(username, password)

    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Аутентифицирует пользователя и загружает его чаты."""
        user = self.user_manager.authenticate(username, password)
        if user is None:
            return None
        user.load_chats()
        return user
    
    def logout_user(self, user_id: int) -> None:
        """Завершает сессию пользователя."""
        self.user_manager.logout_user(user_id)

    # === CHAT MANAGEMENT ===
    def list_user_chats(self, user_id: int) -> list[str]:
        """Возвращает список chat_id для пользователя"""
        user = self.user_manager.get_user_by_id(user_id)
        return user.list_chats()

    def create_chat_for_user(self, user_id: int, chat_id: str) -> bool:
        """Создаёт чат. Возвращает True при успехе."""
        user = self.user_manager.get_user_by_id(user_id)
        return user.create_chat(chat_id)

    def get_chat_history(self, user_id: int, chat_id: str) -> List[dict]:
        """Возвращает историю чата в формате[{"role": "...", "content": "..."}, ...]"""
        user = self.user_manager.get_user_by_id(user_id)
        chat_history = user.get_chat(chat_id)
        return chat_history.get_history()

    def add_message_to_chat(self, user_id: int, chat_id: str, role: str, content: str) -> bool:
        """Добавляет сообщение в чат"""
        user = self.user_manager.get_user_by_id(user_id)
        chat_history = user.get_chat(chat_id)
        return chat_history.add_message(role, content)
    
    # === COLLECTIONS ===
    def get_supported_extensions(self) -> List[str]:
        """Возвращает список разрешённых расширений файлов."""
        return self.doc_processor.get_supported_extensions()

    def list_user_collections(self, user_id: int) -> List[str]:
        """Возвращает список имён коллекций пользователя"""
        user = self.user_manager.get_user_by_id(user_id)
        return user.list_collections()
    
    def list_user_files(self, user_id: int) -> Dict[str, List[str]]:
        """Возвращает словарь, в котором по названию коллекции лежит список файлов в ней."""
        user = self.user_manager.get_user_by_id(user_id)
        collections = user.list_collections()
        result = {}
        for coll in collections:
            coll_id = user.get_collection_id(coll)
            files = self.chroma_adapter.list_files(coll_id) # List[str]
            result[coll] = files
        return result

    def create_collection_for_user(self, user_id: int, collection_name: str) -> None:
        """Создаёт новую коллекцию"""
        user = self.user_manager.get_user_by_id(user_id)
        user.create_collection(collection_name)

    def get_collection_id(self, user_id: int, collection_name: str) -> str:
        """Возвращает UUID коллекции по её отображаемому имени."""
        user = self.user_manager.get_user_by_id(user_id)
        return user.get_collection_id(collection_name)

    def upload_file_for_user(self, user_id: int, collection_name: str, file_path: str, quick: bool = True) -> None:
        chunks = self.doc_processor.process_pdf(file_path, quick)
        self.chroma_adapter.add_documents(
            collection_name=collection_name,
            file_name=file_path,
            texts=chunks
        )
        # user.add_file_to_collection(collection_name, file_path.name) - не реализовано

    # Для промежуточной версии
    """
    def upload_directory_for_user(self, user_id: int, collection_name: str, quick: bool = False) -> None:
        user = self.user_manager.get_user_by_id(user_id)
        self._client = chromadb.PersistentClient(path=self.config.CHROMA_DB_DIR)

        DIRECTORIES_NAMES = {
            "math": r"/app/files/math",
            "it": r"/app/files/it",
            # "ml": r"/app/files/ML"
        }

        print("Начинаем загрузку тестовых файлов...")

        for name in DIRECTORIES_NAMES.keys():
            dir_path = DIRECTORIES_NAMES[name]
            if not os.path.exists(dir_path):
                raise FileNotFoundError(f"Папка {dir_path} не найдена.")
            self.create_collection_for_user(user_id, name)
            # print(f"Коллекции: {user.list_collections()}")
            collection_id = user.get_collection_id(name)

            for file_name in os.listdir(dir_path):
                if self.chroma_adapter.is_in_collection(collection_id, file_name):
                    print(f"Файл {file_name} уже есть в коллекции {collection_id}.")
                    continue
                file_path = os.path.join(dir_path, file_name)
                if file_path.endswith(".pdf"):
                    chunks = self.doc_processor.process_pdf(file_path, quick)
                    self.chroma_adapter.add_documents(
                        collection_name=collection_id,
                        file_name=file_name,
                        texts=chunks
                    )
                print(f"Файл {file_name} загружен успешно в коллекцию {collection_id}.")
        print("Загрузка файлов завершена!")
    """
        
    
    # === MODELS ===
    def list_available_models(self) -> list[str]:
        """Список доступных моделей"""
        return self.chat_agent.list_models()

    def get_current_model(self) -> str:
        """Возвращает текущую используемую модель."""
        return self.chat_agent.model

    def set_current_model(self, model_name: str) -> None:
        """Устанавливает модель, если она доступна."""
        if model_name in self.list_available_models():
            self.chat_agent.model = model_name
        else:
            raise ValueError(f"Model '{model_name}' is not available")
                
    # === CHAT GENERATION ===
    def generate_response(
        self,
        user_id: int,
        chat_id: str,
        query: str,
        collection_names: List[str]
    ) -> str:
        """Генерирует ответ на запрос пользователя с учётом контекста и выбранных коллекций."""
        user = self.user_manager.get_user_by_id(user_id)
        history = self.get_chat_history(user_id, chat_id)

        collection_ids = [
            user.get_collection_id(name) for name in collection_names
        ]

        return self.chat_agent.answer_question(
            question=query,
            chat_history=history,
            collection_names=collection_ids
        )

