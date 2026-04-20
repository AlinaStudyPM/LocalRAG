
# src/UserManager.py
from typing import Dict, List, Optional

import sqlite3
import bcrypt
import uuid

from src.Config import Config
from src.ChatHistory import ChatHistory
from src.ChromaAdapter import ChromaAdapter

# TODO: зачем пользователю хранить конфигурацию всей системы?
# TODO: убрать связь с chroma
class User:
    """
    Хранит информацию о пользователе системы, управляет его чатами и коллекциями документов.
    """
    def __init__(self, user_id: int, username: str, config, chroma_adapter=None):
        """
        Инициализирует пользователя и подключается к SQLite.

        Создаёт таблицы `chats` и `user_collections`, если они не существуют.
        Загружает список коллекций из БД.
        """
        self.user_id = user_id
        self.username = username
        self.config = config
        self.chats: Dict[str, ChatHistory] = {}  # chat_id -> ChatHistory
        self.collections: Dict[str, str] = {}

        self.chroma_adapter = ChromaAdapter(self.config)

        self.conn_sqlite3 = sqlite3.connect(self.config.SQLITE_DB_PATH, check_same_thread=False)
        cursor = self.conn_sqlite3.cursor()
        # Создание таблицы чатов
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chats (
                chat_id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(user_id)
            )
        """)
        self.load_chats()
        # Создание таблицы коллекций
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_collections (
                id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                collection_name TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(user_id) ON DELETE CASCADE
            )
        """)
        self.conn_sqlite3.commit()
        self._load_collections()

    def _load_collections(self) -> None:
        """
        Загружает список названий коллекций пользователя из таблицы `user_collections` в SQLite.
        """
        cursor = self.conn_sqlite3.execute(
            "SELECT id, collection_name FROM user_collections WHERE user_id = ?",
            (self.user_id,)
        )
        self.collections = {row[0]: row[1] for row in cursor.fetchall()}

    def create_collection(self, collection_name: str) -> None:
        """
        Создаёт коллекцию в Chroma и регистрирует её в SQLite.
        """
        if not collection_name.strip():
            raise ValueError("Название коллекции не может быть пустым")
        if collection_name in self.collections.values():
            raise ValueError(f"Коллекция '{collection_name}' уже существует")
        collection_id = uuid.uuid4().hex
        self.chroma_adapter.create_collection(collection_name=collection_id)
        self.conn_sqlite3.execute(
            "INSERT INTO user_collections (id, user_id, collection_name) VALUES (?, ?, ?)",
            (collection_id, self.user_id, collection_name)
        )
        self.conn_sqlite3.commit()
        self.collections[collection_id] = collection_name
         
    def get_collection_id(self, collection_name: str) -> str:
        for uuid, name in self.collections.items():
            if name == collection_name:
                return uuid
        raise ValueError(f"Коллекция '{collection_name}' не принадлежит пользователю {self.user_id}")
    
    def list_collections(self) -> List[str]:
        """Возвращает копию списка имён коллекций пользователя."""
        return list(self.collections.values())  
    
    def load_chats(self) -> None:
        """
        Загружает все чаты пользователя из таблицы `chats` и создаёт объекты `ChatHistory`.
        """
        cur = self.conn_sqlite3.execute(
            "SELECT chat_id FROM chats WHERE user_id=? ORDER BY created_at",
            (self.user_id,)
        )
        for (chat_id,) in cur.fetchall():
            if chat_id not in self.chats:
                self.chats[chat_id] = ChatHistory(
                    chat_id=chat_id,
                    user_id=self.user_id,
                    config=self.config
                )

    def create_chat(self, chat_id: str) -> ChatHistory:
        """
        Создаёт новый чат для пользователя.
        Добавляет запись в таблицу `chats` и возвращает объект `ChatHistory`.
        """
        self.conn_sqlite3.execute(
            "INSERT OR IGNORE INTO chats(chat_id, user_id) VALUES (?,?)",
            (chat_id, self.user_id)
        )
        self.conn_sqlite3.commit()
        chat_history = ChatHistory(
            chat_id=chat_id,
            user_id=self.user_id,
            config=self.config,
            # chroma_client=self.chroma_client
        )
        self.chats[chat_id] = chat_history
        return chat_history

    def get_chat(self, chat_id: str) -> ChatHistory:
        """
        Возвращает объект чата по его идентификатору.
        """
        return self.chats.get(chat_id)

    def list_chats(self) -> List[str]:
        """
        Возвращает список идентификаторов чатов пользователя.
        """
        return list(self.chats.keys())

    def close(self):
        """
        Закрывает все чаты и соединение с SQLite.
        """
        for chat in self.chats.values():
            chat.close()
        self.chats.clear()
        if self.conn_sqlite3:
            self.conn_sqlite3.close()

    

class UserManager:
    """
    Управляет регистрацией, аутентификацией и кэшированием пользователей.
    """
    def __init__(self, config: Config):
        """
        Инициализирует менеджер пользователей и создаёт таблицу `users`, если необходимо.
        """
        self.config = config
        self.conn_sqlite3 = sqlite3.connect(self.config.SQLITE_DB_PATH, check_same_thread=False)
        self.cursor_sqlite3 = self.conn_sqlite3.cursor()

        # Словарь пользователей: user_id -> User
        self.users: Dict[int, User] = {}

        # Создание таблицы пользователей
        self.cursor_sqlite3.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self.conn_sqlite3.commit()

    def register(self, username: str, password: str) -> User | None:
        """
        Создание нового пользователя, возвращает user_id.
        """
        password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        try:
            self.cursor_sqlite3.execute(
                "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                (username, password_hash)
            )
            self.conn_sqlite3.commit()
        except sqlite3.IntegrityError:
            return None
        
        user_id = self.cursor_sqlite3.lastrowid
        user = User(
            user_id=user_id, 
            username=username, 
            config=self.config
        )
        self.users[user_id] = user
        return user
    
    def authenticate(self, username: str, password: str) -> User | None:
        """
        Аутентифицирует пользователя по имени и паролю.
        """
        self.cursor_sqlite3.execute(
            "SELECT user_id, password_hash FROM users WHERE username = ?",
            (username,)
        )
        row = self.cursor_sqlite3.fetchone()

        if row is None:
            return None
        
        user_id, stored_hash = row
        if bcrypt.checkpw(password.encode(), stored_hash.encode()):
            return self.get_user_by_id(user_id)
        return None
    
    def logout_user(self, user_id: int):
        """
        Удаляет пользователя из кэша и закрывает его ресурсы.
        """
        if user_id in self.users:
            user = self.users.pop(user_id)
            user.close()
    
    def get_user_by_id(self, user_id: int) -> Optional[User]:
        """
        Возвращает пользователя по ID (из кэша или БД).
        """
        if user_id in self.users:
            return self.users[user_id]

        # Ищем в базе
        self.cursor_sqlite3.execute(
            "SELECT username FROM users WHERE user_id = ?",
            (user_id,)
        )
        row = self.cursor_sqlite3.fetchone()

        if row is None:
            return None

        # Кэшируем
        username = row[0]
        user = User(
            user_id=user_id,
            username=username,
            config=self.config
        )
        self.users[user_id] = user
        return user

    
    def get_user_by_name(self, username: str) -> Optional[User]:
        """
        Возвращает пользователя по имени.
        """
        self.cursor_sqlite3.execute("SELECT user_id FROM users WHERE username = ?", (username,))
        res = self.cursor_sqlite3.fetchone()

        if res is None:
            return None
        
        user_id = res[0]

        # Если уже есть в кэше — возвращаем
        if user_id in self.users:
            return self.users[user_id]

        # Создаём новый User из БД
        user = User(
            user_id=user_id,
            username=username,
            config=self.config,
            # chroma_client=self.chroma_client
        )
        self.users[user_id] = user
        return user
    
    def list_users(self) -> List[Dict]:
        """
        Возвращает список всех пользователей.
        """
        self.cursor_sqlite3.execute("SELECT user_id, username, created_at FROM users")
        rows = self.cursor_sqlite3.fetchall()
        return [{"user_id": r[0], "username": r[1], "created_at": r[2]} for r in rows]

    """
    def list_chats_for_user(self, user_id: int) -> List[str]:
        user = self.get_user_by_id(user_id)
        return user.list_chats() if user else []
    """
    
    def close(self):
        """
        Закрывает соединение с SQLite.
        """
        self.conn_sqlite3.close()
