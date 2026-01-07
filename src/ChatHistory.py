from typing import List, Dict
from datetime import datetime

import sqlite3

from Config import Config

class ChatHistory:
    """
    Хранит историю сообщений одного чата.
    Загружает и сохраняет сообщения в SQLite.
    """
    def __init__(self, chat_id: str, user_id: int, config: Config, chroma_client=None):
        """
        Инициализирует историю чата и загружает сообщения из БД.
        """
        self.chat_id = chat_id
        self.user_id = user_id
        self.config = config
        self.messages: List[Dict[str, str]] = []

        # --- SQLite --- 
        self.conn_sqlite3 = sqlite3.connect(config.SQLITE_DB_PATH, check_same_thread=False)
        self.conn_sqlite3.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            chat_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )                  
        """)
        self.conn_sqlite3.execute("""
            CREATE INDEX IF NOT EXISTS idx_msg_chat ON messages(user_id, chat_id)
        """)
        self.load_history() 


    def add_message(self, role: str, content: str):
        """
        Добавляет сообщение в чат и сохраняет его в базу данных.
        """
        timestamp = datetime.now().isoformat()
        self.messages.append({"role": role, "content": content, "timestamp": timestamp})
        
        cursor = self.conn_sqlite3.cursor()
        cursor.execute(
            "INSERT INTO messages (user_id, chat_id, role, content) VALUES (?, ?, ?, ?)",
            (self.user_id, self.chat_id, role, content)
        )
        self.conn_sqlite3.commit()

    def load_history(self) -> None:
        """
        Загружает историю сообщений из базы данных в память.
        Сортирует по возрастанию времени (от старых к новым).
        """
        cursor = self.conn_sqlite3.execute("""
            SELECT id, role, content, timestamp FROM messages 
            WHERE user_id=? AND chat_id=? ORDER BY timestamp ASC
        """, (self.user_id, self.chat_id))
        rows = cursor.fetchall()

        self.messages = [
            {"role": role, "content": content, "timestamp": ts}
            for _, role, content, ts in rows
        ]

    def get_history(self):
        """
        Возвращает копию истории сообщений.
        """
        return self.messages

    def close(self):
        """
        Закрывает соединение с базой данных.
        """
        if self.conn_sqlite3:
            self.conn_sqlite3.close()