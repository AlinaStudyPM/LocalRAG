# src/SQLiteAdapter.py
import sqlite3

from Config import Config

class SQLiteAdapter:
    def __init__(self, config: Config):

        self.connection = sqlite3.connect(self.config.SQLITE_DB_PATH, check_same_thread=False)
        self.cursor = self.connection.cursor()

        self.create_table_users = """
        CREATE TABLE IF NOT EXISTS users (
	    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
	    username TEXT UNIQUE NOT NULL,
	    password_hash TEXT NOT NULL,
	    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """

        self.create_table_collections = """
        CREATE TABLE IF NOT EXISTS collections (
	    user_id INTEGER NOT NULL,
	    name TEXT NOT NULL,
	    embedding_model TEXT DEFAULT 'intfloat/multilingual-e5-small',
	    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
	    PRIMARY KEY (user_id, name),
	    FOREIGN KEY(user_id) REFERENCES users(user_id) ON DELETE CASCADE
        )
        CREATE INDEX IF NOT EXISTS idx_collections ON collections(user_id);
        """

        self.create_table_chats = """
        CREATE TABLE IF NOT EXISTS chats (
	    chat_id TEXT PRIMARY KEY,
	    user_id INTEGER NOT NULL,
	    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
	    FOREIGN KEY(user_id) REFERENCES users(user_id)
        )
        CREATE INDEX IF NOT EXISTS idx_chats ON chats(user_id);
        """

        self.create_table_messages = """
        CREATE TABLE IF NOT EXISTS messages (
	    id INTEGER PRIMARY KEY AUTOINCREMENT,
	    user_id INTEGER NOT NULL,
	    chat_id TEXT NOT NULL,
	    role TEXT NOT NULL,
	    content TEXT NOT NULL,
	    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
	    FOREIGN KEY(user_id) REFERENCES users(user_id) ON DELETE CASCADE,
	    FOREIGN KEY(chat_id) REFERENCES chats(chat_id) ON DELETE CASCADE
        )
        CREATE INDEX IF NOT EXISTS idx_messages ON messages(chat_id, timestamp);
        """

    def add_user(self, username: str, password_hash: str):
        self.cursor.execute(
        "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                (username, password_hash)
        )
        self.connection.commit()

    def get_user_by_id(self, user_id: str):
        self.cursor(
            "SELECT username, password_hash FROM users WHERE user_id = ?",
            (user_id,)
        )
        row = self.cursor.fetchone()
        username, stored_hash = row
        return (username, stored_hash)


    def get_user_by_name(self, username: str):
        self.cursor.execute(
            "SELECT user_id, password_hash FROM users WHERE username = ?",
            (username,)
        )
        row = self.cursor.fetchone()
        user_id, stored_hash = row
        return (user_id, stored_hash)
    
    def get_users(self):
        self.cursor.execute("SELECT user_id, username FROM users")
        return self.cursor.fetchall()
    
    # ================== collections =====================

    def create_collection(self, user_id: int, name: str) -> None:
        self.connection.execute(
                "INSERT INTO collections (user_id, name) VALUES (?, ?)",
                (user_id, name)
        )
        self.connection.commit()
    
    def get_collections(self, user_id: str):
        self.cursor.execute(
            "SELECT name FROM collections WHERE user_id = ?",
            (user_id,)
        )
        return self.cursor.fetchall()

    def close(self):
        """
        Закрывает соединение с SQLite.
        """
        self.conn_sqlite3.close()