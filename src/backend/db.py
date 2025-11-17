import sqlite3
from typing import Optional, Dict, Any, List
from passlib.hash import bcrypt

class DB:
    def __init__(self, path: str = "Users.db"):
        self.path = path
        self._init()

    def _connect(self):
        conn = sqlite3.connect(self.path)
        # makes them into dictionarys
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _init(self):
        conn = self._connect()
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL,
            topic TEXT NOT NULL CHECK (topic IN ('CS','MED')),
            query TEXT NOT NULL,
            FOREIGN KEY(email) REFERENCES users(email) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
        CREATE INDEX IF NOT EXISTS idx_interactions_email_id ON interactions(email, id DESC);
        """)
        conn.commit()
        conn.close()

    # -------- Users --------
    def user_exists(self, email: str) -> bool:
        conn = self._connect()
        row = conn.execute("SELECT 1 FROM users WHERE email = ?", (email,)).fetchone()
        conn.close()
        return row is not None

    def save_user(self, d: Dict[str, Any]) -> int:
        username = d.get("username") or d.get("name")
        email = d.get("email")
        pw = d.get("password")
        if not username or not email or not pw:
            raise ValueError("username/name, email, password required")
        hashed = bcrypt.hash(pw)
        conn = self._connect()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
            (username, email, hashed),
        )
        conn.commit()
        uid = cur.lastrowid
        conn.close()
        return uid

    def get_user_by_id(self, uid: int) -> Optional[Dict[str, Any]]:
        conn = self._connect()
        row = conn.execute("SELECT id, username, email FROM users WHERE id = ?", (uid,)).fetchone()
        conn.close()
        return dict(row) if row else None

    def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        conn = self._connect()
        row = conn.execute("SELECT id, username, email FROM users WHERE email = ?", (email,)).fetchone()
        conn.close()
        return dict(row) if row else None

    def delete_user_by_email(self, email: str) -> int:
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("DELETE FROM users WHERE email = ?", (email,))
        conn.commit()
        count = cur.rowcount
        conn.close()
        return count

    # -------- Interactions --------
    def log_interaction(self, d: Dict[str, Any]) -> int:
        # use get so we dont crash
        email = d.get("email")
        query = (d.get("query") or "").strip()
        topic = (d.get("topic") or "").upper()
        if not email or not query or topic not in ("CS", "MED"):
            raise ValueError("email, query, topic('CS'|'MED') required")

        conn = self._connect()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO interactions (email, topic, query) VALUES (?, ?, ?)",
            (email, topic, query),
        )
        conn.commit()
        iid = cur.lastrowid
        conn.close()
        print(iid)
        return iid

    def recent_interactions(self, email: str, limit: int = 10) -> List[Dict[str, Any]]:
        conn = self._connect()
        rows = conn.execute(
            "SELECT id, email, topic, query FROM interactions WHERE email = ? ORDER BY id DESC LIMIT ?",
            (email, limit),
        ).fetchall()
        conn.close()
        a =  [dict(r) for r in rows]
        print(a)
        return a
