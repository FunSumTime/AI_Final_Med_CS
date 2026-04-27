import sqlite3
from typing import Optional, Dict, Any, List
from passlib.hash import bcrypt


def dict_factory(cursor, row):
    fields = []
    for column in cursor.description:
        fields.append(column[0])

    result_dict = {}
    for i in range(len(fields)):
        result_dict[fields[i]] = row[i]

    return result_dict


class DB:
    def __init__(self, dbfilename: str = "Users.db"):
        self.dbfilename = dbfilename
        # Single connection per DB instance
        self.connection = sqlite3.connect(dbfilename)
        self.connection.execute("PRAGMA foreign_keys = ON;")
        self.cursor = self.connection.cursor()
        self._init()

    def _init(self):
        self.cursor.executescript(
            """
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

            CREATE INDEX IF NOT EXISTS idx_users_email
                ON users(email);

            CREATE INDEX IF NOT EXISTS idx_interactions_email_id
                ON interactions(email, id DESC);
            """
        )
        self.cursor.executescript(
            """
            CREATE TABLE IF NOT EXISTS quizzes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL,
                topic TEXT NOT NULL,       -- 'CS' or 'MED'
                quiz_json TEXT NOT NULL,   -- JSON with questions & correct answers
                user_answers_json TEXT,    -- JSON of user's answers (filled when done)
                score REAL,                -- score when completed, 0–100 or 0–1
                status TEXT NOT NULL       -- 'in_progress' or 'completed'
            );
            """
        )
        self.connection.commit()

    # ---------- Users ----------

    def user_exists(self, email: str) -> bool:
        self.cursor.execute("SELECT 1 FROM users WHERE email = ?;", [email])
        row = self.cursor.fetchone()
        return row is not None

    def save_user(self, d: Dict[str, Any]) -> int:
        username = d.get("username") or d.get("name")
        email = d.get("email")
        pw = d.get("password")

        if not username or not email or not pw:
            raise ValueError("username/name, email, password required")

        hashed = bcrypt.hash(pw)
        data = [username, email, hashed]

        self.cursor.execute(
            "INSERT INTO users (username, email, password) VALUES (?,?,?);",
            data,
        )
        self.connection.commit()
        uid = self.cursor.lastrowid
        return uid

    def get_user_by_id(self, uid: int) -> Optional[Dict[str, Any]]:
        self.cursor.execute(
            "SELECT id, username, email FROM users WHERE id = ?;",
            [uid],
        )
        row = self.cursor.fetchone()
        if not row:
            return None
        return dict_factory(self.cursor, row)

    def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        self.cursor.execute(
            "SELECT id, username, email FROM users WHERE email = ?;",
            [email],
        )
        row = self.cursor.fetchone()
        if not row:
            return None
        return dict_factory(self.cursor, row)

    def delete_user_by_email(self, email: str) -> int:
        self.cursor.execute("DELETE FROM users WHERE email = ?;", [email])
        self.connection.commit()
        return self.cursor.rowcount

    # ---------- Interactions ----------

    def log_interaction(self, d: Dict[str, Any]) -> int:
        email = d.get("email")
        query = (d.get("query") or "").strip()
        topic = (d.get("topic") or "").upper()

        if not email or not query or topic not in ("CS", "MED"):
            raise ValueError("email, query, topic('CS'|'MED') required")

        self.cursor.execute(
            "INSERT INTO interactions (email, topic, query) VALUES (?,?,?);",
            [email, topic, query],
        )
        self.connection.commit()
        iid = self.cursor.lastrowid
        print(iid)
        return iid

    def recent_interactions(self, email: str, limit: int = 10) -> List[Dict[str, Any]]:
        self.cursor.execute(
            """
            SELECT id, email, topic, query
            FROM interactions
            WHERE email = ?
            ORDER BY id DESC
            LIMIT ?;
            """,
            [email, limit],
        )
        rows = self.cursor.fetchall()
        result = [dict_factory(self.cursor, r) for r in rows]
        print(result)
        return result

    # ---------- Login / verification ----------

    def user_verify(self, d: Dict[str, Any]) -> bool:
        email = d.get("email")
        pw = d.get("password")

        if not email or not pw:
            return False

        self.cursor.execute(
            "SELECT password FROM users WHERE email = ?;",
            [email],
        )
        row = self.cursor.fetchone()
        if not row:
            print("No user found for", email)
            return False

        row_dict = dict_factory(self.cursor, row)
        hashed_pw = row_dict["password"]

        try:
            ok = bcrypt.verify(pw, hashed_pw)
        except Exception as e:
            print("Error verifying password:", e)
            return False

        print("verify result for", email, "=", ok)
        return ok

    # ---------- Quizzes ----------

    def save_quiz(self, email: str, topic: str, quiz_json: str) -> int:
        """
        Insert a new quiz for this user.
        Status starts as 'in_progress'.
        """
        cur = self.connection.cursor()
        cur.execute(
            """
            INSERT INTO quizzes (email, topic, quiz_json, status)
            VALUES (?, ?, ?, 'in_progress')
            """,
            (email, topic, quiz_json),
        )
        self.connection.commit()
        quiz_id = cur.lastrowid
        return quiz_id

    def mark_quiz_completed(self, quiz_id: int, user_answers_json: str, score: float) -> None:
        """
        Mark a quiz as completed, storing the user's answers and score.
        """
        cur = self.connection.cursor()
        cur.execute(
            """
            UPDATE quizzes
               SET user_answers_json = ?,
                   score = ?,
                   status = 'completed'
             WHERE id = ?
            """,
            (user_answers_json, score, quiz_id),
        )
        self.connection.commit()

    def get_completed_quizzes_by_email(self, email: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Return the most recent completed quizzes for a user as list[dict].
        """
        # Use Row so we can easily convert to dict
        self.connection.row_factory = sqlite3.Row
        cur = self.connection.cursor()
        cur.execute(
            """
            SELECT id, topic, quiz_json, user_answers_json, score
              FROM quizzes
             WHERE email = ?
               AND status = 'completed'
             ORDER BY id DESC
             LIMIT ?
            """,
            (email, limit),
        )
        rows = cur.fetchall()
        return [dict(r) for r in rows]
    
    def get_quizzes_by_email(self, email: str, limit: int = 10):
        cur = self.connection.cursor()
        cur.execute(
            """
            SELECT id, topic, quiz_json, user_answers_json, score, status
              FROM quizzes
             WHERE email = ?
             ORDER BY id DESC
             LIMIT ?;
            """,
            (email, limit),
        )
        rows = cur.fetchall()

        # convert rows to list[dict]
        cols = [col[0] for col in cur.description]
        result = []
        for row in rows:
            d = {}
            for i, col in enumerate(cols):
                d[col] = row[i]
            result.append(d)

        return result


    # ---------- housekeeping ----------

    def close(self):
        self.connection.close()
