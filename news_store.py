import sqlite3
from typing import List, Optional, Tuple

DB_PATH = "news_store.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS articles (
            uid TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            date TEXT NOT NULL,
            state TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def create_article(uid: str, title: str, content: str, date: str, state: str) -> bool:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Check if article with same title already exists
    c.execute("SELECT uid FROM articles WHERE title = ?", (title,))
    existing = c.fetchone()
    
    if existing:
        conn.close()
        return False  # Article with same title already exists
    
    # Proceed with insertion if title is unique
    c.execute("""
        INSERT INTO articles (uid, title, content, date, state)
        VALUES (?, ?, ?, ?, ?)
    """, (uid, title, content, date, state))
    conn.commit()
    conn.close()
    return True  # Successfully created

def get_article_by_uid(uid: str) -> Optional[Tuple]:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM articles WHERE uid = ?", (uid,))
    result = c.fetchone()
    conn.close()
    return result

def get_articles_by_state(state: str) -> List[Tuple]:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM articles WHERE state = ?", (state,))
    results = c.fetchall()
    conn.close()
    return results

def get_articles_by_date(date: str) -> List[Tuple]:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM articles WHERE date = ?", (date,))
    results = c.fetchall()
    conn.close()
    return results

def get_articles_in_date_range(start_date: str, end_date: str) -> List[Tuple]:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM articles WHERE date BETWEEN ? AND ?", (start_date, end_date))
    results = c.fetchall()
    conn.close()
    return results

def update_article(uid: str, title: Optional[str]=None, content: Optional[str]=None, date: Optional[str]=None, state: Optional[str]=None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    fields = []
    values = []
    if title:
        fields.append("title = ?")
        values.append(title)
    if content:
        fields.append("content = ?")
        values.append(content)
    if date:
        fields.append("date = ?")
        values.append(date)
    if state:
        fields.append("state = ?")
        values.append(state)
    values.append(uid)
    if fields:
        c.execute(f"UPDATE articles SET {', '.join(fields)} WHERE uid = ?", values)
        conn.commit()
    conn.close()

def delete_article(uid: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM articles WHERE uid = ?", (uid,))
    conn.commit()
    conn.close()

def fuzzy_search(query: str, n: int = 10) -> List[Tuple]:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    like_query = f"%{query}%"
    c.execute("""
        SELECT * FROM articles
        WHERE title LIKE ? OR content LIKE ? OR state LIKE ?
        LIMIT ?
    """, (like_query, like_query, like_query, n))
    results = c.fetchall()
    conn.close()
    return results

# Initialize DB on import
init_db()