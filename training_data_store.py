import sqlite3
from typing import List, Optional, Tuple

DB_PATH = "training_data_store.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS training_data (
            uuid TEXT PRIMARY KEY,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            state TEXT NOT NULL,
            uuid_of_used_article TEXT,
            master_model TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def create_training_data(uuid: str, question: str, answer: str, state: str, uuid_of_used_article: Optional[str], master_model: str) -> bool:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Check if training data with same question already exists
    c.execute("SELECT uuid FROM training_data WHERE question = ?", (question,))
    existing = c.fetchone()
    
    if existing:
        conn.close()
        return False  # Training data with same question already exists
    
    # Proceed with insertion if question is unique
    c.execute("""
        INSERT INTO training_data (uuid, question, answer, state, uuid_of_used_article, master_model)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (uuid, question, answer, state, uuid_of_used_article, master_model))
    conn.commit()
    conn.close()
    return True  # Successfully created

def get_training_data_by_uuid(uuid: str) -> Optional[Tuple]:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM training_data WHERE uuid = ?", (uuid,))
    result = c.fetchone()
    conn.close()
    return result

def get_training_data_by_state(state: str) -> List[Tuple]:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM training_data WHERE state = ?", (state,))
    results = c.fetchall()
    conn.close()
    return results

def get_training_data_by_article_uuid(uuid_of_used_article: str) -> List[Tuple]:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM training_data WHERE uuid_of_used_article = ?", (uuid_of_used_article,))
    results = c.fetchall()
    conn.close()
    return results

def get_training_data_by_master_model(master_model: str) -> List[Tuple]:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM training_data WHERE master_model = ?", (master_model,))
    results = c.fetchall()
    conn.close()
    return results

def update_training_data(uuid: str, question: Optional[str]=None, answer: Optional[str]=None, state: Optional[str]=None, uuid_of_used_article: Optional[str]=None, master_model: Optional[str]=None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    fields = []
    values = []
    if question:
        fields.append("question = ?")
        values.append(question)
    if answer:
        fields.append("answer = ?")
        values.append(answer)
    if state:
        fields.append("state = ?")
        values.append(state)
    if uuid_of_used_article:
        fields.append("uuid_of_used_article = ?")
        values.append(uuid_of_used_article)
    if master_model:
        fields.append("master_model = ?")
        values.append(master_model)
    values.append(uuid)
    if fields:
        c.execute(f"UPDATE training_data SET {', '.join(fields)} WHERE uuid = ?", values)
        conn.commit()
    conn.close()

def delete_training_data(uuid: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM training_data WHERE uuid = ?", (uuid,))
    conn.commit()
    conn.close()

def fuzzy_search(query: str, n: int = 10) -> List[Tuple]:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    like_query = f"%{query}%"
    c.execute("""
        SELECT * FROM training_data
        WHERE question LIKE ? OR answer LIKE ? OR state LIKE ? OR master_model LIKE ?
        LIMIT ?
    """, (like_query, like_query, like_query, like_query, n))
    results = c.fetchall()
    conn.close()
    return results

# Initialize DB on import
init_db()