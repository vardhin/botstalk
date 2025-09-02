import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Optional
import uuid

class SQLiteArticleDatabase:
    """
    SQLite database for article metadata and training tracking
    """
    
    def __init__(self, db_path: str = "news_articles.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.setup_schema()
        print(f"SQLite database initialized: {db_path}")
    
    def setup_schema(self):
        """Setup database schema"""
        cursor = self.conn.cursor()
        
        # Articles table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS articles (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                summary TEXT NOT NULL,
                content TEXT,
                region TEXT NOT NULL,
                source TEXT NOT NULL,
                link TEXT,
                published TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                word_count INTEGER,
                quality_score REAL,
                processed_for_embedding BOOLEAN DEFAULT FALSE,
                processed_for_training BOOLEAN DEFAULT FALSE
            )
        ''')
        
        # Training batches table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_batches (
                batch_id TEXT PRIMARY KEY,
                region TEXT NOT NULL,
                article_ids TEXT NOT NULL,  -- JSON array
                training_examples_count INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'pending',  -- pending, training, completed, failed
                epochs INTEGER DEFAULT 0,
                model_path TEXT
            )
        ''')
        
        # Model versions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_versions (
                version_id TEXT PRIMARY KEY,
                region TEXT,
                model_name TEXT,
                training_batch_id TEXT,
                model_path TEXT,
                performance_metrics TEXT,  -- JSON
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT TRUE,
                FOREIGN KEY (training_batch_id) REFERENCES training_batches (batch_id)
            )
        ''')
        
        # RAG queries log table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS rag_queries (
                query_id TEXT PRIMARY KEY,
                query_text TEXT NOT NULL,
                region TEXT,
                retrieved_article_ids TEXT,  -- JSON array
                response_text TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                response_quality_score REAL
            )
        ''')
        
        self.conn.commit()
        print("Database schema created")
    
    def store_articles(self, articles: List[Dict]) -> int:
        """Store articles in SQLite"""
        stored_count = 0
        cursor = self.conn.cursor()
        
        for article in articles:
            try:
                # Generate unique ID
                article_id = f"{article['region']}_{article['source']}_{hash(article['title'])}_{article['published'][:10]}"
                
                # Calculate quality score
                quality_score = self._calculate_quality_score(article)
                word_count = len(article['summary'].split()) + len(article.get('content', '').split())
                
                cursor.execute('''
                    INSERT OR REPLACE INTO articles 
                    (id, title, summary, content, region, source, link, published, word_count, quality_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    article_id,
                    article['title'],
                    article['summary'],
                    article.get('content', ''),
                    article['region'],
                    article['source'],
                    article.get('link', ''),
                    article['published'],
                    word_count,
                    quality_score
                ))
                
                stored_count += 1
                
            except Exception as e:
                print(f"Error storing article: {e}")
                continue
        
        self.conn.commit()
        print(f"Stored {stored_count} articles in SQLite")
        return stored_count
    
    def get_articles_for_region(self, region: str, min_quality: float = 0.5, 
                               limit: int = 1000) -> List[Dict]:
        """Get high-quality articles for a region"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT id, title, summary, content, source, published, quality_score
            FROM articles 
            WHERE region = ? AND quality_score >= ?
            ORDER BY quality_score DESC, published DESC
            LIMIT ?
        ''', (region, min_quality, limit))
        
        articles = cursor.fetchall()
        
        return [{
            'id': row[0],
            'title': row[1],
            'summary': row[2],
            'content': row[3],
            'source': row[4],
            'published': row[5],
            'quality_score': row[6],
            'region': region
        } for row in articles]
    
    def get_unprocessed_articles_for_embedding(self, limit: int = 500) -> List[Dict]:
        """Get articles that haven't been processed for embeddings yet"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT id, title, summary, content, region, source
            FROM articles 
            WHERE processed_for_embedding = FALSE
            ORDER BY created_at DESC
            LIMIT ?
        ''', (limit,))
        
        articles = cursor.fetchall()
        
        return [{
            'id': row[0],
            'title': row[1],
            'summary': row[2],
            'content': row[3],
            'region': row[4],
            'source': row[5]
        } for row in articles]
    
    def mark_articles_processed_for_embedding(self, article_ids: List[str]):
        """Mark articles as processed for embeddings"""
        cursor = self.conn.cursor()
        placeholders = ','.join(['?' for _ in article_ids])
        cursor.execute(f'''
            UPDATE articles 
            SET processed_for_embedding = TRUE 
            WHERE id IN ({placeholders})
        ''', article_ids)
        self.conn.commit()
    
    def create_training_batch(self, region: str, article_ids: List[str], 
                            training_examples_count: int) -> str:
        """Create a new training batch"""
        batch_id = str(uuid.uuid4())
        
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO training_batches 
            (batch_id, region, article_ids, training_examples_count)
            VALUES (?, ?, ?, ?)
        ''', (batch_id, region, json.dumps(article_ids), training_examples_count))
        
        self.conn.commit()
        print(f"Created training batch {batch_id} for {region}")
        return batch_id
    
    def update_training_batch_status(self, batch_id: str, status: str, 
                                   epochs: int = 0, model_path: str = None):
        """Update training batch status"""
        cursor = self.conn.cursor()
        cursor.execute('''
            UPDATE training_batches 
            SET status = ?, epochs = ?, model_path = ?
            WHERE batch_id = ?
        ''', (status, epochs, model_path, batch_id))
        self.conn.commit()
    
    def log_rag_query(self, query_text: str, region: str, retrieved_article_ids: List[str], 
                     response_text: str, quality_score: float = None):
        """Log RAG query for analytics"""
        query_id = str(uuid.uuid4())
        
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO rag_queries 
            (query_id, query_text, region, retrieved_article_ids, response_text, response_quality_score)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (query_id, query_text, region, json.dumps(retrieved_article_ids), 
              response_text, quality_score))
        
        self.conn.commit()
        return query_id
    
    def _calculate_quality_score(self, article: Dict) -> float:
        """Calculate article quality score"""
        score = 0.0
        
        # Length scoring
        if len(article['summary']) > 100:
            score += 0.3
        if len(article.get('content', '')) > 500:
            score += 0.2
        
        # Title quality
        if len(article['title']) > 20:
            score += 0.2
        
        # Source reliability
        reliable_sources = ['reuters', 'bbc', 'ap_news', 'npr', 'dw', 'cnn']
        if any(source in article['source'].lower() for source in reliable_sources):
            score += 0.3
        
        return min(score, 1.0)
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        cursor = self.conn.cursor()
        
        # Article counts by region
        cursor.execute('''
            SELECT region, COUNT(*), AVG(quality_score)
            FROM articles 
            GROUP BY region
        ''')
        region_stats = cursor.fetchall()
        
        # Training batch stats
        cursor.execute('''
            SELECT status, COUNT(*)
            FROM training_batches
            GROUP BY status
        ''')
        batch_stats = cursor.fetchall()
        
        return {
            'region_stats': {row[0]: {'count': row[1], 'avg_quality': row[2]} 
                           for row in region_stats},
            'batch_stats': {row[0]: row[1] for row in batch_stats}
        }
    
    def close(self):
        """Close database connection"""
        self.conn.close()


if __name__ == "__main__":
    # Test the database
    db = SQLiteArticleDatabase()
    
    # Test article storage
    test_articles = [
        {
            'title': 'Test Article 1',
            'summary': 'This is a test summary with enough content to pass quality filters.',
            'content': 'More detailed content here.',
            'region': 'north_america',
            'source': 'cnn',
            'link': 'https://example.com/1',
            'published': '2024-01-01T12:00:00'
        }
    ]
    
    db.store_articles(test_articles)
    print("Database test completed")
    print(db.get_stats())