import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Dict, List, Optional
import uuid

class ChromaDBVectorStore:
    """
    ChromaDB vector database for article embeddings
    """
    
    def __init__(self, db_path: str = "./chroma_news_db", 
                 embedding_model: str = "all-MiniLM-L6-v2"):
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=db_path)
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name="news_articles",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(embedding_model)
        print(f"ChromaDB initialized: {db_path}")
        print(f"Embedding model: {embedding_model}")
    
    def add_articles(self, articles: List[Dict]) -> int:
        """Add articles to vector database"""
        
        if not articles:
            return 0
        
        # Prepare data for ChromaDB
        embeddings = []
        documents = []
        metadatas = []
        ids = []
        
        print(f"Generating embeddings for {len(articles)} articles...")
        
        for article in articles:
            # Create text for embedding
            text_for_embedding = f"{article['title']} {article['summary']}"
            if article.get('content'):
                text_for_embedding += f" {article['content'][:500]}"
            
            # Generate embedding
            embedding = self.embedding_model.encode(text_for_embedding)
            
            embeddings.append(embedding.tolist())
            documents.append(text_for_embedding)
            
            # Metadata for filtering and retrieval
            metadata = {
                "article_id": article['id'],
                "region": article['region'],
                "source": article['source'],
                "title": article['title'][:100],  # Truncate for ChromaDB
                "published": article.get('published', ''),
                "quality_score": article.get('quality_score', 0.0)
            }
            metadatas.append(metadata)
            ids.append(article['id'])
        
        # Add to ChromaDB
        try:
            self.collection.add(
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            print(f"Added {len(articles)} articles to vector database")
            return len(articles)
            
        except Exception as e:
            print(f"Error adding to ChromaDB: {e}")
            return 0
    
    def semantic_search(self, query: str, region: Optional[str] = None,
                       top_k: int = 5, min_quality: float = 0.0) -> List[Dict]:
        """Semantic search for articles"""
        
        # Build where filter
        where_filter = {}
        if region:
            where_filter["region"] = region
        if min_quality > 0:
            where_filter["quality_score"] = {"$gte": min_quality}
        
        try:
            # Search
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where_filter if where_filter else None,
                include=["metadatas", "documents", "distances"]
            )
            
            # Format results
            search_results = []
            if results['ids'] and results['ids'][0]:
                for i, article_id in enumerate(results['ids'][0]):
                    metadata = results['metadatas'][0][i]
                    document = results['documents'][0][i]
                    distance = results['distances'][0][i]
                    
                    search_results.append({
                        'article_id': article_id,
                        'title': metadata['title'],
                        'region': metadata['region'],
                        'source': metadata['source'],
                        'published': metadata['published'],
                        'quality_score': metadata['quality_score'],
                        'document_text': document,
                        'similarity_score': 1 - distance,  # Convert distance to similarity
                        'distance': distance
                    })
            
            print(f"Found {len(search_results)} results for query: {query[:50]}...")
            return search_results
            
        except Exception as e:
            print(f"Error in semantic search: {e}")
            return []
    
    def get_similar_articles(self, article_id: str, top_k: int = 5) -> List[Dict]:
        """Find articles similar to a given article"""
        
        try:
            # Get the article's embedding
            result = self.collection.get(
                ids=[article_id],
                include=["embeddings", "metadatas"]
            )
            
            if not result['embeddings']:
                print(f"Article {article_id} not found")
                return []
            
            # Search for similar articles
            embedding = result['embeddings'][0]
            
            similar_results = self.collection.query(
                query_embeddings=[embedding],
                n_results=top_k + 1,  # +1 to exclude the original article
                include=["metadatas", "distances"]
            )
            
            # Filter out the original article and format results
            similar_articles = []
            for i, sim_id in enumerate(similar_results['ids'][0]):
                if sim_id != article_id:  # Exclude original
                    metadata = similar_results['metadatas'][0][i]
                    distance = similar_results['distances'][0][i]
                    
                    similar_articles.append({
                        'article_id': sim_id,
                        'title': metadata['title'],
                        'region': metadata['region'],
                        'source': metadata['source'],
                        'similarity_score': 1 - distance,
                        'distance': distance
                    })
            
            return similar_articles[:top_k]
            
        except Exception as e:
            print(f"Error finding similar articles: {e}")
            return []
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the vector collection"""
        
        try:
            count = self.collection.count()
            
            # Get sample of metadata to analyze
            sample = self.collection.get(
                limit=min(1000, count),
                include=["metadatas"]
            )
            
            # Analyze regions and sources
            regions = {}
            sources = {}
            
            if sample['metadatas']:
                for metadata in sample['metadatas']:
                    region = metadata.get('region', 'unknown')
                    source = metadata.get('source', 'unknown')
                    
                    regions[region] = regions.get(region, 0) + 1
                    sources[source] = sources.get(source, 0) + 1
            
            return {
                'total_articles': count,
                'regions': regions,
                'sources': sources,
                'sample_size': len(sample['metadatas']) if sample['metadatas'] else 0
            }
            
        except Exception as e:
            print(f"Error getting collection stats: {e}")
            return {'total_articles': 0, 'regions': {}, 'sources': {}}
    
    def delete_articles(self, article_ids: List[str]) -> bool:
        """Delete articles by IDs"""
        try:
            self.collection.delete(ids=article_ids)
            print(f"Deleted {len(article_ids)} articles from vector database")
            return True
        except Exception as e:
            print(f"Error deleting articles: {e}")
            return False
    
    def reset_collection(self):
        """Reset the entire collection (use carefully!)"""
        try:
            self.client.delete_collection("news_articles")
            self.collection = self.client.create_collection(
                name="news_articles",
                metadata={"hnsw:space": "cosine"}
            )
            print("Vector collection reset")
        except Exception as e:
            print(f"Error resetting collection: {e}")


if __name__ == "__main__":
    # Test the vector database
    vector_db = ChromaDBVectorStore()
    
    # Test data
    test_articles = [
        {
            'id': 'test_1',
            'title': 'Technology News Update',
            'summary': 'Latest developments in artificial intelligence and machine learning.',
            'region': 'north_america',
            'source': 'cnn',
            'quality_score': 0.8
        }
    ]
    
    # Test operations
    vector_db.add_articles(test_articles)
    results = vector_db.semantic_search("AI technology", top_k=3)
    print(f"Search results: {len(results)}")
    print(vector_db.get_collection_stats())