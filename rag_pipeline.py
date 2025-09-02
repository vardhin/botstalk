from vector_database import ChromaDBVectorStore
from sql_database import SQLiteArticleDatabase
from typing import Dict, List, Optional
import json

class NewsRAGPipeline:
    """
    RAG pipeline that combines vector search with SQL metadata
    """
    
    def __init__(self, sql_db_path: str = "news_articles.db",
                 vector_db_path: str = "./chroma_news_db"):
        
        self.sql_db = SQLiteArticleDatabase(sql_db_path)
        self.vector_db = ChromaDBVectorStore(vector_db_path)
        
        print("RAG pipeline initialized")
    
    def retrieve_and_augment(self, query: str, region: Optional[str] = None,
                           top_k: int = 5, min_quality: float = 0.5) -> Dict:
        """
        Main RAG retrieval function
        """
        
        # Step 1: Semantic search in vector database
        vector_results = self.vector_db.semantic_search(
            query=query,
            region=region,
            top_k=top_k * 2,  # Get more candidates
            min_quality=min_quality
        )
        
        if not vector_results:
            return {
                'query': query,
                'region': region,
                'retrieved_articles': [],
                'context': '',
                'status': 'no_results'
            }
        
        # Step 2: Get full article details from SQL database
        article_ids = [result['article_id'] for result in vector_results[:top_k]]
        
        retrieved_articles = []
        for result in vector_results[:top_k]:
            # Get full article from SQL
            sql_articles = self.sql_db.get_articles_for_region(
                region=result['region'], 
                limit=1000
            )
            
            # Find matching article
            full_article = None
            for article in sql_articles:
                if article['id'] == result['article_id']:
                    full_article = article
                    break
            
            if full_article:
                # Combine vector and SQL data
                enriched_article = {
                    **full_article,
                    'similarity_score': result['similarity_score'],
                    'distance': result['distance']
                }
                retrieved_articles.append(enriched_article)
        
        # Step 3: Create context for generation
        context = self._create_context(retrieved_articles)
        
        # Step 4: Log the query
        self.sql_db.log_rag_query(
            query_text=query,
            region=region,
            retrieved_article_ids=article_ids,
            response_text=context[:500]  # Truncated context
        )
        
        return {
            'query': query,
            'region': region,
            'retrieved_articles': retrieved_articles,
            'context': context,
            'status': 'success',
            'num_retrieved': len(retrieved_articles)
        }
    
    def _create_context(self, articles: List[Dict]) -> str:
        """Create context string from retrieved articles"""
        
        if not articles:
            return ""
        
        context_parts = []
        
        for i, article in enumerate(articles, 1):
            # Format each article
            article_context = f"""
Source {i} - {article['source']} ({article['published'][:10]}):
Title: {article['title']}
Summary: {article['summary']}
"""
            
            # Add content if available and not too long
            if article.get('content') and len(article['content']) > 100:
                article_context += f"Details: {article['content'][:300]}...\n"
            
            context_parts.append(article_context)
        
        return "\n".join(context_parts)
    
    def search_by_topic(self, topic: str, region: Optional[str] = None,
                       days_back: int = 7, top_k: int = 10) -> List[Dict]:
        """
        Search articles by topic with time filtering
        """
        
        # Use semantic search
        results = self.vector_db.semantic_search(
            query=topic,
            region=region,
            top_k=top_k
        )
        
        # Filter by date if needed (this would need date parsing in vector metadata)
        # For now, return semantic results
        return results
    
    def get_trending_topics(self, region: Optional[str] = None,
                          days_back: int = 3) -> List[Dict]:
        """
        Analyze trending topics (basic implementation)
        """
        
        # Get recent articles from SQL
        if region:
            articles = self.sql_db.get_articles_for_region(region, limit=200)
        else:
            # Would need a method to get articles across all regions
            articles = []
        
        # Simple keyword extraction (could be improved with NLP)
        word_counts = {}
        for article in articles:
            words = article['title'].lower().split()
            for word in words:
                if len(word) > 3:  # Filter short words
                    word_counts[word] = word_counts.get(word, 0) + 1
        
        # Sort by frequency
        trending = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        
        return [{'topic': word, 'frequency': count} for word, count in trending[:10]]
    
    def get_similar_news(self, article_id: str, top_k: int = 5) -> List[Dict]:
        """
        Find news articles similar to a given article
        """
        
        # Use vector similarity
        similar_vector_results = self.vector_db.get_similar_articles(
            article_id=article_id,
            top_k=top_k
        )
        
        # Enrich with SQL data
        enriched_results = []
        for result in similar_vector_results:
            # Get full article details
            # (This could be optimized with a batch SQL query)
            enriched_results.append(result)
        
        return enriched_results
    
    def generate_summary(self, query: str, region: Optional[str] = None) -> str:
        """
        Generate a summary response based on retrieved articles
        """
        
        # Retrieve relevant articles
        rag_result = self.retrieve_and_augment(
            query=query,
            region=region,
            top_k=3
        )
        
        if rag_result['status'] != 'success':
            return f"No relevant news found for: {query}"
        
        # Create a simple summary (could be enhanced with LLM)
        articles = rag_result['retrieved_articles']
        
        if not articles:
            return f"No relevant news found for: {query}"
        
        # Basic summary generation
        summary_parts = [f"Based on recent news sources, here's what's happening with {query}:"]
        
        for i, article in enumerate(articles[:3], 1):
            summary_parts.append(
                f"\n{i}. {article['source'].upper()}: {article['summary'][:200]}..."
            )
        
        return "\n".join(summary_parts)
    
    def get_pipeline_stats(self) -> Dict:
        """Get statistics about the RAG pipeline"""
        
        sql_stats = self.sql_db.get_stats()
        vector_stats = self.vector_db.get_collection_stats()
        
        return {
            'sql_database': sql_stats,
            'vector_database': vector_stats,
            'status': 'operational'
        }


if __name__ == "__main__":
    # Test the RAG pipeline
    rag = NewsRAGPipeline()
    
    # Test query
    result = rag.retrieve_and_augment("artificial intelligence technology")
    print(f"Retrieved {result['num_retrieved']} articles")
    
    # Test summary generation
    summary = rag.generate_summary("AI developments")
    print(f"Summary: {summary[:200]}...")
    
    # Get stats
    stats = rag.get_pipeline_stats()
    print(f"Pipeline stats: {stats}")