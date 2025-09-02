import torch
from main import ILoRATrainer  # Your existing I-LoRA implementation
from sql_database import SQLiteArticleDatabase
from typing import Dict, List
import os
import json

class ModelTrainingPipeline:
    """
    Handles model training with database integration
    """
    
    def __init__(self, sql_db_path: str = "news_articles.db",
                 model_save_dir: str = "./trained_models"):
        
        self.sql_db = SQLiteArticleDatabase(sql_db_path)
        self.model_save_dir = model_save_dir
        os.makedirs(model_save_dir, exist_ok=True)
        
        print("Model training pipeline initialized")
    
    def prepare_training_data(self, region: str, min_quality: float = 0.5,
                            max_examples: int = 500) -> List[Dict]:
        """
        Prepare training data from database
        """
        
        # Get high-quality articles
        articles = self.sql_db.get_articles_for_region(
            region=region,
            min_quality=min_quality,
            limit=max_examples // 2  # Each article generates ~2 training examples
        )
        
        if not articles:
            print(f"No articles found for region {region}")
            return []
        
        # Convert to training format
        training_data = []
        
        for article in articles:
            # Create multiple training examples per article
            examples = [
                {
                    "text": f"What's the latest news about {article['title'][:80]}?",
                    "label": article['summary'],
                    "source": article['source'],
                    "region": region,
                    "article_id": article['id']
                },
                {
                    "text": f"Tell me about news from {article['source']}",
                    "label": f"According to {article['source']}: {article['summary']}",
                    "source": article['source'],
                    "region": region,
                    "article_id": article['id']
                }
            ]
            
            # Add content-based example if available
            if article['content'] and len(article['content']) > 100:
                examples.append({
                    "text": f"Give me details about {article['title'][:60]}",
                    "label": article['content'][:800],
                    "source": article['source'],
                    "region": region,
                    "article_id": article['id']
                })
            
            training_data.extend(examples)
        
        print(f"Prepared {len(training_data)} training examples for {region}")
        return training_data[:max_examples]
    
    def train_regional_model(self, region: str, model_name: str = "meta-llama/Llama-3.2-1B",
                           epochs: int = 3, batch_size: int = 1) -> str:
        """
        Train a regional model and save it
        """
        
        print(f"Starting training for region: {region}")
        
        # Prepare training data
        training_data = self.prepare_training_data(region)
        
        if len(training_data) < 10:
            print(f"Insufficient training data for {region}: {len(training_data)} examples")
            return None
        
        # Create training batch in database
        article_ids = list(set([item['article_id'] for item in training_data]))
        batch_id = self.sql_db.create_training_batch(
            region=region,
            article_ids=article_ids,
            training_examples_count=len(training_data)
        )
        
        # Update status to training
        self.sql_db.update_training_batch_status(batch_id, "training")
        
        try:
            # Initialize I-LoRA trainer
            trainer = ILoRATrainer(
                model_name=model_name,
                lora_rank=16,
                lambda_interpolation=0.1
            )
            
            # Train the model
            trainer.train_regional_model(
                regional_data=training_data,
                region_name=region,
                epochs=epochs,
                batch_size=batch_size
            )
            
            # Save model
            model_path = os.path.join(self.model_save_dir, f"{region}_model")
            trainer.save_model(self.model_save_dir, region)
            
            # Update database with completion
            self.sql_db.update_training_batch_status(
                batch_id=batch_id,
                status="completed",
                epochs=epochs,
                model_path=model_path
            )
            
            print(f"Training completed for {region}. Model saved to {model_path}")
            return batch_id
            
        except Exception as e:
            print(f"Training failed for {region}: {e}")
            
            # Update status to failed
            self.sql_db.update_training_batch_status(batch_id, "failed")
            return None
    
    def train_all_regions(self, regions: List[str] = None) -> Dict[str, str]:
        """
        Train models for multiple regions
        """
        
        if regions is None:
            # Get regions from database
            stats = self.sql_db.get_stats()
            regions = list(stats['region_stats'].keys())
        
        results = {}
        
        for region in regions:
            print(f"\n=== Training model for {region} ===")
            batch_id = self.train_regional_model(region)
            results[region] = batch_id
        
        print(f"\nTraining completed for {len(results)} regions")
        return results
    
    def consolidation_training(self, regions: List[str], 
                             final_model_name: str = "consolidated_news_model") -> str:
        """
        Train a final consolidated model using data from all regions
        """
        
        print("Starting consolidation training...")
        
        # Collect training data from all regions
        all_training_data = []
        all_article_ids = []
        
        for region in regions:
            region_data = self.prepare_training_data(region, max_examples=200)
            all_training_data.extend(region_data)
            
            # Track article IDs
            region_article_ids = [item['article_id'] for item in region_data]
            all_article_ids.extend(region_article_ids)
        
        if len(all_training_data) < 50:
            print(f"Insufficient data for consolidation: {len(all_training_data)} examples")
            return None
        
        # Create consolidation batch
        batch_id = self.sql_db.create_training_batch(
            region="consolidated",
            article_ids=list(set(all_article_ids)),
            training_examples_count=len(all_training_data)
        )
        
        try:
            # Initialize trainer
            trainer = ILoRATrainer(
                model_name="meta-llama/Llama-3.2-1B",
                lambda_interpolation=0.05  # Lower lambda for final training
            )
            
            # Train consolidated model
            trainer.train_regional_model(
                regional_data=all_training_data,
                region_name="consolidated",
                epochs=5,
                batch_size=1
            )
            
            # Save consolidated model
            model_path = os.path.join(self.model_save_dir, final_model_name)
            trainer.save_model(self.model_save_dir, "consolidated")
            
            # Update database
            self.sql_db.update_training_batch_status(
                batch_id=batch_id,
                status="completed",
                epochs=5,
                model_path=model_path
            )
            
            print(f"Consolidation training completed. Model saved to {model_path}")
            return batch_id
            
        except Exception as e:
            print(f"Consolidation training failed: {e}")
            self.sql_db.update_training_batch_status(batch_id, "failed")
            return None
    
    def get_training_history(self) -> List[Dict]:
        """Get training history from database"""
        
        cursor = self.sql_db.conn.cursor()
        cursor.execute('''
            SELECT batch_id, region, training_examples_count, status, epochs, created_at, model_path
            FROM training_batches
            ORDER BY created_at DESC
        ''')
        
        batches = cursor.fetchall()
        
        return [{
            'batch_id': row[0],
            'region': row[1],
            'training_examples': row[2],
            'status': row[3],
            'epochs': row[4],
            'created_at': row[5],
            'model_path': row[6]
        } for row in batches]


if __name__ == "__main__":
    # Test training pipeline
    trainer = ModelTrainingPipeline()
    
    # Train a single region
    batch_id = trainer.train_regional_model("north_america")
    
    if batch_id:
        print(f"Training completed with batch ID: {batch_id}")
    
    # Get training history
    history = trainer.get_training_history()
    print(f"Training history: {len(history)} batches")