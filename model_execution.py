import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from rag_pipeline import NewsRAGPipeline
from sql_database import SQLiteArticleDatabase
import os
from typing import Dict, List, Optional

class ModelExecutionPipeline:
    """
    Handles model loading and inference with RAG integration
    """
    
    def __init__(self, model_dir: str = "./trained_models",
                 sql_db_path: str = "news_articles.db",
                 vector_db_path: str = "./chroma_news_db"):
        
        self.model_dir = model_dir
        self.loaded_models = {}  # Cache for loaded models
        self.tokenizer = None
        
        # Initialize RAG pipeline
        self.rag_pipeline = NewsRAGPipeline(sql_db_path, vector_db_path)
        
        print("Model execution pipeline initialized")
    
    def load_model(self, region: str, base_model: str = "meta-llama/Llama-3.2-1B") -> bool:
        """
        Load a trained regional model
        """
        
        model_path = os.path.join(self.model_dir, f"{region}_model")
        
        if not os.path.exists(model_path):
            print(f"Model not found for region {region} at {model_path}")
            return False
        
        try:
            print(f"Loading model for {region}...")
            
            # Load base model and tokenizer
            base_model_obj = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            
            # Load LoRA weights
            model = PeftModel.from_pretrained(base_model_obj, model_path)
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(base_model)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Cache the model
            self.loaded_models[region] = {
                'model': model,
                'tokenizer': tokenizer
            }
            
            # Set default tokenizer
            if self.tokenizer is None:
                self.tokenizer = tokenizer
            
            print(f"Model loaded successfully for {region}")
            return True
            
        except Exception as e:
            print(f"Error loading model for {region}: {e}")
            return False
    
    def generate_response(self, prompt: str, region: Optional[str] = None,
                         max_length: int = 200, temperature: float = 0.7) -> str:
        """
        Generate response using trained model
        """
        
        # Use consolidated model if no specific region
        model_key = region if region and region in self.loaded_models else "consolidated"
        
        if model_key not in self.loaded_models:
            # Try to load the model
            if not self.load_model(model_key):
                return f"Model not available for region: {model_key}"
        
        model_info = self.loaded_models[model_key]
        model = model_info['model']
        tokenizer = model_info['tokenizer']
        
        try:
            # Prepare input
            formatted_prompt = f"<|begin_of_text|>{prompt}\nResponse:"
            
            inputs = tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(model.device)
            
            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # Decode response
            response = tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            ).strip()
            
            return response
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return "Error generating response"
    
    def rag_enhanced_generation(self, query: str, region: Optional[str] = None,
                              use_model: bool = True) -> Dict:
        """
        Generate response using RAG + fine-tuned model
        """
        
        # Step 1: Retrieve relevant context
        rag_result = self.rag_pipeline.retrieve_and_augment(
            query=query,
            region=region,
            top_k=3
        )
        
        if rag_result['status'] != 'success':
            return {
                'query': query,
                'response': f"No relevant news found for: {query}",
                'sources': [],
                'method': 'fallback'
            }
        
        # Step 2: Create enhanced prompt with context
        context = rag_result['context']
        enhanced_prompt = f"""Based on the following recent news:

{context}

Question: {query}
Please provide a comprehensive answer based on the news context:"""
        
        # Step 3: Generate response
        if use_model and (region in self.loaded_models or "consolidated" in self.loaded_models):
            response = self.generate_response(
                prompt=enhanced_prompt,
                region=region,
                max_length=300
            )
            method = "rag_plus_model"
        else:
            # Fallback to simple context summarization
            response = self._create_simple_summary(rag_result['retrieved_articles'], query)
            method = "rag_only"
        
        return {
            'query': query,
            'response': response,
            'sources': [
                {
                    'title': article['title'],
                    'source': article['source'],
                    'published': article['published'][:10],
                    'similarity_score': article.get('similarity_score', 0)
                }
                for article in rag_result['retrieved_articles']
            ],
            'method': method,
            'num_sources': len(rag_result['retrieved_articles'])
        }
    
    def _create_simple_summary(self, articles: List[Dict], query: str) -> str:
        """Create a simple summary when model generation is not available"""
        
        if not articles:
            return f"No relevant information found for: {query}"
        
        summary = f"Based on recent news about {query}:\n\n"
        
        for i, article in enumerate(articles[:3], 1):
            summary += f"{i}. From {article['source']}: {article['summary'][:150]}...\n\n"
        
        return summary.strip()
    
    def batch_generate(self, queries: List[str], region: Optional[str] = None) -> List[Dict]:
        """
        Generate responses for multiple queries
        """
        
        results = []
        
        for query in queries:
            result = self.rag_enhanced_generation(query, region)
            results.append(result)
        
        return results
    
    def get_available_models(self) -> List[str]:
        """Get list of available trained models"""
        
        if not os.path.exists(self.model_dir):
            return []
        
        models = []
        for item in os.listdir(self.model_dir):
            item_path = os.path.join(self.model_dir, item)
            if os.path.isdir(item_path) and item.endswith('_model'):
                region = item.replace('_model', '')
                models.append(region)
        
        return models
    
    def preload_all_models(self) -> Dict[str, bool]:
        """Preload all available models"""
        
        available_models = self.get_available_models()
        results = {}
        
        for region in available_models:
            results[region] = self.load_model(region)
        
        return results


if __name__ == "__main__":
    # Test model execution
    executor = ModelExecutionPipeline()
    
    # Get available models
    models = executor.get_available_models()
    print(f"Available models: {models}")
    
    # Test RAG-enhanced generation
    test_queries = [
        "What's happening in European politics?",
        "Latest economic news",
        "Technology developments"
    ]
    
    for query in test_queries:
        result = executor.rag_enhanced_generation(query)
        print(f"\nQuery: {query}")
        print(f"Response: {result['response'][:200]}...")
        print(f"Sources: {result['num_sources']}")
        print(f"Method: {result['method']}")