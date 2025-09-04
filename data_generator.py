import uuid
import json
import re
import time
import os
from typing import List, Dict, Optional, Tuple
import requests
import subprocess
from dotenv import load_dotenv

import news_store
import training_data_store
import navarasa  # Import the new navarasa module

class DataGenerator:
    def __init__(self):
        self.selected_model = None
        self.model_type = None  # 'gemini', 'ollama', or 'navarasa'
        self.gemini_api_key = None
        self.navarasa_instance = None
        self.processed_articles = set()  # Track articles being processed in current session
        
        # Load environment variables
        load_dotenv()
        
    def get_gemini_api_key(self) -> Optional[str]:
        """Get Gemini API key from .env or user input"""
        # Try to get from .env file first
        api_key = os.getenv('GEMINI_API_KEY')
        
        if api_key:
            print("✓ Found Gemini API key in .env file")
            return api_key
        else:
            print("✗ Gemini API key not found in .env file")
            print("\nTo avoid entering the API key each time, add it to your .env file:")
            print("Format: GEMINI_API_KEY=your_api_key_here")
            print("Example: GEMINI_API_KEY=AIzaSyBxxxxxxxxxxxxxxxxxxxxxxx")
            print("\nAlternatively, you can enter it now:")
            
            api_key = input("Enter your Gemini API key: ").strip()
            if not api_key:
                print("API key is required for Gemini models.")
                return None
            return api_key
        
    def get_gemini_models(self) -> List[str]:
        """Get available Gemini models from API"""
        if not self.gemini_api_key:
            return []
        
        url = "https://generativelanguage.googleapis.com/v1beta/models"
        headers = {
            "x-goog-api-key": self.gemini_api_key
        }
        
        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            models = []
            
            if "models" in result:
                for model in result["models"]:
                    model_name = model.get("name", "")
                    # Extract just the model name (remove "models/" prefix)
                    if model_name.startswith("models/"):
                        model_name = model_name[7:]
                    
                    # Filter for text generation models only
                    supported_methods = model.get("supportedGenerationMethods", [])
                    if "generateContent" in supported_methods:
                        models.append(model_name)
            
            return sorted(models)  # Sort alphabetically
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching Gemini models: {e}")
            # Fallback to hardcoded models if API fails
            return [
                "gemini-1.5-flash",
                "gemini-1.5-pro", 
                "gemini-1.0-pro"
            ]
        except KeyError as e:
            print(f"Unexpected API response format: {e}")
            # Fallback to hardcoded models
            return [
                "gemini-1.5-flash",
                "gemini-1.5-pro",
                "gemini-1.0-pro"
            ]
    
    def get_ollama_models(self) -> List[str]:
        """Get available Ollama models"""
        try:
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                models = []
                for line in lines:
                    if line.strip():
                        model_name = line.split()[0]
                        models.append(model_name)
                return models
            else:
                print("Error getting Ollama models:", result.stderr)
                return []
        except FileNotFoundError:
            print("Ollama not found. Please install Ollama first.")
            return []
    
    def select_model(self):
        """Interactive model selection"""
        print("\n=== Model Selection ===")
        print("1. Gemini API (Online)")
        print("2. Ollama (Offline)")
        print("3. Navarasa 2.0 (Local - Indic Multilingual)")
        
        choice = input("Select model type (1, 2, or 3): ").strip()
        
        if choice == "1":
            self.model_type = "gemini"
            
            # Get API key from .env or user input
            api_key = self.get_gemini_api_key()
            if not api_key:
                return False
            self.gemini_api_key = api_key
            
            print("\nFetching available Gemini models...")
            gemini_models = self.get_gemini_models()
            
            if not gemini_models:
                print("No Gemini models available.")
                return False
                
            print("\nAvailable Gemini models:")
            for i, model in enumerate(gemini_models, 1):
                print(f"{i}. {model}")
            
            model_choice = input(f"Select model (1-{len(gemini_models)}): ").strip()
            try:
                model_idx = int(model_choice) - 1
                if 0 <= model_idx < len(gemini_models):
                    self.selected_model = gemini_models[model_idx]
                    print(f"Selected: {self.selected_model}")
                    return True
                else:
                    print("Invalid model selection.")
                    return False
            except ValueError:
                print("Invalid input.")
                return False
                
        elif choice == "2":
            self.model_type = "ollama"
            ollama_models = self.get_ollama_models()
            
            if not ollama_models:
                print("No Ollama models found.")
                return False
                
            print("\nAvailable Ollama models:")
            for i, model in enumerate(ollama_models, 1):
                print(f"{i}. {model}")
            
            model_choice = input(f"Select model (1-{len(ollama_models)}): ").strip()
            try:
                model_idx = int(model_choice) - 1
                if 0 <= model_idx < len(ollama_models):
                    self.selected_model = ollama_models[model_idx]
                    print(f"Selected: {self.selected_model}")
                    return True
                else:
                    print("Invalid model selection.")
                    return False
            except ValueError:
                print("Invalid input.")
                return False
                
        elif choice == "3":
            self.model_type = "navarasa"
            
            # Check if Navarasa is available
            if not navarasa.is_navarasa_available():
                print("\n❌ Navarasa is not available on this system.")
                print("Requirements:")
                print("- CUDA GPU with 12GB+ VRAM (recommended)")
                print("- OR 24GB+ system RAM for CPU inference")
                print("- PyTorch and Transformers libraries")
                
                install_deps = input("Try to install missing dependencies? (y/n): ").strip().lower()
                if install_deps in ['y', 'yes']:
                    navarasa_temp = navarasa.NavaraSa()
                    if not navarasa_temp.install_dependencies():
                        return False
                else:
                    return False
            
            # Show Navarasa info
            info = navarasa.get_navarasa_info()
            print(f"\n📋 Model Information:")
            print(f"   Name: {info['name']}")
            print(f"   Size: {info['size']}")
            print(f"   Device: {info['device']}")
            print(f"   Languages: {', '.join(info['languages'][:5])}... (+{len(info['languages'])-5} more)")
            
            # Ask for HF token if needed
            hf_token = os.getenv('HF_TOKEN')
            if not hf_token:
                print("\n💡 Optional: Hugging Face token can help with faster downloads")
                hf_token = input("Enter HF token (or press Enter to skip): ").strip()
                if not hf_token:
                    hf_token = None
            
            # Initialize and load Navarasa
            self.navarasa_instance = navarasa.create_navarasa_instance()
            
            print("\n🔄 Loading Navarasa model...")
            if not self.navarasa_instance.load_model(hf_token):
                print("Failed to load Navarasa model.")
                return False
            
            self.selected_model = "Navarasa-2.0"
            print(f"✅ Selected: {self.selected_model}")
            return True
            
        else:
            print("Invalid choice.")
            return False
    
    def is_article_already_processed(self, article_uid: str) -> bool:
        """Check if article has already been processed"""
        # Check if article is being processed in current session
        if article_uid in self.processed_articles:
            return True
            
        # Check database for existing training data from this article
        existing_data = training_data_store.get_training_data_by_article_uuid(article_uid)
        return len(existing_data) > 0
    
    def call_gemini_api(self, prompt: str, article_content: str) -> Optional[str]:
        """Call Gemini API with rate limiting"""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.selected_model}:generateContent"
        
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.gemini_api_key
        }
        
        full_prompt = f"{prompt}\n\nArticle:\n{article_content}"
        
        data = {
            "contents": [{
                "parts": [{
                    "text": full_prompt
                }]
            }]
        }
        
        try:
            # Rate limiting - wait 1 second between requests
            time.sleep(1)
            
            response = requests.post(url, headers=headers, json=data, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            if "candidates" in result and result["candidates"]:
                return result["candidates"][0]["content"]["parts"][0]["text"]
            else:
                print("No response from Gemini API")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"Error calling Gemini API: {e}")
            return None
        except KeyError as e:
            print(f"Unexpected response format: {e}")
            return None
    
    def call_ollama(self, prompt: str, article_content: str) -> Optional[str]:
        """Call Ollama model"""
        full_prompt = f"{prompt}\n\nArticle:\n{article_content}"
        
        try:
            result = subprocess.run([
                'ollama', 'run', self.selected_model, full_prompt
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                print(f"Error calling Ollama: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            print("Ollama call timed out")
            return None
        except FileNotFoundError:
            print("Ollama not found")
            return None
    
    def generate_qa_pairs(self, article_content: str) -> Optional[str]:
        """Generate Q&A pairs for an article"""
        if self.model_type == "navarasa":
            if not self.navarasa_instance:
                print("Navarasa instance not available")
                return None
            return self.navarasa_instance.generate_qa_pairs(article_content)
        elif self.model_type == "gemini":
            prompt = """Provide me some good quality question and answers for the attached article to generate synthetic training data to perform a parameter efficient fine tuning on an LLM. The questions should be how a user might ask about the article content, and the answers should be how the model must respond to that question.

Strictly follow this format:

QUESTION: <enter your question>
ANSWER: <enter your answer>

QUESTION: <enter your question>
ANSWER: <enter your answer>

Generate 3-5 high-quality question-answer pairs that cover different aspects of the article."""
            return self.call_gemini_api(prompt, article_content)
        elif self.model_type == "ollama":
            prompt = """Provide me some good quality question and answers for the attached article to generate synthetic training data to perform a parameter efficient fine tuning on an LLM. The questions should be how a user might ask about the article content, and the answers should be how the model must respond to that question.

Strictly follow this format:

QUESTION: <enter your question>
ANSWER: <enter your answer>

QUESTION: <enter your question>
ANSWER: <enter your answer>

Generate 3-5 high-quality question-answer pairs that cover different aspects of the article."""
            return self.call_ollama(prompt, article_content)
        else:
            return None
    
    def parse_qa_response(self, response: str) -> List[Tuple[str, str]]:
        """Parse the model response to extract Q&A pairs"""
        qa_pairs = []
        
        # Split by QUESTION: to get individual pairs
        sections = re.split(r'\n*QUESTION:\s*', response, flags=re.IGNORECASE)
        
        for section in sections[1:]:  # Skip first empty section
            # Split each section by ANSWER:
            parts = re.split(r'\n*ANSWER:\s*', section, maxsplit=1, flags=re.IGNORECASE)
            
            if len(parts) == 2:
                question = parts[0].strip()
                answer = parts[1].strip()
                
                # Clean up the answer (remove next QUESTION if present)
                answer = re.split(r'\n*QUESTION:\s*', answer, flags=re.IGNORECASE)[0].strip()
                
                if question and answer:
                    qa_pairs.append((question, answer))
        
        return qa_pairs
    
    def process_article(self, article: Tuple) -> bool:
        """Process a single article and generate training data"""
        uid, title, content, date, state = article
        
        print(f"\nProcessing article: {title}")
        
        # Check if already processed
        if self.is_article_already_processed(uid):
            print(f"Article '{title}' already processed. Skipping.")
            return True
        
        # Mark as being processed
        self.processed_articles.add(uid)
        
        # Generate Q&A pairs
        response = self.generate_qa_pairs(content)
        if not response:
            print(f"Failed to generate Q&A pairs for article: {title}")
            return False
        
        # Parse Q&A pairs
        qa_pairs = self.parse_qa_response(response)
        
        if not qa_pairs:
            print(f"No valid Q&A pairs found in response for article: {title}")
            return False
        
        print(f"Found {len(qa_pairs)} Q&A pairs")
        
        # Store each Q&A pair in training data store
        success_count = 0
        for question, answer in qa_pairs:
            training_uuid = str(uuid.uuid4())
            
            success = training_data_store.create_training_data(
                uuid=training_uuid,
                question=question,
                answer=answer,
                state="generated",
                uuid_of_used_article=uid,
                master_model=self.selected_model
            )
            
            if success:
                success_count += 1
                print(f"  ✓ Stored Q&A pair {success_count}")
            else:
                print(f"  ✗ Failed to store Q&A pair (question may already exist)")
        
        print(f"Successfully stored {success_count}/{len(qa_pairs)} Q&A pairs")
        return success_count > 0
    
    def run(self):
        """Main execution method"""
        print("=== Training Data Generator ===")
        
        # Model selection
        if not self.select_model():
            print("Model selection failed. Exiting.")
            return
        
        # Get articles to process
        print(f"\nSelected model: {self.selected_model} ({self.model_type})")
        
        # Get all articles from both states
        all_articles = []
        
        # Fetch articles from Andhra Pradesh
        ap_articles = news_store.get_articles_by_state("Andhra Pradesh")
        all_articles.extend(ap_articles)
        print(f"Found {len(ap_articles)} articles from Andhra Pradesh")
        
        # Fetch articles from Telangana  
        ts_articles = news_store.get_articles_by_state("Telangana")
        all_articles.extend(ts_articles)
        print(f"Found {len(ts_articles)} articles from Telangana")
        
        if not all_articles:
            print("No articles found in the database.")
            return
        
        print(f"Total articles to process: {len(all_articles)}")
        
        # Ask user if they want to process all articles
        confirm = input(f"\nProcess all {len(all_articles)} articles? (y/n): ").strip().lower()
        if confirm not in ['y', 'yes']:
            print("Processing cancelled.")
            return
        
        # Process articles
        processed_count = 0
        skipped_count = 0
        failed_count = 0
        
        for i, article in enumerate(all_articles, 1):
            print(f"\n--- Article {i}/{len(all_articles)} ---")
            
            try:
                if self.process_article(article):
                    processed_count += 1
                else:
                    skipped_count += 1
            except Exception as e:
                print(f"Error processing article: {e}")
                failed_count += 1
            
            # Show progress every 10 articles
            if i % 10 == 0:
                print(f"\n🔄 Progress Update:")
                print(f"   Checked: {i}/{len(all_articles)} articles")
                print(f"   ✅ Processed: {processed_count}")
                print(f"   ⏭️  Skipped: {skipped_count}")
                print(f"   ❌ Failed: {failed_count}")
        
        print(f"\n=== Final Summary ===")
        print(f"📊 Total articles checked: {len(all_articles)}")
        print(f"✅ Successfully processed: {processed_count}")
        print(f"⏭️  Skipped (already processed): {skipped_count}")
        print(f"❌ Failed to process: {failed_count}")
        print(f"📈 Success rate: {(processed_count/(len(all_articles)-skipped_count)*100):.1f}%" if (len(all_articles)-skipped_count) > 0 else "N/A")
        print("\n🎉 Training data generation complete!")

    def __del__(self):
        """Cleanup when object is destroyed"""
        if self.navarasa_instance:
            self.navarasa_instance.unload_model()

def main():
    generator = DataGenerator()
    generator.run()

if __name__ == "__main__":
    main()