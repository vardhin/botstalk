import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import subprocess
from typing import Optional, List
import os

class Navarasa:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_name = "Telugu-LLM-Labs/Indic-gemma-7b-finetuned-sft-Navarasa-2.0"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.loaded = False
        
    def check_dependencies(self) -> bool:
        """Check if required dependencies are installed"""
        try:
            import transformers
            import torch
            return True
        except ImportError:
            return False
    
    def install_dependencies(self) -> bool:
        """Install required dependencies"""
        try:
            print("Installing required dependencies...")
            # Install transformers and torch if not available
            subprocess.run([
                "pip", "install", "torch", "torchvision", "torchaudio", 
                "--index-url", "https://download.pytorch.org/whl/cu121"
            ], check=True)
            subprocess.run(["pip", "install", "transformers"], check=True)
            print("Dependencies installed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Failed to install dependencies: {e}")
            return False
    
    def load_model(self, hf_token: Optional[str] = None) -> bool:
        """Load the Navarasa model"""
        if self.loaded:
            print("Model already loaded!")
            return True
            
        try:
            print(f"Loading Navarasa model on {self.device}...")
            print("This may take a few minutes for the first time...")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                token=hf_token
            )
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                load_in_4bit=False,
                token=hf_token,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            self.loaded = True
            print(f"✓ Navarasa model loaded successfully on {self.device}!")
            return True
            
        except Exception as e:
            print(f"✗ Failed to load Navarasa model: {e}")
            print("This might be due to:")
            print("1. Network connectivity issues")
            print("2. Insufficient GPU/RAM memory")
            print("3. Missing Hugging Face token (if required)")
            return False
    
    def format_prompt(self, instruction: str, input_text: str = "") -> str:
        """Format prompt according to Navarasa's expected format"""
        prompt_template = """### Instruction:
{}

### Input:
{}

### Response:
{}"""
        
        return prompt_template.format(instruction, input_text, "")
    
    def generate_response(self, prompt: str, max_new_tokens: int = 300) -> Optional[str]:
        """Generate response using the Navarasa model"""
        if not self.loaded:
            print("Model not loaded. Please load the model first.")
            return None
            
        try:
            # Tokenize input
            inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            full_response = self.tokenizer.batch_decode(outputs)[0]
            
            # Extract only the generated part (after "### Response:")
            if "### Response:" in full_response:
                response = full_response.split("### Response:")[-1].strip()
                # Remove any special tokens
                response = response.replace("<eos>", "").replace("<pad>", "").strip()
                return response
            else:
                return full_response.replace(prompt, "").strip()
                
        except Exception as e:
            print(f"Error generating response: {e}")
            return None
    
    def generate_qa_pairs(self, article_content: str) -> Optional[str]:
        """Generate Q&A pairs for an article using Navarasa"""
        instruction = """Provide me some good quality question and answers for the attached article to generate synthetic training data to perform a parameter efficient fine tuning on an LLM. The questions should be how a user might ask about the article content, and the answers should be how the model must respond to that question.

Strictly follow this format:

QUESTION: <enter your question>
ANSWER: <enter your answer>

QUESTION: <enter your question>
ANSWER: <enter your answer>

Generate 3-5 high-quality question-answer pairs that cover different aspects of the article."""
        
        formatted_prompt = self.format_prompt(instruction, article_content)
        return self.generate_response(formatted_prompt, max_new_tokens=800)
    
    def is_available(self) -> bool:
        """Check if Navarasa can be used (dependencies + GPU/sufficient RAM)"""
        if not self.check_dependencies():
            return False
            
        # Check if we have enough memory (rough estimate)
        if torch.cuda.is_available():
            try:
                # Check GPU memory (Navarasa needs ~16GB for full precision)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                gpu_memory_gb = gpu_memory / (1024**3)
                return gpu_memory_gb >= 12  # Minimum 12GB for safe operation
            except:
                return False
        else:
            # For CPU, check system RAM (needs ~32GB+)
            try:
                import psutil
                ram_gb = psutil.virtual_memory().total / (1024**3)
                return ram_gb >= 24  # Minimum 24GB RAM for CPU inference
            except ImportError:
                # If psutil not available, assume it might work
                return True
    
    def get_model_info(self) -> dict:
        """Get information about the Navarasa model"""
        return {
            "name": "Navarasa 2.0",
            "full_name": self.model_name,
            "description": "Indic multilingual LLM based on Gemma-7B, supports 15 Indian languages + English",
            "languages": [
                "Hindi", "Telugu", "Marathi", "Urdu", "Assamese", "Konkani", 
                "Nepali", "Sindhi", "Tamil", "Kannada", "Malayalam", "Gujarati", 
                "Punjabi", "Bengali", "Odia", "English"
            ],
            "size": "7B parameters",
            "device": self.device,
            "loaded": self.loaded
        }
    
    def unload_model(self):
        """Unload model to free memory"""
        if self.loaded:
            del self.model
            del self.tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.model = None
            self.tokenizer = None
            self.loaded = False
            print("Navarasa model unloaded.")

# Convenience functions for external use
def create_navarasa_instance() -> Navarasa:
    """Create a new Navarasa instance"""
    return Navarasa()

def is_navarasa_available() -> bool:
    """Check if Navarasa can be used on this system"""
    navarasa = Navarasa()
    return navarasa.is_available()

def get_navarasa_info() -> dict:
    """Get information about Navarasa model"""
    navarasa = Navarasa()
    return navarasa.get_model_info()