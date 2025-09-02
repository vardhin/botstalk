import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from typing import Dict, List, Tuple
import json
import os

class ILoRATrainer:
    """
    I-LoRA implementation optimized for Llama 3.2 1B
    """
    
    def __init__(self, model_name: str = "meta-llama/Llama-3.2-1B", 
                 lora_rank: int = 16, lambda_interpolation: float = 0.1):
        self.model_name = model_name
        self.lora_rank = lora_rank
        self.lambda_interpolation = lambda_interpolation
        
        print(f"Loading {model_name}...")
        
        # Initialize base model - Llama 3.2 1B (NOT quantized)
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.bfloat16,  # Use BF16 for efficiency (not quantization)
            device_map="auto",
            trust_remote_code=True
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"Model loaded: {self.count_parameters()} parameters")
        
        # Dual memory system
        self.working_memory = None  # Fast learner (θw)
        self.long_term_memory = None  # Slow learner (θl)
        self.episodic_memory = []  # Experience replay buffer
        
        self.setup_lora_models()
    
    def count_parameters(self):
        """Count total model parameters"""
        total = sum(p.numel() for p in self.base_model.parameters())
        trainable = sum(p.numel() for p in self.base_model.parameters() if p.requires_grad)
        return f"{total:,} total, {trainable:,} trainable"
    
    def setup_lora_models(self):
        """Setup dual LoRA models optimized for Llama 3.2"""
        # Optimized LoRA config for Llama 3.2 1B
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.lora_rank,
            lora_alpha=32,  # 2 * rank for stability
            lora_dropout=0.1,
            # Target all attention matrices for better performance
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
            use_rslora=True  # Rank-stabilized LoRA for better training
        )
        
        print("Setting up I-LoRA dual memory system...")
        
        # Working memory (fast learner)
        self.working_memory = get_peft_model(self.base_model, lora_config)
        
        # Long-term memory (slow learner) - independent copy
        self.long_term_memory = get_peft_model(
            AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            ), 
            lora_config
        )
        
        # Initialize synchronization
        self._sync_memories()
        
        trainable_params = sum(p.numel() for p in self.working_memory.parameters() if p.requires_grad)
        print(f"LoRA trainable parameters: {trainable_params:,} ({trainable_params/1.23e9*100:.2f}% of total)")
    
    def _sync_memories(self):
        """Synchronize long-term memory with working memory"""
        working_state = self.working_memory.state_dict()
        self.long_term_memory.load_state_dict(working_state)
        print("Memory synchronization complete")
    
    def exponential_moving_average_update(self):
        """Update long-term memory using exponential moving average"""
        with torch.no_grad():
            for (name_w, param_w), (name_l, param_l) in zip(
                self.working_memory.named_parameters(),
                self.long_term_memory.named_parameters()
            ):
                if 'lora' in name_w and param_w.requires_grad:
                    # θl = λ * θl + (1-λ) * θw
                    param_l.data = (
                        self.lambda_interpolation * param_l.data + 
                        (1 - self.lambda_interpolation) * param_w.data
                    )
    
    def compute_embedding_distillation_loss(self, inputs, labels):
        """Compute MSE loss between working and long-term memory embeddings"""
        # Get embeddings from both memories
        with torch.no_grad():
            long_term_outputs = self.long_term_memory(**inputs, output_hidden_states=True)
            long_term_embeddings = long_term_outputs.hidden_states[-1]  # Last layer
        
        working_outputs = self.working_memory(**inputs, output_hidden_states=True)
        working_embeddings = working_outputs.hidden_states[-1]
        
        # MSE loss on embeddings
        embedding_loss = F.mse_loss(working_embeddings, long_term_embeddings)
        
        # Standard cross-entropy loss (ignore pad tokens)
        shift_logits = working_outputs.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        ce_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100
        )
        
        return ce_loss, embedding_loss
    
    def train_regional_model(self, regional_data: List[Dict], region_name: str, 
                           epochs: int = 3, batch_size: int = 2,  # Smaller batch for 1B model
                           learning_rate: float = 2e-4, gamma: float = 0.5):
        """
        Train model on regional data using I-LoRA approach
        Optimized for Llama 3.2 1B memory constraints
        """
        
        print(f"Training on {region_name} data with {len(regional_data)} samples")
        
        # Setup optimizer with appropriate settings for Llama 3.2
        optimizer = torch.optim.AdamW(
            self.working_memory.parameters(), 
            lr=learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.95)  # Optimized for Llama
        )
        
        # Cosine learning rate scheduler
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs * len(regional_data) // batch_size)
        
        # Prepare data
        train_data = self._prepare_training_data(regional_data, batch_size)
        
        self.working_memory.train()
        self.long_term_memory.eval()
        
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            for batch_idx, batch in enumerate(train_data):
                optimizer.zero_grad()
                
                # Move batch to correct device
                inputs = {
                    'input_ids': batch['input_ids'].to(self.working_memory.device),
                    'attention_mask': batch['attention_mask'].to(self.working_memory.device)
                }
                labels = batch['labels'].to(self.working_memory.device)
                
                # Compute losses
                ce_loss, embedding_loss = self.compute_embedding_distillation_loss(inputs, labels)
                
                # Sample from episodic memory if available
                if self.episodic_memory and batch_size > 1:
                    memory_batch = self._sample_episodic_memory(max(1, batch_size // 2))
                    memory_inputs = {
                        'input_ids': memory_batch['input_ids'].to(self.working_memory.device),
                        'attention_mask': memory_batch['attention_mask'].to(self.working_memory.device)
                    }
                    memory_labels = memory_batch['labels'].to(self.working_memory.device)
                    
                    memory_ce_loss, memory_embedding_loss = self.compute_embedding_distillation_loss(
                        memory_inputs, memory_labels
                    )
                    
                    # Combine losses
                    total_ce_loss = (ce_loss + memory_ce_loss) / 2
                    total_embedding_loss = (embedding_loss + memory_embedding_loss) / 2
                else:
                    total_ce_loss = ce_loss
                    total_embedding_loss = embedding_loss
                
                # Final loss
                loss = total_ce_loss + gamma * total_embedding_loss
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.working_memory.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                # Update long-term memory every few steps
                if batch_idx % 2 == 0:  # Update every 2 steps
                    self.exponential_moving_average_update()
                
                total_loss += loss.item()
                num_batches += 1
                
                if batch_idx % 10 == 0:
                    print(f"  Batch {batch_idx}/{len(train_data)}, Loss: {loss.item():.4f}")
            
            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")
        
        # Update episodic memory
        self._update_episodic_memory(regional_data[:50])  # Store fewer samples for 1B model
        
        print(f"Completed training on {region_name}")
    
    def _prepare_training_data(self, data: List[Dict], batch_size: int):
        """Convert data to batched tensors optimized for Llama 3.2"""
        batches = []
        
        for i in range(0, len(data), batch_size):
            batch_data = data[i:i + batch_size]
            
            # Prepare input text with proper Llama 3.2 format
            texts = []
            for item in batch_data:
                # Use Llama 3.2 chat format if possible
                text = f"<|begin_of_text|>Question: {item['text']}\nAnswer: {item['label']}<|end_of_text|>"
                texts.append(text)
            
            # Tokenize with appropriate settings
            encoding = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=384,  # Smaller max length for 1B model
                return_tensors='pt'
            )
            
            # Create labels (for language modeling)
            labels = encoding['input_ids'].clone()
            labels[labels == self.tokenizer.pad_token_id] = -100
            
            batch = {
                'input_ids': encoding['input_ids'],
                'attention_mask': encoding['attention_mask'],
                'labels': labels
            }
            
            batches.append(batch)
        
        return batches
    
    def _update_episodic_memory(self, new_samples: List[Dict]):
        """Update episodic memory with new samples"""
        # Simple reservoir sampling to maintain fixed memory size
        max_memory_size = 500
        
        for sample in new_samples:
            if len(self.episodic_memory) < max_memory_size:
                self.episodic_memory.append(sample)
            else:
                # Random replacement
                idx = np.random.randint(0, max_memory_size)
                self.episodic_memory[idx] = sample
    
    def _sample_episodic_memory(self, batch_size: int):
        """Sample from episodic memory for experience replay"""
        if len(self.episodic_memory) < batch_size:
            sampled = self.episodic_memory
        else:
            indices = np.random.choice(len(self.episodic_memory), batch_size, replace=False)
            sampled = [self.episodic_memory[i] for i in indices]
        
        return self._prepare_training_data(sampled, len(sampled))[0]
    
    def generate_qa_pairs(self, prompt: str, num_questions: int = 50) -> List[Dict]:
        """Generate Q&A pairs using Llama 3.2 optimized generation"""
        self.working_memory.eval()
        qa_pairs = []
        
        print(f"Generating {num_questions} Q&A pairs for region prompt: {prompt}")
        
        for i in range(num_questions):
            try:
                # Generate question with Llama 3.2 format
                question_prompt = f"<|begin_of_text|>Generate a specific question about {prompt}:\nQuestion:"
                
                question_inputs = self.tokenizer(
                    question_prompt,
                    return_tensors='pt',
                    truncation=True,
                    max_length=200
                ).to(self.working_memory.device)
                
                with torch.no_grad():
                    question_outputs = self.working_memory.generate(
                        **question_inputs,
                        max_new_tokens=60,
                        temperature=0.8,
                        do_sample=True,
                        top_p=0.9,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                
                question = self.tokenizer.decode(
                    question_outputs[0][question_inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                ).strip()
                
                # Generate answer
                answer_prompt = f"<|begin_of_text|>Question: {question}\nAnswer:"
                
                answer_inputs = self.tokenizer(
                    answer_prompt,
                    return_tensors='pt',
                    truncation=True,
                    max_length=300
                ).to(self.working_memory.device)
                
                with torch.no_grad():
                    answer_outputs = self.working_memory.generate(
                        **answer_inputs,
                        max_new_tokens=100,
                        temperature=0.7,
                        do_sample=True,
                        top_p=0.9,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                
                answer = self.tokenizer.decode(
                    answer_outputs[0][answer_inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                ).strip()
                
                if question and answer and len(question) > 10 and len(answer) > 10:
                    qa_pairs.append({
                        "text": question,
                        "label": answer,
                        "region": prompt.split()[-1] if prompt else "unknown"
                    })
                    
                    if (i + 1) % 10 == 0:
                        print(f"  Generated {i + 1}/{num_questions} Q&A pairs")
                
            except Exception as e:
                print(f"Error generating Q&A pair {i+1}: {e}")
                continue
        
        print(f"Successfully generated {len(qa_pairs)} Q&A pairs")
        return qa_pairs


def run_llama32_experiment():
    """
    Experiment using Llama 3.2 1B with I-LoRA
    """
    
    # Initialize I-LoRA trainer with Llama 3.2 1B
    trainer = ILoRATrainer(
        model_name="meta-llama/Llama-3.2-1B",
        lora_rank=16,
        lambda_interpolation=0.1
    )
    
    # Your regional data
    regions = {
        "north_america": [
            {"text": "What's the main economic indicator in the US?", "label": "GDP growth rate and unemployment rate"},
            {"text": "How does healthcare work in Canada?", "label": "Universal healthcare funded by taxes"},
            {"text": "What's the capital of Mexico?", "label": "Mexico City"},
            # Add more data...
        ],
        "europe": [
            {"text": "What's the European Union's main goal?", "label": "Economic and political integration"},
            {"text": "How does Brexit affect trade?", "label": "Creates trade barriers with EU"},
            {"text": "What's the currency of Germany?", "label": "Euro"},
            # Add more data...
        ]
    }
    
    all_qa_pairs = []
    
    # Phase 1: Train regional models
    print("=== Phase 1: Training Regional Models with Llama 3.2 1B ===")
    for region, data in regions.items():
        print(f"\nTraining model for {region}...")
        
        # Train on regional data
        trainer.train_regional_model(data, region, epochs=2, batch_size=1)  # Small batch for 1B
        
        # Generate Q&A pairs
        regional_prompt = f"news and information about {region.replace('_', ' ')}"
        qa_pairs = trainer.generate_qa_pairs(regional_prompt, num_questions=20)  # Fewer for testing
        
        # Store Q&A pairs
        all_qa_pairs.extend(qa_pairs)
        
        # Save regional model
        trainer.save_model("./regional_models", region)
        
        print(f"Generated {len(qa_pairs)} Q&A pairs for {region}")
    
    # Phase 2: Consolidation training
    print("\n=== Phase 2: Consolidation Training ===")
    print(f"Training final model on {len(all_qa_pairs)} consolidated Q&A pairs")
    
    # Train final model on consolidated data
    if all_qa_pairs:
        trainer.train_regional_model(all_qa_pairs, "consolidated", epochs=3, batch_size=1)
        trainer.save_model("./final_model", "consolidated")
    
    print("\n=== Experiment Complete ===")
    print(f"Trained on {len(regions)} regions")
    print(f"Generated {len(all_qa_pairs)} total Q&A pairs")
    print("Models saved to ./regional_models and ./final_model")


if __name__ == "__main__":
    # Run the experiment with Llama 3.2 1B
    run_llama32_experiment()