#!/usr/bin/env python3
"""
Qwen Fine-Tuning Script for PAI-DSW
Using LoRA (Low-Rank Adaptation) for efficient training

Usage:
    python train_qwen_lora.py
    
Requirements:
    - PAI-DSW instance with GPU
    - Preprocessed dataset in 'processed_dataset' folder
"""

import os
import torch

# Try ModelScope first (Alibaba Cloud), fallback to HuggingFace
try:
    from modelscope import AutoModelForCausalLM, AutoTokenizer
    print("Using ModelScope for model loading")
except ImportError:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("Using HuggingFace Transformers for model loading")

from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import load_from_disk
import wandb

# ==================== Configuration ====================

class Config:
    """Training configuration."""
    # Model - ModelScope model ID (recommended for PAI)
    # Options: qwen/Qwen-7B-Chat, qwen/Qwen-14B-Chat, qwen/Qwen-72B-Chat
    # For HuggingFace: Qwen/Qwen-7B-Chat
    MODEL_NAME = "qwen/Qwen-7B-Chat"
    
    # LoRA Configuration
    LORA_R = 64              # LoRA rank (8, 16, 32, 64)
    LORA_ALPHA = 16          # LoRA alpha
    LORA_DROPOUT = 0.1       # Dropout rate
    TARGET_MODULES = [       # Modules to apply LoRA
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
    
    # Training Configuration
    OUTPUT_DIR = "./qwen-lora-finetuned"
    NUM_EPOCHS = 3
    BATCH_SIZE = 4
    GRADIENT_ACCUMULATION_STEPS = 4
    LEARNING_RATE = 2e-4
    WARMUP_STEPS = 100
    MAX_GRAD_NORM = 0.3
    
    # Checkpointing
    SAVE_STEPS = 500
    LOGGING_STEPS = 10
    SAVE_TOTAL_LIMIT = 3
    
    # Data
    MAX_LENGTH = 2048
    DATASET_PATH = "processed_dataset"
    
    # Optimization
    FP16 = True
    BF16 = False  # Set True for A100
    GRADIENT_CHECKPOINTING = True
    GROUP_BY_LENGTH = True
    OPTIM = "paged_adamw_32bit"
    
    # Logging
    WANDB_PROJECT = "qwen-finetuning"
    WANDB_RUN_NAME = "qwen-7b-lora-demo"


# ==================== Model Setup ====================

def setup_model_and_tokenizer():
    """Load model and tokenizer from ModelScope."""
    
    print(f"Loading model: {Config.MODEL_NAME}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        Config.MODEL_NAME,
        trust_remote_code=True,
        pad_token='<|extra_0|>',
        eos_token='<|endoftext|>'
    )
    
    # Load model with automatic device mapping
    model = AutoModelForCausalLM.from_pretrained(
        Config.MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Prepare model for training (handles gradient checkpointing, etc.)
    model = prepare_model_for_kbit_training(model)
    
    print(f"Model loaded on device: {model.device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    return model, tokenizer


def setup_lora(model):
    """Configure and apply LoRA to model."""
    
    print("\nConfiguring LoRA...")
    
    lora_config = LoraConfig(
        r=Config.LORA_R,
        lora_alpha=Config.LORA_ALPHA,
        target_modules=Config.TARGET_MODULES,
        lora_dropout=Config.LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    return model


# ==================== Training Setup ====================

def setup_training_args():
    """Configure training arguments."""
    
    return TrainingArguments(
        output_dir=Config.OUTPUT_DIR,
        num_train_epochs=Config.NUM_EPOCHS,
        per_device_train_batch_size=Config.BATCH_SIZE,
        gradient_accumulation_steps=Config.GRADIENT_ACCUMULATION_STEPS,
        optim=Config.OPTIM,
        save_steps=Config.SAVE_STEPS,
        logging_steps=Config.LOGGING_STEPS,
        learning_rate=Config.LEARNING_RATE,
        warmup_steps=Config.WARMUP_STEPS,
        max_grad_norm=Config.MAX_GRAD_NORM,
        fp16=Config.FP16,
        bf16=Config.BF16,
        gradient_checkpointing=Config.GRADIENT_CHECKPOINTING,
        group_by_length=Config.GROUP_BY_LENGTH,
        report_to="wandb",
        run_name=Config.WANDB_RUN_NAME,
        remove_unused_columns=False,
        save_total_limit=Config.SAVE_TOTAL_LIMIT,
        load_best_model_at_end=False,
        ddp_find_unused_parameters=False,
    )


# ==================== Main Training ====================

def main():
    """Main training function."""
    
    print("=" * 60)
    print("Qwen Fine-Tuning with LoRA")
    print("=" * 60)
    
    # Initialize wandb
    wandb.init(
        project=Config.WANDB_PROJECT,
        name=Config.WANDB_RUN_NAME,
        config={
            "model": Config.MODEL_NAME,
            "lora_r": Config.LORA_R,
            "lora_alpha": Config.LORA_ALPHA,
            "learning_rate": Config.LEARNING_RATE,
            "batch_size": Config.BATCH_SIZE,
            "epochs": Config.NUM_EPOCHS,
        }
    )
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer()
    
    # Setup LoRA
    model = setup_lora(model)
    
    # Load dataset
    print(f"\nLoading dataset from: {Config.DATASET_PATH}")
    dataset = load_from_disk(Config.DATASET_PATH)
    print(f"Dataset size: {len(dataset)} examples")
    
    # Data collator for language modeling
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        pad_to_multiple_of=8,
        return_tensors="pt",
        padding=True,
        label_pad_token_id=-100
    )
    
    # Training arguments
    training_args = setup_training_args()
    
    # Initialize trainer
    print("\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    
    # Start training
    print("\nStarting training...")
    print("=" * 60)
    trainer.train()
    
    # Save final model
    print("\n" + "=" * 60)
    print(f"Saving model to: {Config.OUTPUT_DIR}")
    trainer.save_model(Config.OUTPUT_DIR)
    tokenizer.save_pretrained(Config.OUTPUT_DIR)
    
    # Save LoRA config
    model.save_pretrained(os.path.join(Config.OUTPUT_DIR, "lora_adapter"))
    
    print("\nTraining complete!")
    print(f"Model saved to: {Config.OUTPUT_DIR}")
    
    # Finish wandb
    wandb.finish()


if __name__ == "__main__":
    # Set environment variables for better error messages
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    main()
