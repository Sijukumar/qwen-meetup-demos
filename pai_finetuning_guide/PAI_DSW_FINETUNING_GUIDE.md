# Alibaba Cloud PAI-DSW Fine-Tuning Guide

## Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Environment Setup](#environment-setup)
4. [Data Preparation](#data-preparation)
5. [Fine-Tuning Code](#fine-tuning-code)
6. [Training Configuration](#training-configuration)
7. [Running Training](#running-training)
8. [Model Evaluation](#model-evaluation)
9. [Model Deployment](#model-deployment)
10. [Troubleshooting](#troubleshooting)

## Overview

This guide covers fine-tuning Qwen models using Alibaba Cloud PAI-DSW (Data Science Workshop) with LoRA (Low-Rank Adaptation) for efficient training.

### Why PAI-DSW?
- **Jupyter Environment**: Interactive development
- **Pre-installed Frameworks**: PyTorch, Transformers, DeepSpeed
- **GPU Access**: V100, A100, A10 GPUs
- **Integrated Storage**: Direct OSS access
- **Easy Debugging**: Step-through training code

## Prerequisites

### 1. Alibaba Cloud Account
- Valid Alibaba Cloud account
- PAI service activated
- OSS bucket created

### 2. PAI-DSW Instance
- **Instance Type**: GPU instance (V100/A100 recommended)
- **Image**: PyTorch 2.0 + Python 3.9
- **Storage**: 100GB+ (for model checkpoints)

### 3. Quota & Permissions
- PAI-DSW quota available
- OSS read/write permissions
- RAM role for PAI access

## Environment Setup

### Step 1: Create PAI-DSW Instance

```bash
# Via Alibaba Cloud Console
1. Navigate to PAI Console → DSW
2. Click "Create Instance"
3. Select:
   - Region: Singapore (or nearest)
   - Instance Type: ecs.gn6v-c8g1.2xlarge (V100) or better
   - Image: pytorch:2.0.1-gpu-py39-cu118
   - Storage: 100GB
4. Click "Create"
```

### Step 2: Install Dependencies

Create `requirements.txt`:
```
torch>=2.0.0
transformers>=4.35.0
peft>=0.6.0
datasets>=2.14.0
accelerate>=0.24.0
deepspeed>=0.12.0
wandb>=0.15.0
modelscope>=1.9.0
tensorboard>=2.14.0
```

Install in DSW terminal:
```bash
pip install -r requirements.txt
```

**Note on Model Download Source:**
- **ModelScope** (default): Faster for China/Asia regions, integrated with Alibaba Cloud
- **HuggingFace**: Alternative for other regions
- Code auto-detects and uses best available source

### Step 3: Configure OSS Access

```bash
# Install OSS utilities
pip install oss2

# Configure credentials (in DSW, use instance role instead)
# Or use environment variables
export OSS_ACCESS_KEY_ID=your_key
export OSS_ACCESS_KEY_SECRET=your_secret
export OSS_ENDPOINT=oss-ap-southeast-1.aliyuncs.com
export OSS_BUCKET=your-bucket-name
```

## Data Preparation

### Data Format

Create training data in JSONL format:

```json
{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "What is AI?"}, {"role": "assistant", "content": "AI stands for Artificial Intelligence..."}]}
{"messages": [{"role": "user", "content": "Explain machine learning"}, {"role": "assistant", "content": "Machine learning is a subset of AI..."}]}
```

### Data Preprocessing Script

```python
# preprocess_data.py
import json
from datasets import Dataset
from transformers import AutoTokenizer

def load_and_preprocess_data(data_path, tokenizer, max_length=2048):
    """Load and preprocess training data."""
    
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    
    def format_conversation(example):
        """Format conversation for Qwen."""
        messages = example['messages']
        
        # Format using Qwen chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        return {'text': text}
    
    # Create dataset
    dataset = Dataset.from_list(data)
    dataset = dataset.map(format_conversation)
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=max_length,
            padding='max_length'
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    return tokenized_dataset

# Usage
if __name__ == "__main__":
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(
        "qwen/Qwen-7B-Chat",
        trust_remote_code=True
    )
    
    dataset = load_and_preprocess_data(
        'train_data.jsonl',
        tokenizer
    )
    
    dataset.save_to_disk('processed_dataset')
    print(f"Processed {len(dataset)} examples")
```

## Fine-Tuning Code

### Main Training Script

```python
# train_qwen_lora.py
import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
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

# Configuration
class Config:
    # Model
    MODEL_NAME = "qwen/Qwen-7B-Chat"
    
    # LoRA
    LORA_R = 64
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.1
    TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    # Training
    OUTPUT_DIR = "./qwen-lora-finetuned"
    NUM_EPOCHS = 3
    BATCH_SIZE = 4
    GRADIENT_ACCUMULATION_STEPS = 4
    LEARNING_RATE = 2e-4
    WARMUP_STEPS = 100
    SAVE_STEPS = 500
    LOGGING_STEPS = 10
    MAX_LENGTH = 2048
    
    # Optimization
    FP16 = True
    GRADIENT_CHECKPOINTING = True
    OPTIM = "paged_adamw_32bit"


def setup_model_and_tokenizer():
    """Load model and tokenizer."""
    
    print(f"Loading model: {Config.MODEL_NAME}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        Config.MODEL_NAME,
        trust_remote_code=True,
        pad_token='<|extra_0|>',
        eos_token='<|endoftext|>'
    )
    
    # Load model in 4-bit for memory efficiency
    model = AutoModelForCausalLM.from_pretrained(
        Config.MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Prepare for training
    model = prepare_model_for_kbit_training(model)
    
    return model, tokenizer


def setup_lora(model):
    """Configure LoRA."""
    
    lora_config = LoraConfig(
        r=Config.LORA_R,
        lora_alpha=Config.LORA_ALPHA,
        target_modules=Config.TARGET_MODULES,
        lora_dropout=Config.LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model


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
        fp16=Config.FP16,
        gradient_checkpointing=Config.GRADIENT_CHECKPOINTING,
        report_to="wandb",
        run_name="qwen-7b-lora-finetune",
        remove_unused_columns=False,
    )


def main():
    """Main training function."""
    
    # Initialize wandb
    wandb.init(project="qwen-finetuning", name="qwen-7b-lora")
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer()
    
    # Setup LoRA
    model = setup_lora(model)
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_from_disk('processed_dataset')
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        pad_to_multiple_of=8,
        return_tensors="pt",
        padding=True
    )
    
    # Training arguments
    training_args = setup_training_args()
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save model
    print(f"Saving model to {Config.OUTPUT_DIR}")
    trainer.save_model(Config.OUTPUT_DIR)
    tokenizer.save_pretrained(Config.OUTPUT_DIR)
    
    print("Training complete!")


if __name__ == "__main__":
    main()
```

### DeepSpeed Configuration (Optional)

Create `ds_config.json` for distributed training:

```json
{
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 1000,
    "initial_scale_power": 16,
    "hysteresis": 2,
    "min_loss_scale": 1
  },
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 2e-4,
      "betas": [0.9, 0.999],
      "eps": 1e-8,
      "weight_decay": 0.01
    }
  },
  "scheduler": {
    "type": "WarmupLR",
    "params": {
      "warmup_min_lr": 0,
      "warmup_max_lr": 2e-4,
      "warmup_num_steps": 100
    }
  },
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "allgather_partitions": true,
    "allgather_bucket_size": 2e8,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 2e8,
    "contiguous_gradients": true
  },
  "train_batch_size": 16,
  "train_micro_batch_size_per_gpu": 4,
  "gradient_accumulation_steps": 4,
  "wall_clock_breakdown": false
}
```

## Model Download Configuration

The training script automatically handles model downloads from the best available source:

### Option 1: ModelScope (Recommended for PAI-DSW in Asia)
```python
MODEL_NAME = "qwen/Qwen-7B-Chat"  # Uses ModelScope by default
```

**Cache Location:** `~/.cache/modelscope/hub/qwen/Qwen-7B-Chat/`

**Benefits:**
- Faster download in China/Asia regions (~14GB for 7B model)
- Better integration with Alibaba Cloud
- No authentication token required

### Option 2: HuggingFace
```python
MODEL_NAME = "Qwen/Qwen-7B-Chat"  # HF format with capital Q
```

**Cache Location:** `~/.cache/huggingface/hub/models--Qwen--Qwen-7B-Chat/`

**Setup:**
```bash
huggingface-cli login  # Enter your HF token
```

### Pre-download Model (Optional)

```python
# download_model.py
from modelscope import AutoModelForCausalLM, AutoTokenizer

model_name = "qwen/Qwen-7B-Chat"
print("Downloading model...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
print("Ready for training!")
```

### Monitor Download
```bash
watch -n 2 du -sh ~/.cache/modelscope/hub/qwen/
```

## Training Configuration

### Resource Requirements

| Model | GPU Memory | Recommended Instance | Training Time (1 epoch) |
|-------|-----------|---------------------|------------------------|
| Qwen-7B | 16GB | V100 (16GB) | ~2 hours |
| Qwen-14B | 24GB | A10 (24GB) | ~4 hours |
| Qwen-72B | 80GB | A100 (80GB) | ~12 hours |

### Hyperparameter Tuning

```python
# Grid search configuration
configs = [
    {"lora_r": 8, "lr": 1e-4, "batch_size": 4},
    {"lora_r": 16, "lr": 2e-4, "batch_size": 4},
    {"lora_r": 64, "lr": 2e-4, "batch_size": 2},
]

for config in configs:
    # Run training with each config
    pass
```

## Running Training

### Option 1: Direct Python

```bash
# In PAI-DSW terminal
python train_qwen_lora.py
```

### Option 2: With DeepSpeed

```bash
deepspeed --num_gpus=1 train_qwen_lora.py --deepspeed ds_config.json
```

### Option 3: Distributed Training

```bash
# Multi-GPU
torchrun --nproc_per_node=2 train_qwen_lora.py
```

## Model Evaluation

```python
# evaluate.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def evaluate_model(base_model_path, lora_path, test_prompts):
    """Evaluate fine-tuned model."""
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load LoRA weights
    model = PeftModel.from_pretrained(model, lora_path)
    model = model.merge_and_unload()  # Merge for inference
    
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True
    )
    
    results = []
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        results.append({"prompt": prompt, "response": response})
    
    return results

# Test prompts
test_prompts = [
    "Explain quantum computing in simple terms",
    "Write a Python function to calculate fibonacci",
    "What are the benefits of cloud computing?"
]

results = evaluate_model(
    "qwen/Qwen-7B-Chat",
    "./qwen-lora-finetuned",
    test_prompts
)

for r in results:
    print(f"Prompt: {r['prompt']}")
    print(f"Response: {r['response']}\n")
```

## Model Deployment

### Upload to OSS

```python
# upload_to_oss.py
import oss2
import os

def upload_directory_to_oss(local_dir, oss_bucket, oss_prefix):
    """Upload model to OSS."""
    
    auth = oss2.Auth(
        os.getenv('OSS_ACCESS_KEY_ID'),
        os.getenv('OSS_ACCESS_KEY_SECRET')
    )
    bucket = oss2.Bucket(
        auth,
        os.getenv('OSS_ENDPOINT'),
        oss_bucket
    )
    
    for root, dirs, files in os.walk(local_dir):
        for file in files:
            local_path = os.path.join(root, file)
            oss_path = os.path.join(
                oss_prefix,
                os.path.relpath(local_path, local_dir)
            )
            
            bucket.put_object_from_file(oss_path, local_path)
            print(f"Uploaded: {local_path} -> {oss_path}")

# Usage
upload_directory_to_oss(
    './qwen-lora-finetuned',
    'your-bucket',
    'models/qwen-finetuned/'
)
```

### Deploy to PAI-EAS

```bash
# Via Alibaba Cloud Console
1. Navigate to PAI → EAS
2. Click "Deploy Service"
3. Select "Custom Runtime"
4. Upload model from OSS
5. Configure instance: GPU instance
6. Deploy
```

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| OOM Error | Reduce batch size, enable gradient checkpointing, use DeepSpeed ZeRO-3 |
| Slow Training | Use DeepSpeed, increase batch size, use mixed precision (FP16) |
| NaN Loss | Reduce learning rate, check data quality, use gradient clipping |
| OSS Upload Fail | Check credentials, increase timeout, use multipart upload |

### Monitoring Training

```python
# Add to training script
from transformers import TrainerCallback

class LoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            print(f"Step {state.global_step}: {logs}")

trainer.add_callback(LoggingCallback())
```

### Check GPU Usage

```bash
# In DSW terminal
nvidia-smi
watch -n 1 nvidia-smi
```

## Best Practices

1. **Start Small**: Test with small dataset first
2. **Version Control**: Track experiments with wandb/mlflow
3. **Checkpointing**: Save checkpoints frequently
4. **Validation**: Use validation set to prevent overfitting
5. **Quantization**: Use 4-bit/8-bit for large models
6. **Cleanup**: Stop DSW instance when not training to save costs

## References

- [PAI Documentation](https://www.alibabacloud.com/help/en/pai)
- [Qwen GitHub](https://github.com/QwenLM/Qwen)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [DeepSpeed Documentation](https://www.deepspeed.ai/)
