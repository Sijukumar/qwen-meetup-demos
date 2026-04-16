#!/usr/bin/env python3
"""
Data Preprocessing Script for Qwen Fine-Tuning

Converts raw JSONL data into tokenized format for training.

Usage:
    python preprocess_data.py --input train_data.jsonl --output processed_dataset
"""

import json
import argparse
from pathlib import Path
from datasets import Dataset

# Try ModelScope first, fallback to HuggingFace
try:
    from modelscope import AutoTokenizer
    print("Using ModelScope for tokenizer")
except ImportError:
    from transformers import AutoTokenizer
    print("Using HuggingFace Transformers for tokenizer")


def load_jsonl_data(file_path):
    """Load data from JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON on line {line_num}: {e}")
    return data


def format_conversation(example, tokenizer):
    """Format conversation using Qwen chat template."""
    messages = example.get('messages', [])
    
    if not messages:
        return {'text': ''}
    
    # Apply Qwen chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    
    return {'text': text}


def tokenize_function(examples, tokenizer, max_length):
    """Tokenize text data."""
    # Tokenize with padding and truncation
    result = tokenizer(
        examples['text'],
        truncation=True,
        max_length=max_length,
        padding='max_length',
        return_tensors=None  # Return lists, not tensors
    )
    
    # Create labels (same as input_ids for causal LM)
    result['labels'] = result['input_ids'].copy()
    
    return result


def preprocess_data(
    input_file,
    output_dir,
    model_name="qwen/Qwen-7B-Chat",
    max_length=2048,
    test_split=0.1
):
    """
    Preprocess data for fine-tuning.
    
    Args:
        input_file: Path to input JSONL file
        output_dir: Directory to save processed dataset
        model_name: Model name for tokenizer
        max_length: Maximum sequence length
        test_split: Fraction of data to use for validation
    """
    print("=" * 60)
    print("Data Preprocessing for Qwen Fine-Tuning")
    print("=" * 60)
    
    # Load tokenizer
    print(f"\nLoading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        pad_token='<|extra_0|>',
        eos_token='<|endoftext|>'
    )
    
    # Load raw data
    print(f"Loading data from: {input_file}")
    raw_data = load_jsonl_data(input_file)
    print(f"Loaded {len(raw_data)} examples")
    
    # Create dataset
    print("\nCreating dataset...")
    dataset = Dataset.from_list(raw_data)
    
    # Format conversations
    print("Formatting conversations...")
    dataset = dataset.map(
        lambda x: format_conversation(x, tokenizer),
        remove_columns=dataset.column_names
    )
    
    # Filter empty texts
    dataset = dataset.filter(lambda x: len(x['text']) > 0)
    print(f"After filtering: {len(dataset)} examples")
    
    # Tokenize
    print(f"Tokenizing (max_length={max_length})...")
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer, max_length),
        batched=True,
        batch_size=1000,
        remove_columns=['text']
    )
    
    # Split into train/validation
    if test_split > 0:
        print(f"Splitting dataset (test_split={test_split})...")
        split_dataset = tokenized_dataset.train_test_split(
            test_size=test_split,
            shuffle=True,
            seed=42
        )
        
        train_dataset = split_dataset['train']
        val_dataset = split_dataset['test']
        
        print(f"Train: {len(train_dataset)} examples")
        print(f"Validation: {len(val_dataset)} examples")
        
        # Save both
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        train_dataset.save_to_disk(output_path / 'train')
        val_dataset.save_to_disk(output_path / 'validation')
        
        print(f"\nSaved to:")
        print(f"  Train: {output_path / 'train'}")
        print(f"  Validation: {output_path / 'validation'}")
    else:
        # Save entire dataset as train
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        tokenized_dataset.save_to_disk(output_path)
        print(f"\nSaved to: {output_path}")
    
    # Print sample
    print("\n" + "=" * 60)
    print("Sample processed example:")
    print("=" * 60)
    sample = tokenized_dataset[0]
    print(f"Input IDs length: {len(sample['input_ids'])}")
    print(f"Attention mask length: {len(sample['attention_mask'])}")
    print(f"Labels length: {len(sample['labels'])}")
    
    # Decode sample
    decoded = tokenizer.decode(sample['input_ids'], skip_special_tokens=False)
    print(f"\nDecoded text (first 200 chars):")
    print(decoded[:200] + "...")
    
    print("\nPreprocessing complete!")


def create_sample_data(output_file="train_data.jsonl", num_samples=100):
    """Create sample training data for testing."""
    
    sample_data = [
        {
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": "What is machine learning?"},
                {"role": "assistant", "content": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It uses algorithms and statistical models to analyze data, identify patterns, and make predictions or decisions."}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Explain cloud computing"},
                {"role": "assistant", "content": "Cloud computing is the delivery of computing services—including servers, storage, databases, networking, software, and analytics—over the internet (\"the cloud\"). It offers faster innovation, flexible resources, and economies of scale."}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Write a Python hello world program"},
                {"role": "assistant", "content": "Here's a simple Python hello world program:\n\n```python\nprint(\"Hello, World!\")\n```\n\nWhen you run this, it will output:\n```\nHello, World!\n```"}
            ]
        }
    ]
    
    # Repeat to create more samples
    with open(output_file, 'w', encoding='utf-8') as f:
        for i in range(num_samples):
            sample = sample_data[i % len(sample_data)]
            # Add variation
            sample_copy = json.loads(json.dumps(sample))
            if 'variation' in sample_copy['messages'][0]:
                sample_copy['messages'][0]['content'] += f" (Sample {i})"
            f.write(json.dumps(sample_copy, ensure_ascii=False) + '\n')
    
    print(f"Created {num_samples} sample records in {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess data for Qwen fine-tuning')
    parser.add_argument('--input', type=str, default='train_data.jsonl',
                        help='Input JSONL file')
    parser.add_argument('--output', type=str, default='processed_dataset',
                        help='Output directory for processed dataset')
    parser.add_argument('--model', type=str, default='qwen/Qwen-7B-Chat',
                        help='Model name for tokenizer')
    parser.add_argument('--max-length', type=int, default=2048,
                        help='Maximum sequence length')
    parser.add_argument('--test-split', type=float, default=0.1,
                        help='Validation split ratio (0-1)')
    parser.add_argument('--create-sample', action='store_true',
                        help='Create sample data file')
    
    args = parser.parse_args()
    
    if args.create_sample:
        create_sample_data(args.input)
    else:
        preprocess_data(
            input_file=args.input,
            output_dir=args.output,
            model_name=args.model,
            max_length=args.max_length,
            test_split=args.test_split
        )
