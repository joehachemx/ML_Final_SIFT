import json
import torch
import numpy as np
import os
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    Trainer, 
    TrainingArguments,
    DataCollatorForSeq2Seq,
    DataCollatorWithPadding
)
from datasets import Dataset
import argparse
from sklearn.model_selection import train_test_split
import gc
from tqdm.auto import tqdm

# Check if GPU is available (Colab-specific)
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("GPU not available, using CPU")

def cleanup_memory():
    """Clean up memory to avoid OOM errors"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def load_data(data_file):
    """Load the processed JSON data"""
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data['messages'])} messages from {data_file}")
    return data['messages']

def prepare_classification_dataset(messages):
    """Prepare dataset for importance classification"""
    # Extract features and labels
    texts = []
    importance_labels = []
    
    for message in tqdm(messages, desc="Preparing classification data"):
        # Combine message with context for better understanding
        context = " ".join(message['context']) if message['context'] else ""
        full_text = f"[{message['speaker']}] [{message['timestamp']}] {context} [SEP] {message['text']}"
        
        texts.append(full_text)
        importance_labels.append(message['importance'])
    
    # Create train and validation splits
    texts_train, texts_val, labels_train, labels_val = train_test_split(
        texts, importance_labels, test_size=0.2, random_state=42
    )
    
    print(f"Classification dataset: {len(texts_train)} training examples, {len(texts_val)} validation examples")
    
    # Create datasets
    train_dataset = Dataset.from_dict({
        'text': texts_train,
        'label': labels_train
    })
    
    val_dataset = Dataset.from_dict({
        'text': texts_val,
        'label': labels_val
    })
    
    return train_dataset, val_dataset

def prepare_summarization_dataset(messages):
    """Prepare dataset for message summarization"""
    # Extract features and labels
    texts = []
    summaries = []
    
    for message in tqdm(messages, desc="Preparing summarization data"):
        if message['summary']:  # Only use messages with summaries
            # Combine message with context for better understanding
            context = " ".join(message['context']) if message['context'] else ""
            full_text = f"[{message['speaker']}] [{message['timestamp']}] {context} [SEP] {message['text']}"
            
            texts.append(full_text)
            summaries.append(message['summary'])
    
    # Create train and validation splits
    texts_train, texts_val, summaries_train, summaries_val = train_test_split(
        texts, summaries, test_size=0.2, random_state=42
    )
    
    print(f"Summarization dataset: {len(texts_train)} training examples, {len(texts_val)} validation examples")
    
    # Create datasets
    train_dataset = Dataset.from_dict({
        'text': texts_train,
        'summary': summaries_train
    })
    
    val_dataset = Dataset.from_dict({
        'text': texts_val,
        'summary': summaries_val
    })
    
    return train_dataset, val_dataset

def setup_classification_model():
    """Setup model for importance classification"""
    # Use XLM-RoBERTa for multilingual support
    model_name = "xlm-roberta-base"  # Better model for GPU
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=2  # Binary classification: important or not
    ).to(device)
    
    def tokenize_function(examples):
        return tokenizer(
            examples['text'], 
            padding='max_length', 
            truncation=True, 
            max_length=256  # Can use larger inputs on GPU
        )
    
    return model, tokenizer, tokenize_function

def setup_summarization_model():
    """Setup model for message summarization"""
    # Use mT5 for multilingual summarization
    model_name = "google/mt5-small"  # Better model for GPU
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    
    def tokenize_function(examples):
        inputs = tokenizer(
            examples['text'], 
            padding='max_length', 
            truncation=True, 
            max_length=256  # Can use larger inputs on GPU
        )
        
        # Tokenize summaries
        with tokenizer.as_target_tokenizer():
            outputs = tokenizer(
                examples['summary'], 
                padding='max_length', 
                truncation=True, 
                max_length=64  # Can use larger outputs on GPU
            )
        
        inputs['labels'] = outputs['input_ids']
        return inputs
    
    return model, tokenizer, tokenize_function

def train_classification_model(data_file, output_dir):
    """Train importance classification model"""
    print("Loading data...")
    messages = load_data(data_file)
    cleanup_memory()
    
    print("Preparing classification dataset...")
    train_dataset, val_dataset = prepare_classification_dataset(messages)
    cleanup_memory()
    
    print("Setting up classification model...")
    model, tokenizer, tokenize_function = setup_classification_model()
    cleanup_memory()
    
    # Tokenize datasets
    print("Tokenizing datasets...")
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    cleanup_memory()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up training arguments - optimized for GPU
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=5e-5,
        per_device_train_batch_size=16,  # Increased for GPU
        per_device_eval_batch_size=16,   # Increased for GPU
        gradient_accumulation_steps=2,   # Reduced for GPU
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy="steps",     # Explicitly set to steps
        save_strategy="steps",           # Explicitly set to steps
        eval_steps=100,                  # Number of steps between evaluations
        save_steps=100,                  # Number of steps between saves
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        warmup_ratio=0.1,
        fp16=True,  # Mixed precision training
        gradient_checkpointing=True,  # Memory optimization
        report_to="tensorboard",
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
    )
    
    # Train the model
    print("Training classification model...")
    trainer.train()
    cleanup_memory()
    
    # Save the model
    print(f"Saving model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    cleanup_memory()

def train_summarization_model(data_file, output_dir):
    """Train message summarization model"""
    print("Loading data...")
    messages = load_data(data_file)
    cleanup_memory()
    
    print("Preparing summarization dataset...")
    train_dataset, val_dataset = prepare_summarization_dataset(messages)
    cleanup_memory()
    
    print("Setting up summarization model...")
    model, tokenizer, tokenize_function = setup_summarization_model()
    cleanup_memory()
    
    # Tokenize datasets
    print("Tokenizing datasets...")
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    cleanup_memory()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up training arguments - optimized for GPU
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=5e-5,
        per_device_train_batch_size=8,  # Increased for GPU
        per_device_eval_batch_size=8,   # Increased for GPU
        gradient_accumulation_steps=2,  # Reduced for GPU
        num_train_epochs=3,
        weight_decay=0.01,
        eval_steps=100,                  # Number of steps between evaluations
        save_steps=100,                  # Number of steps between saves
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        warmup_ratio=0.1,
        fp16=True,  # Mixed precision training
        gradient_checkpointing=True,  # Memory optimization
        report_to="tensorboard",
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model
        )
    )
    
    # Train the model
    print("Training summarization model...")
    trainer.train()
    cleanup_memory()
    
    # Save the model
    print(f"Saving model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    cleanup_memory()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train models for WhatsApp message processing.')
    parser.add_argument('--data', default='data/structured_corpus.json', help='Structured data file')
    parser.add_argument('--task', choices=['classification', 'summarization', 'both'], default='both',
                        help='Which task to train for')
    parser.add_argument('--class_output', default='models/importance_classifier', 
                        help='Output directory for classification model')
    parser.add_argument('--summ_output', default='models/message_summarizer', 
                        help='Output directory for summarization model')
    
    args = parser.parse_args()
    
    # Print GPU info
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    if args.task in ['classification', 'both']:
        train_classification_model(args.data, args.class_output)
    
    if args.task in ['summarization', 'both']:
        train_summarization_model(args.data, args.summ_output)