from transformers import BartForConditionalGeneration, BartTokenizer, Trainer, TrainingArguments, pipeline
from datasets import Dataset
import json
from datetime import datetime, timedelta
import torch

def parse_timestamp(timestamp):
    try:
        # Try the first format
        return datetime.strptime(timestamp, "%d/%m/%Y, %I:%M:%S %p")
    except ValueError:
        try:
            # Try the second format (without seconds)
            return datetime.strptime(timestamp, "%d/%m/%Y, %I:%M %p")
        except ValueError:
            try:
                # Try the third format (without year)
                return datetime.strptime(timestamp, "%d/%m, %I:%M")
            except ValueError:
                # If all formats fail, return current time
                print(f"Warning: Could not parse timestamp: {timestamp}")
                return datetime.now()

def prepare_data(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Get messages from the data structure
    messages = data["messages"]
    
    # Convert timestamps and sort
    for msg in messages:
        msg["dt"] = parse_timestamp(msg["timestamp"])
    messages = sorted(messages, key=lambda x: x["dt"])
    
    # Group messages
    grouped = []
    current_group = []
    prev_time = None
    time_gap = timedelta(minutes=15)
    
    for msg in messages:
        if not prev_time or (msg["dt"] - prev_time) <= time_gap:
            current_group.append(msg)
        else:
            grouped.append(current_group)
            current_group = [msg]
        prev_time = msg["dt"]
    
    if current_group:
        grouped.append(current_group)
    
    # Prepare training data
    train_data = []
    for group in grouped:
        convo = []
        for msg in group:
            if "context" in msg:
                convo.extend(msg["context"])
            line = f"[{msg['timestamp']}] {msg['speaker']}: {msg['text']}"
            convo.append(line)
        
        # Use all summaries in the group as targets
        if group:
            summaries = [msg["summary"] for msg in group if "summary" in msg]
            if summaries:
                train_data.append({
                    "text": "\n".join(convo),
                    "summary": " ".join(summaries)  # Combine all summaries
                })
    
    return train_data

def test_model(model_path):
    # Load the fine-tuned model
    summarizer = pipeline("summarization", model=model_path)
    
    # Test with a sample conversation
    test_convo = """[05/03/2025, 2:18:17 PM] Myriam: hi kings
[05/03/2025, 2:18:22 PM] Myriam: balasho les projets"""
    
    # Generate summary
    summary = summarizer(test_convo, max_length=80, min_length=20, do_sample=False)
    print("\nTest Conversation:")
    print(test_convo)
    print("\nGenerated Summary:")
    print(summary[0]["summary_text"])

def main():
    # Load model and tokenizer
    model_name = "facebook/bart-large-cnn"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    
    # Prepare data
    train_data = prepare_data("./data/structured_corpus.json")
    
    # Convert to Dataset
    dataset = Dataset.from_list(train_data)
    
    def tokenize_function(examples):
        inputs = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=1024)
        targets = tokenizer(examples["summary"], padding="max_length", truncation=True, max_length=128)
        
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": targets["input_ids"]
        }
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
        learning_rate=5e-5,
        warmup_steps=500,
        weight_decay=0.01,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )
    
    # Train
    trainer.train()
    
    # Save model
    model.save_pretrained("./fine_tuned_bart")
    tokenizer.save_pretrained("./fine_tuned_bart")
    
    # Test the model
    print("\nTesting the fine-tuned model...")
    test_model("./fine_tuned_bart")

if __name__ == "__main__":
    main() 