import json
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM
)
from tqdm.auto import tqdm
import os

class WhatsAppMessageProcessor:    
    def __init__(self, classification_model_path, summarization_model_path):
        # Check GPU availability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load classification model
        print("Loading classification model...")
        self.classification_tokenizer = AutoTokenizer.from_pretrained(classification_model_path)
        self.classification_model = AutoModelForSequenceClassification.from_pretrained(classification_model_path)
        self.classification_model.to(self.device)
        self.classification_model.eval()  # Set to evaluation mode
        
        # Load summarization model
        print("Loading summarization model...")
        self.summarization_tokenizer = AutoTokenizer.from_pretrained(summarization_model_path)
        self.summarization_model = AutoModelForSeq2SeqLM.from_pretrained(summarization_model_path)
        self.summarization_model.to(self.device)
        self.summarization_model.eval()  # Set to evaluation mode
        
        # Enable mixed precision for faster inference
        if torch.cuda.is_available():
            self.classification_model = self.classification_model.half()
            self.summarization_model = self.summarization_model.half()
        
    def classify_importance(self, message, context=None):
        """Classify if message is important"""
        # Prepare input text with context if available
        if context:
            if isinstance(context, list):
                context = " ".join(context)
            input_text = f"[{context}] [SEP] {message}"
        else:
            input_text = message
        
        # Tokenize
        inputs = self.classification_tokenizer(
            input_text,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        ).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.classification_model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            importance_score = predictions[0, 1].item()  # Probability of being important
        
        # Classify as important if score > 0.5
        is_important = importance_score > 0.5
        
        return {
            "is_important": bool(is_important),
            "importance_score": importance_score
        }
    
    def summarize_message(self, message, context=None):
        """Generate summary for message"""
        # Prepare input text with context if available
        if context:
            if isinstance(context, list):
                context = " ".join(context)
            input_text = f"[{context}] [SEP] {message}"
        else:
            input_text = message
        
        # Tokenize
        inputs = self.summarization_tokenizer(
            input_text,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate summary
        with torch.no_grad():
            output_ids = self.summarization_model.generate(
                inputs["input_ids"],
                max_length=64,
                num_beams=4,
                early_stopping=True,
                do_sample=True,
                top_k=50,
                top_p=0.95
            )
        
        # Decode summary
        summary = self.summarization_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        return summary
    
    def process_message(self, message, context=None):
        """Process message - classify importance and generate summary"""
        # Classify importance
        importance_result = self.classify_importance(message, context)
        
        # Generate summary
        summary = self.summarize_message(message, context)
        
        return {
            "message": message,
            "context": context,
            "is_important": importance_result["is_important"],
            "importance_score": importance_result["importance_score"],
            "summary": summary
        }

def predict(message=None, input_file=None, output_file='data/processed_results.json', context=None):
    """
    Process WhatsApp messages using trained models.
    
    Args:
        message (str, optional): Single message to process
        input_file (str, optional): Path to file containing messages (one per line)
        output_file (str): Path to save results (default: 'data/processed_results.json')
        context (list, optional): List of context messages for single message processing
    
    Returns:
        dict or list: Processed results
    """
    # Initialize processor
    processor = WhatsAppMessageProcessor(
        classification_model_path='models/importance_classifier',
        summarization_model_path='models/message_summarizer'
    )
    
    # Process single message
    if message:
        result = processor.process_message(message, context)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return result
    
    # Process messages from file
    elif input_file:
        print(f"Processing messages from {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            messages = [line.strip() for line in f if line.strip()]
        
        results = []
        context = []
        
        # Process messages with progress bar
        for message in tqdm(messages, desc="Processing messages"):
            result = processor.process_message(message, context)
            results.append(result)
            
            # Update context for next message
            context.append(message)
            if len(context) > 3:  # Keep only last 3 messages as context
                context.pop(0)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save results
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
        print(f"Processed {len(results)} messages. Results saved to {output_file}")
        return results
    
    else:
        raise ValueError("Either message or input_file must be provided")

# Example usage:
if __name__ == "__main__":
    # Process a single message
    result = predict(
        message="Your message here",
        context=["Previous message 1", "Previous message 2"]
    )
    
    # Process messages from a file
    results = predict(
        input_file="data/test_messages.txt",
        output_file="data/processed_results.json"
    ) 