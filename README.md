# WhatsApp Message Processor

This project processes WhatsApp messages in multilingual contexts (English, French, and Franco-Arabic) to:
1. Classify message importance
2. Generate concise summaries
3. Handle transliteration variations

Perfect for filtering notifications and focusing on what matters.

## Project Structure

```
├── data/
│   ├── raw_chats/            # Original WhatsApp exports
│   ├── processed_chats/      # Cleaned chat data
│   ├── only_text_chats/      # Text-only processed chats
│   ├── clusters/             # Clustered message data
│   └── structured_corpus.json # Final processed dataset
├── models/                   # Trained models
├── tests_ignore/            # Test files
├── transliteration/         # Transliteration utilities
├── data_cleaning.ipynb      # Data cleaning pipeline
├── data_processor.ipynb     # Data processing pipeline
├── dataset_statistics.ipynb # Data analysis
├── summarizer_demo.ipynb    # Summarization demo
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Process raw chat data:
   - Run `data_cleaning.ipynb` to clean raw WhatsApp exports
   - Run `data_processor.ipynb` to process cleaned data
   - Run `dataset_statistics.ipynb` to analyze the dataset

## Usage

For a quick demo of the summarization capabilities:
```
jupyter notebook summarizer_demo.ipynb
```

## Data Processing Pipeline

1. **Raw Data**: WhatsApp exports in `data/raw_chats/`
2. **Cleaning**: Remove media, system messages, and standardize formatting
3. **Processing**: Extract text, handle multilingual content
4. **Structuring**: Create `structured_corpus.json` with labeled data

## Model Architecture

The project uses:
1. **Importance Classifier**: Fine-tuned BART model for binary classification
2. **Message Summarizer**: Fine-tuned BART model for multilingual summarization
3. **Transliteration Handler**: Custom utilities for Franco-Arabic variations

## Notes on Franco-Arabic

Franco-Arabic has high spelling variation. The same word can be written in multiple ways:
- "kifak" / "kif" / "kifik" (how are you)
- "shu" / "chu" / "shou" (what)

The transliteration module handles these variations through pattern matching and contextual analysis. 