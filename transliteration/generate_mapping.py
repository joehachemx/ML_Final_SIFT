import json
import asyncio
from googletrans import Translator
from langdetect import detect
import time

def is_french(text):
    try:
        return detect(text) == 'fr'
    except:
        return False

def is_english(text):
    try:
        return detect(text) == 'en'
    except:
        return False

async def translate_french(word, translator):
    try:
        result = await translator.translate(word, src='fr', dest='en')
        return result.text.lower()
    except:
        return word

async def create_translation_map(clusters_file='../data/clusters/normal_clusters.json', 
                               output_file='../data/clusters/translation_map.json'):
    print("Loading clusters file...")
    with open(clusters_file, 'r', encoding='utf-8') as f:
        clusters = json.load(f)
    
    clusters = dict(list(clusters.items()))
    print(f"Processing first {len(clusters)} clusters")
    
    translator = Translator()
    translation_map = {}
    
    total_words = len(clusters)
    processed = 0
    
    for cluster_name, variations in clusters.items():
        processed += 1
        print(f"\nProcessing word {processed}/{total_words} ({processed/total_words*100:.1f}% complete)")
        print(f"Word: {cluster_name}")
        print(f"Variations: {variations}")
        
        # Check if word is French
        if is_french(cluster_name):
            print("Detected as French, translating automatically...")
            translated = await translate_french(cluster_name, translator)
            print(f"Translated to: {translated}")
        # Check if word is English
        elif is_english(cluster_name):
            print("Detected as English, keeping original")
            translated = cluster_name
        # Everything else is considered Franco-Arabic
        else:
            print("Detected as Franco-Arabic")
            print("Please provide English translation (press Enter to skip):")
            translated = input().strip()
            if not translated:
                translated = cluster_name
        
        # Only store the main word and its translation
        translation_map[cluster_name] = translated
        print(f"Final mapping: {cluster_name} -> {translated}")
    
    print("\nSaving translation map to file...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(translation_map, f, ensure_ascii=False, indent=2)
    print(f"Saved to {output_file}")
    
    return translation_map

# Run the function
if __name__ == "__main__":
    print("Starting translation process...")
    start_time = time.time()
    asyncio.run(create_translation_map())
    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")