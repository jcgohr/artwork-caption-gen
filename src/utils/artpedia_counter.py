import json

"""
Count attributes of the Artpedia dataset
"""

def merge_splits(test, train, val):
    with open(test, 'r', encoding='utf-8') as f1:
        test_split = json.load(f1)
    with open(train, 'r', encoding='utf-8') as f2:
        train_split = json.load(f2)
    with open(val, 'r', encoding='utf-8') as f3:
        val_split = json.load(f3)    
    return {**test_split, **train_split, **val_split}

def count_words(text):
    return len(text.split())

def process_counts(merged_dataset):
    caption_word_counts = {}
    
    for key, value in merged_dataset.items():
        total_words = sum(count_words(sentence) for sentence in value["visual_sentences"])
        caption_word_counts[key] = total_words

    min_id = min(caption_word_counts, key=caption_word_counts.get)
    max_id = max(caption_word_counts, key=caption_word_counts.get)
    min_words = caption_word_counts[min_id]
    max_words = caption_word_counts[max_id]
    avg_words = sum(caption_word_counts.values()) / len(caption_word_counts)
    
    print(f"Minimum caption length: {min_words} words (ID: {min_id})")
    print(f"Maximum caption length: {max_words} words (ID: {max_id})")
    print(f"Average caption length: {avg_words:.2f} words")

if __name__ == '__main__':
    test_path = ""
    train_path = ""
    val_path = ""
    merged_dataset = merge_splits(test_path, train_path, val_path)
    process_counts(merged_dataset)

