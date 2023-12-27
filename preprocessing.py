from nltk.corpus import stopwords
import nltk
from collections import Counter
import pickle
import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer
import re

def create_dynamic_stopwords(corpus_path, tfidf_threshold_percentile=10, freq_threshold=0.01):
    """
    Create a dynamic list of stopwords based on both TF-IDF scores and word frequencies.
    
    Args:
    - corpus_path (str): Path to the corpus file.
    - tfidf_threshold_percentile (float): The percentile threshold for considering a word as a stopword based on TF-IDF scores.
    - freq_threshold (float): The threshold for considering a word as a stopword based on frequency.
    
    Returns:
    - set: A set of dynamically determined stopwords.
    """
    # Read the corpus
    with open(corpus_path, 'r') as file:
        text = file.read()

    # Extract lowercase sentences (utterances) from the text
    utterances = re.findall(r'\n([a-z ].+)', text)

    # Calculate TF-IDF scores
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(utterances)
    feature_names = tfidf_vectorizer.get_feature_names_out()
    word_tfidf_scores = tfidf_matrix.sum(axis=0).A1
    word_tfidf_dict = dict(zip(feature_names, word_tfidf_scores))

    # Determine the TF-IDF threshold for stopwords
    tfidf_threshold = np.percentile(list(word_tfidf_dict.values()), tfidf_threshold_percentile)

    # Calculate word frequencies
    word_freq = Counter(word.lower() for utterance in utterances for word in utterance.split())
    total_words = sum(word_freq.values())

    # Identify stopwords based on TF-IDF and frequency
    stopwords = {word for word, score in word_tfidf_dict.items() if score <= tfidf_threshold}
    stopwords.update({word for word, freq in word_freq.items() if freq / total_words > freq_threshold})
    return stopwords

def process_rollins(corpus_path, stopwords):
    """
    Process the Rollins corpus with stopwords removal from the sentences.
    
    Args:
    - corpus_path (str): Path to the Rollins corpus file.
    - stopwords (set): A set of stopwords to be filtered out.
    
    Returns:
    - list: A list of (spoken_words, referents) pairs.
    """
    with open(corpus_path, 'r') as file:
        lines = file.readlines()

    data = []
    spoken_words = []
    referents = []

    for line in lines:
        words = line.strip().split()
        
        if not words:
            continue
        
        if words[0].isupper():
            referents = words
            if spoken_words:
                data.append((spoken_words, referents))
                spoken_words = []

        else:
            filtered_words = [word for word in words if word.lower() not in stopwords]
            spoken_words = filtered_words
        
    return data

def process_raw_rollins(corpus_path):
    """
    Process the Rollins corpus with stopwords removal from the sentences.
    
    Args:
    - corpus_path (str): Path to the Rollins corpus file.
    - stopwords (set): A set of stopwords to be filtered out.
    
    Returns:
    - list: A list of (spoken_words, referents) pairs.
    """
    with open(corpus_path, 'r') as file:
        lines = file.readlines()

    data = []
    spoken_words = []
    referents = []

    for line in lines:
        words = line.strip().split()
        
        if not words:
            continue
        
        if words[0].isupper():
            referents = words
            if spoken_words:
                data.append((spoken_words, referents))
                spoken_words = []
        
        else:
            filtered_words = [word for word in words if word.lower()]
            spoken_words = filtered_words
    return data

def process_gold(lexicon_path):
    """
    Process the gold standard lexicon.
    
    Args:
    - lexicon_path (str): Path to the gold standard lexicon file.
    
    Returns:
    - dict: The gold standard lexicon.
    """
    # Open and read the file specified by the lexicon_path
    with open(lexicon_path, 'r') as file:
        lines = file.readlines()

    # Initialize the gold standard lexicon
    gold_standard = {}

    # Process each line in the file
    for line in lines:
        word, ref = line.strip().split()
        gold_standard.setdefault(ref, []).append(word)

    return gold_standard

def process_rollins_combined(corpus_path, static_stopwords, dynamic_stopwords):
    """
    Process the Rollins corpus with both static and dynamic stopwords removal.
    
    Args:
    - corpus_path (str): Path to the Rollins corpus file.
    - static_stopwords (set): A set of static stopwords to be filtered out.
    - dynamic_stopwords (set): A set of dynamic stopwords to be filtered out.
    
    Returns:
    - list: A list of (spoken_words, referents) pairs.
    """
    combined_stopwords = static_stopwords.union(dynamic_stopwords)

    with open(corpus_path, 'r') as file:
        lines = file.readlines()

    data = []
    spoken_words = []
    referents = []

    for line in lines:
        words = line.strip().split()
        
        if not words:
            continue
        
        if words[0].isupper():
            referents = words
            if spoken_words:
                data.append((spoken_words, referents))
                spoken_words = []

        else:
            filtered_words = [word for word in words if word.lower() not in combined_stopwords]
            spoken_words = filtered_words

    return data

# Paths to the corpus and lexicon files
corpus_path = "rollins.txt"
lexicon_path = "gold.txt"

nltk.download('stopwords')
static_stopwords = set(stopwords.words('english'))

data_pairs_raw = process_raw_rollins(corpus_path)

# Process the file with static stopwords
data_pairs_static = process_rollins(corpus_path, static_stopwords)

# Create dynamic stopwords list
dynamic_stopwords = create_dynamic_stopwords(corpus_path)
print(dynamic_stopwords)

# Process the file with dynamic stopwords
data_pairs_static = process_rollins(corpus_path, static_stopwords)
data_pairs_dynamic = process_rollins(corpus_path, dynamic_stopwords)
gold_standard = process_gold(lexicon_path)

data_pairs_combined = process_rollins_combined(corpus_path, static_stopwords, dynamic_stopwords)

with open('data_pairs.pkl', 'wb') as file:
    pickle.dump(data_pairs_raw, file)

with open('data_pairs_combined.pkl', 'wb') as file:
    pickle.dump(data_pairs_combined, file)

with open('data_pairs_static.pkl', 'wb') as file:
    pickle.dump(data_pairs_static, file)

with open('data_pairs_dynamic.pkl', 'wb') as file:
    pickle.dump(data_pairs_dynamic, file)

# Save the processed gold_standard using pickle
with open('gold_standard.pkl', 'wb') as file:
    pickle.dump(gold_standard, file)
