from nltk.corpus import stopwords
import nltk
from collections import Counter
import pickle

def create_dynamic_stopwords(corpus_path, threshold=0.01):
    """
    Create a dynamic list of stopwords based on word frequencies.
    
    Args:
    - corpus_path (str): Path to the corpus file.
    - threshold (float): The threshold for considering a word as a stopword,
                         defined as a proportion of the total word count.
    
    Returns:
    - set: A set of dynamically determined stopwords.
    """
    word_freq = Counter()

    with open(corpus_path, 'r') as file:
        for line in file:
            words = line.strip().split()
            word_freq.update(word.lower() for word in words if not word.isupper())

    total_words = sum(word_freq.values())
    dynamic_stopwords = {word for word, freq in word_freq.items() if freq / total_words > threshold}
    return dynamic_stopwords

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

# Combine static and dynamic stopwords


# Paths to the corpus and lexicon files
corpus_path = "rollins.txt"
lexicon_path = "gold.txt"

nltk.download('stopwords')
static_stopwords = set(stopwords.words('english'))

# Process the file with static stopwords
data_pairs_static = process_rollins(corpus_path, static_stopwords)

# Create dynamic stopwords list
dynamic_stopwords = create_dynamic_stopwords(corpus_path)

# Process the file with dynamic stopwords
data_pairs_static = process_rollins(corpus_path, static_stopwords)
data_pairs_dynamic = process_rollins(corpus_path, dynamic_stopwords)
gold_standard = process_gold(lexicon_path)

data_pairs_combined = process_rollins_combined(corpus_path, static_stopwords, dynamic_stopwords)

# Save the processed data with combined stopwords using pickle
combined_file_path = 'data_pairs_combined.pkl'
with open(combined_file_path, 'wb') as file:
    pickle.dump(data_pairs_combined, file)

with open('data_pairs_static.pkl', 'wb') as file:
    pickle.dump(data_pairs_static, file)

with open('data_pairs_dynamic.pkl', 'wb') as file:
    pickle.dump(data_pairs_dynamic, file)

# Save the processed gold_standard using pickle
with open('gold_standard.pkl', 'wb') as file:
    pickle.dump(gold_standard, file)
