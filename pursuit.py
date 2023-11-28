# If w is a novel word, initiate Aw = {A(w, h0) = gamma}, where h0 = arg min max(Am)
# Initialization does not begin with a random word, it takes the most frequent value 
# among the least frequent ones?

# Functions: Initialize, Update, Lexicalize

# Example:  "dog":      {[DOG, 0.8], [CAT, 0.1]}
#           "whisker":  {[CAT, 0.6]}
#           "ball":     {[CAT, 0.1]}
# The model matches ["dog", DOG] and 
# moves onto selecting CAT among ["whisker", "ball"]
# because it is the highest probability among the lowest
# selects ["whisker", CAT] due to 0.6

import pickle
import random
import numpy as np

# Load datasets
with open("data_pairs.pkl", "rb") as file:
    data_pairs = pickle.load(file)

with open("data_pairs_static.pkl", "rb") as file:
    data_pairs_static = pickle.load(file)

with open("data_pairs_dynamic.pkl", "rb") as file:
    data_pairs_dynamic = pickle.load(file)

with open("data_pairs_combined.pkl", "rb") as file:
    data_pairs_combined = pickle.load(file)

with open("gold_standard.pkl", "rb") as file:
    gold_standard = pickle.load(file)

def collect_objects(data_pairs):
    """Collect all unique objects from the dataset."""
    objects = set()
    for _, obj in data_pairs:
        objects.update(obj)
    return objects

def initialize_matrix(data_pairs, objects):
    """Initialize the association matrix using dictionaries."""
    matrix = {}
    for utterance, _ in data_pairs:
        for word in utterance:
            if word not in matrix:
                matrix[word] = {o: 0 for o in objects}
    return matrix

def reward(x, gamma):
    """Increase the association strength between a word and an object."""
    return x + gamma * (1 - x)

def punish(x, gamma):
    """Decrease the association strength between a word and an object."""
    return x * (1 - gamma)

def pursuit(data_pairs, matrix, gamma=0.05, threshold=0.5):
    """Run the Pursuit algorithm to build the lexicon."""
    lexicon = {}
    for U, M in data_pairs:
        for word in U:
            if all(matrix[word][o] == 0 for o in matrix[word]):
                h_init = min(matrix[word], key=matrix[word].get)
                matrix[word][h_init] = gamma
            h = max(matrix[word], key=matrix[word].get)
            if h in M:
                matrix[word][h] = reward(matrix[word][h], gamma)
                if matrix[word][h] > threshold:
                    lexicon[word] = h
            else:
                matrix[word][h] = punish(matrix[word][h], gamma)
                h_new = random.choice(list(M))
                if h_new not in matrix[word]:
                    matrix[word][h_new] = 0
                matrix[word][h_new] = reward(matrix[word][h_new], gamma)
    return lexicon

def evaluate_model(proposed_pairs, gold_standard):
    """
    Evaluate the proposed word-referent pairs against the gold standard.
    
    Args:
    - proposed_pairs (dict): A dictionary of proposed word-referent pairs.
    - gold_standard (dict): A dictionary with referents as keys and associated terms as values.
    
    Returns:
    - precision (float): Precision of the proposed pairs.
    - recall (float): Recall of the proposed pairs.
    - f_score (float): F-score of the proposed pairs.
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    # Check each proposed word-referent pair
    for word, referent in proposed_pairs.items():
        correct_terms = [term for terms in gold_standard.values() for term in terms]
        if (
            referent in gold_standard
            and word in correct_terms
            and word in gold_standard[referent]
        ):
            true_positives += 1
        else:
            false_positives += 1
    # Calculate false negatives (correct pairs that weren't proposed by the model)
    for referent, terms in gold_standard.items():
        for term in terms:
            if term not in proposed_pairs or proposed_pairs.get(term) != referent:
                false_negatives += 1

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f_score

# Lists to store metric values across the 1000 runs
precisions = []
recalls = []
f_scores = []

objects = collect_objects(data_pairs)

# Run the model 1000 times and evaluate its performance
for _ in range(1000):
    # Initialize matrix for each run
    matrix = initialize_matrix(data_pairs, objects)

    # Run the Pursuit algorithm
    lexicon = pursuit(data_pairs, matrix)

    # Evaluate the generated lexicon
    precision, recall, f_score = evaluate_model(lexicon, gold_standard)

    # Store the metrics
    precisions.append(precision)
    recalls.append(recall)
    f_scores.append(f_score)

# Compute mean and standard deviation for each metric over the 1000 runs
precision_mean = np.mean(precisions)
precision_sd = np.std(precisions)
recall_mean = np.mean(recalls)
recall_sd = np.std(recalls)
f_score_mean = np.mean(f_scores)
f_score_sd = np.std(f_scores)

# Print out the results
print(f"Precision: Mean = {precision_mean:.2%}, Standard Deviation = {precision_sd:.2%}")
print(f"Recall: Mean = {recall_mean:.2%}, Standard Deviation = {recall_sd:.2%}")
print(f"F-score: Mean = {f_score_mean:.2%}, Standard Deviation = {f_score_sd:.2%}")

