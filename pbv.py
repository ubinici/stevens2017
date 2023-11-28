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

def propose_but_verify_with_history(data_pairs):
    """
    Propose consistent word-referent pairs based on the given data pairs. This version starts by selecting
    a word and then matches it with a random referent. It checks for the same word-referent pair in subsequent
    sentences. If a word is present but the referent is not, it chooses a different referent. Skips instances
    where the word is absent in a sentence.

    Args:
    - data_pairs (list): A list of tuples, where each tuple contains a sentence and its potential referents.

    Returns:
    - word_referent_pairs (dict): A dictionary of proposed word-referent pairs.
    """
    word_referent_pairs = {}
    word_history = {}

    # Iterate through each sentence and its potential referents
    for sentence, potential_referents in data_pairs:
        # Select a random word from the sentence
        chosen_word = random.choice(sentence)

        # Check if this word has a history and its referent is in the current list of potential referents
        if chosen_word in word_history:
            if word_history[chosen_word] in potential_referents:
                # If the referent is present, keep the existing pair
                word_referent_pairs[chosen_word] = word_history[chosen_word]
            else:
                # If the referent is not present, choose a different referent
                available_referents = [r for r in potential_referents if r != word_history[chosen_word]]
                if available_referents:
                    new_referent = random.choice(available_referents)
                    word_referent_pairs[chosen_word] = new_referent
                    word_history[chosen_word] = new_referent
        else:
            # If this word does not have a history, match it with a random referent from the list
            chosen_referent = random.choice(potential_referents)
            word_referent_pairs[chosen_word] = chosen_referent
            word_history[chosen_word] = chosen_referent

    return word_referent_pairs

# Function to propose word-referent pairs without considering history
def propose_but_verify(data_pairs):
    word_referent_pairs = {}

    for sentence, potential_referents in data_pairs:
        for word in sentence:
            chosen_referent = random.choice(potential_referents)
            word_referent_pairs[word] = chosen_referent

    return word_referent_pairs

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

# Run the model 1000 times and evaluate its performance
for _ in range(1000):
    proposed_pairs = propose_but_verify(data_pairs_combined)
    precision, recall, f_score = evaluate_model(proposed_pairs, gold_standard)
    
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