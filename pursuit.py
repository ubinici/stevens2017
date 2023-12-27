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

def initialize_matrix(data_pairs, objects, smoothing=0.001):
    """Initialize the association matrix using dictionaries."""
    matrix = {}
    for utterance, _ in data_pairs:
        for word in utterance:
            if word not in matrix:
                matrix[word] = {o: smoothing for o in objects}
    return matrix

def reward(x, gamma):
    """Increase the association strength between a word and an object."""
    return x + gamma * (1 - x)

def punish(x, gamma):
    """Decrease the association strength between a word and an object."""
    return x * (1 - gamma)

def pursuit(data_pairs, matrix, gamma=0.05, threshold=0.5, smoothing=0.001):
    """
    Run the Pursuit algorithm to build the lexicon with a smoothing factor.
    
    Args:
        data_pairs (list): A list of tuples, each containing an utterance (list of words) and meanings (set of referents).
        matrix (dict): A dictionary representing the association strengths between words and meanings.
        gamma (float): The learning rate, a factor by which the association strengths are updated.
        threshold (float): The confidence threshold at which a word-meaning pair is added to the lexicon.
        smoothing (float): The smoothing factor λ to be added to each word-meaning association.

    Returns:
        lexicon (dict): A dictionary mapping words to their most strongly associated meanings.
    """
    
    # Initialize the lexicon, which will be a dictionary mapping words to meanings.
    lexicon = {}
    
    # Determine the number of unique referents in the matrix. This will be used in the smoothing calculations.
    # We assume that each word has associations with all referents, so we take the length of the associations
    # of the first word in the matrix as the number of referents.
    N = len(next(iter(matrix.values())))
    
    # Iterate over each utterance-meaning pair in the input data.
    for U, M in data_pairs:
        # For each word in the utterance:
        for word in U:
            # Calculate the total association strength of the word with all meanings, adding the smoothing factor
            # for each meaning. This normalization step is necessary to turn association strengths into probabilities.
            total_association = sum(matrix[word].values()) + N * smoothing
            
            # Calculate the smoothed conditional probability of each meaning given the word.
            # This is done using the provided formula with smoothing.
            probabilities = {m: (matrix[word][m] + smoothing) / total_association for m in matrix[word]}
            
            # Find the meaning with the highest conditional probability for the current word.
            h = max(probabilities, key=probabilities.get)
            
            # If the highest probability meaning is one of the meanings in the current utterance's meaning set:
            if h in M:
                # Increase the association strength for this word-meaning pair using the reward function.
                matrix[word][h] = reward(matrix[word][h], gamma)
                
                # If the updated association strength exceeds the threshold, add the word-meaning pair to the lexicon.
                if matrix[word][h] > threshold:
                    lexicon[word] = h
            else:
                # If the highest probability meaning is not in the utterance's meaning set, decrease the
                # association strength for this word-meaning pair using the punish function.
                matrix[word][h] = punish(matrix[word][h], gamma)
                
                # Also decrease the association strength for other meanings not in the utterance's meaning set.
                for other_meaning in set(matrix[word]) - set(M):
                    matrix[word][other_meaning] = punish(matrix[word][other_meaning], gamma)
                
                # Choose a new meaning randomly from the utterance's meaning set.
                h_new = random.choice(list(M))
                
                # If the new meaning is not already associated with the word, initialize its association strength with the smoothing factor.
                if h_new not in matrix[word]:
                    matrix[word][h_new] = smoothing
                
                # Increase the association strength for the new word-meaning pair using the reward function.
                matrix[word][h_new] = reward(matrix[word][h_new], gamma)
    
    # Return the completed lexicon.
    return lexicon

def pursuit_roulette(data_pairs, matrix, gamma=0.05, threshold=0.5, smoothing=0.001):
    """
    Run the Pursuit algorithm with a modified selection mechanism. If the highest probability 
    meaning is not in the current utterance, a roulette-wheel selection is made based on the 
    association strengths of the meanings in the current utterance.

    Roulette-wheel selection ensures that the probability of selecting a particular meaning
    is proportional to its association strength, reflecting a more naturalistic approach to
    word learning, akin to human language acquisition.
    """
    lexicon = {}  # Initialize the lexicon, which will map words to meanings.

    N = len(next(iter(matrix.values())))  # Number of referents in the matrix for smoothing purposes.

    for U, M in data_pairs:
        for word in U:
            # Calculate the total association of the word with all meanings, adding smoothing.
            total_association = sum(matrix[word].values()) + N * smoothing

            # Calculate the smoothed conditional probabilities of meanings given the word.
            probabilities = {m: (matrix[word][m] + smoothing) / total_association for m in matrix[word]}

            # Apply standard Pursuit algorithm for updating associations.
            h = max(probabilities, key=probabilities.get)
            if h in M:
                matrix[word][h] = reward(matrix[word][h], gamma)
            else:
                matrix[word][h] = punish(matrix[word][h], gamma)

                if current_meanings_weights := [
                    (m, matrix[word][m]) for m in M if matrix[word][m] > 0
                ]:
                    # If there are associated meanings with strength > 0, choose based on their weights.
                    h_new, _ = random.choices(current_meanings_weights, weights=[weight for _, weight in current_meanings_weights], k=1)[0]
                    matrix[word][h_new] = reward(matrix[word][h_new], gamma)
                else:
                    # If no meanings have been associated yet, fall back to a completely random choice.
                    random_meaning = random.choice(list(M))
                    matrix[word][random_meaning] = reward(matrix[word].get(random_meaning, smoothing), gamma)

    # After all iterations, build the lexicon based on the updated matrix.
    for word, meanings in matrix.items():
        most_probable_meaning = max(meanings, key=meanings.get)
        if meanings[most_probable_meaning] > threshold:
            lexicon[word] = most_probable_meaning

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

objects = collect_objects(data_pairs_static)

# Run the model 1000 times and evaluate its performance
for _ in range(10):
    # Initialize matrix for each run
    matrix = initialize_matrix(data_pairs_static, objects)

    # Run the Pursuit algorithm
    lexicon = pursuit(data_pairs_static, matrix)
    print(lexicon)
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

