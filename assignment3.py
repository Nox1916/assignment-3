import numpy as np

# Step 1: Define criteria
criteria = ['ROUGE Score', 'Computational Efficiency', 'Model Size']

# Step 2: Collect data (hypothetical data)
models = ['BERT', 'GPT-2', 'T5']
data = np.array([
    [0.75, 0.85, 0.90],  # ROUGE scores
    [0.95, 0.75, 0.80],  # Computational efficiency (higher is better)
    [0.85, 0.80, 0.70]   # Model size (lower is better)
])

# Step 3: Normalize data
normalized_data = (data - data.min(axis=1, keepdims=True)) / (data.max(axis=1, keepdims=True) - data.min(axis=1, keepdims=True))

# Step 4: Define weights
weights = [0.4, 0.3, 0.3]

# Step 5: Calculate weighted normalized scores
weighted_normalized_scores = normalized_data * weights

# Step 6: Calculate ideal and anti-ideal solutions
ideal_solution = np.max(weighted_normalized_scores, axis=1)
anti_ideal_solution = np.min(weighted_normalized_scores, axis=1)

# Step 7: Calculate similarity to ideal solution
similarity_to_ideal = np.linalg.norm(weighted_normalized_scores - ideal_solution, axis=1)
similarity_to_anti_ideal = np.linalg.norm(weighted_normalized_scores - anti_ideal_solution, axis=1)

# Step 8: Rank models
rank = np.argsort(similarity_to_ideal / (similarity_to_ideal + similarity_to_anti_ideal))

# Print results
print("Ranking of pre-trained models:")
for i, idx in enumerate(rank):
    print(f"{i+1}. {models[idx]}")
