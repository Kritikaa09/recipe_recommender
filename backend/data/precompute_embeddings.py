import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np

# Load cleaned recipes
df = pd.read_csv("recipes_cleaned.csv")

# Load SBERT model
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

# Compute embeddings
embeddings = sbert_model.encode(df['clean_ingredients'].tolist(), convert_to_numpy=True)

# Save embeddings to file
np.save("recipes_embeddings.npy", embeddings)
print("✅ SBERT embeddings saved to recipes_embeddings.npy")
