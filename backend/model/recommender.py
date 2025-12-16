import os
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MinMaxScaler
import joblib
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# ----------------- NLTK -----------------
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# ----------------- THRESHOLDS -----------------
TFIDF_THRESHOLD = 0.08
SBERT_THRESHOLD = 0.45


class RecipeRecommender:
    def __init__(self):
        # ---------- Paths ----------
        BASE_DIR = os.path.dirname(__file__)
        PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

        self.DATA_PATH = os.path.join(PROJECT_ROOT, "data", "recipes_cleaned.csv")
        self.EMB_PATH = os.path.join(PROJECT_ROOT, "data", "recipes_embeddings.npy")
        self.VECT_PATH = os.path.join(PROJECT_ROOT, "data", "tfidf_vectorizer.joblib")
        self.RATINGS_PATH = os.path.join(PROJECT_ROOT, "data", "user_ratings.csv")

        self.SBERT_MODEL_NAME = "all-MiniLM-L6-v2"

        # ---------- Helpers ----------
        self.lemmatizer = WordNetLemmatizer()
        self.STOPWORDS = set(stopwords.words('english'))

        # ---------- Load dataset ----------
        self.df = pd.read_csv(self.DATA_PATH)
        self._prepare_dataset()

        # ---------- TF-IDF ----------
        if os.path.exists(self.VECT_PATH):
            self.vectorizer = joblib.load(self.VECT_PATH)
        else:
            self.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
            self.vectorizer.fit(self.df['Full_Text'].fillna("").astype(str))
            joblib.dump(self.vectorizer, self.VECT_PATH)

        self.tfidf_matrix = self.vectorizer.transform(
            self.df['Full_Text'].fillna("").astype(str)
        )

        # ---------- SBERT ----------
        self.sbert = SentenceTransformer(self.SBERT_MODEL_NAME)
        self.recipe_embeddings = self._load_or_compute_embeddings()

        # ---------- Ratings ----------
        if os.path.exists(self.RATINGS_PATH):
            self.ratings_df = pd.read_csv(self.RATINGS_PATH)
        else:
            self.ratings_df = pd.DataFrame(columns=["user_id", "recipe_name", "rating"])

        self.RETURN_COLUMNS = [
            c for c in [
                'recipe_name', 'rating', 'total_time', 'servings',
                'cuisine', 'course', 'url', 'ingredients',
                'directions', 'clean_ingredients',
                'ingredients_joined', 'img_src'
            ] if c in self.df.columns
        ]

    # ----------------- TEXT CLEANING -----------------
    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        tokens = text.split()
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens if t not in self.STOPWORDS]
        return " ".join(tokens)

    def _build_query_text(self, user_input):
        if isinstance(user_input, list):
            user_input = " ".join(map(str, user_input))
        return self.clean_text(user_input)

    # ----------------- DATASET PREP -----------------
    def _prepare_dataset(self):
        df = self.df

        if 'ingredients_joined' in df.columns:
            df['cleaned_ingredients'] = df['ingredients_joined'].apply(self.clean_text)
        elif 'clean_ingredients' in df.columns:
            df['cleaned_ingredients'] = df['clean_ingredients'].apply(self.clean_text)
        else:
            df['cleaned_ingredients'] = df['recipe_name'].apply(self.clean_text)

        df['Full_Text'] = (
            df['cleaned_ingredients'] + " " +
            df['recipe_name'].fillna("") + " " +
            df.get('cuisine', "") + " " +
            df.get('course', "")
        )

        self.df = df

    # ----------------- EMBEDDINGS -----------------
    def _load_or_compute_embeddings(self):
        if os.path.exists(self.EMB_PATH):
            try:
                emb = np.load(self.EMB_PATH)
                if emb.shape[0] == len(self.df):
                    return emb
            except Exception:
                pass

        texts = self.df['Full_Text'].fillna("").astype(str).tolist()
        emb = self.sbert.encode(texts, convert_to_numpy=True, batch_size=64)
        np.save(self.EMB_PATH, emb)
        return emb

    # ----------------- TF-IDF RECOMMEND -----------------
    def recommend_tfidf(self, user_input, top_n=10):
        query = self._build_query_text(user_input)
        q_vec = self.vectorizer.transform([query])

        sims = cosine_similarity(q_vec, self.tfidf_matrix).flatten()
        max_score = sims.max()

        if max_score < TFIDF_THRESHOLD:
            return pd.DataFrame([{
                "message": "Input does not match any known food or ingredient."
            }])

        idx = sims.argsort()[-top_n:][::-1]
        return self.df.iloc[idx][self.RETURN_COLUMNS].reset_index(drop=True)

    # ----------------- SBERT RECOMMEND -----------------
    def recommend_sbert(self, user_input, top_n=10):
        query = self._build_query_text(user_input)
        q_emb = self.sbert.encode([query], convert_to_numpy=True)

        sims = cosine_similarity(q_emb, self.recipe_embeddings).flatten()
        max_score = sims.max()

        if max_score < SBERT_THRESHOLD:
            return pd.DataFrame([{
                "message": "No meaningful semantic match found."
            }])

        idx = sims.argsort()[-top_n:][::-1]
        return self.df.iloc[idx][self.RETURN_COLUMNS].reset_index(drop=True)

    # ----------------- HYBRID RECOMMEND -----------------
    def recommend_hybrid(self, user_input, top_n=10, alpha=0.3, beta=0.7):
        query = self._build_query_text(user_input)

        # TF-IDF
        q_vec = self.vectorizer.transform([query])
        tfidf_sims = cosine_similarity(q_vec, self.tfidf_matrix).flatten()

        if tfidf_sims.max() < TFIDF_THRESHOLD:
            return pd.DataFrame([{
                "message": "Input does not match any known food or ingredient."
            }])

        # SBERT
        q_emb = self.sbert.encode([query], convert_to_numpy=True)
        sbert_sims = cosine_similarity(q_emb, self.recipe_embeddings).flatten()

        if sbert_sims.max() < SBERT_THRESHOLD:
            return pd.DataFrame([{
                "message": "No relevant recipe found for the given input."
            }])

        # Normalize
        tfidf_norm = (tfidf_sims - tfidf_sims.min()) / (tfidf_sims.max() - tfidf_sims.min() + 1e-9)
        sbert_norm = (sbert_sims - sbert_sims.min()) / (sbert_sims.max() - sbert_sims.min() + 1e-9)

        final_score = alpha * tfidf_norm + beta * sbert_norm
        idx = final_score.argsort()[-top_n:][::-1]

        return self.df.iloc[idx][self.RETURN_COLUMNS].reset_index(drop=True)

    # ----------------- UNIFIED INTERFACE -----------------
    def recommend(self, user_input, top_n=10, method="hybrid"):
        method = method.lower()
        if method == "tfidf":
            return self.recommend_tfidf(user_input, top_n)
        if method == "sbert":
            return self.recommend_sbert(user_input, top_n)
        return self.recommend_hybrid(user_input, top_n)


# ---------- Smoke Test ----------
if __name__ == "__main__":
    rec = RecipeRecommender()
    print("Recommender module loaded. Dataset size:", len(rec.df))
    sample_ingredients = ["paneer","onion","garlic","tomato"]
    print("TF-IDF sample:")
    print(rec.recommend(sample_ingredients, top_n=5, method="tfidf")[['recipe_name']])
    print("SBERT sample:")
    print(rec.recommend(sample_ingredients, top_n=5, method="sbert")[['recipe_name']])
    print("Hybrid sample:")
    print(rec.recommend(sample_ingredients, top_n=5, method="hybrid")[['recipe_name']])
