import os
import pandas as pd
from fuzzywuzzy import fuzz
from recommender import RecipeRecommender
recommender = RecipeRecommender()
TOP_N = 10
FUZZY_THRESHOLD = 80
ING_FUZZY_THRESHOLD = 60  # lower threshold for better ingredient matching

# ---------- Clean ingredients ----------
def clean_ingredient_list(ingredients):
    cleaned = []
    units = {'cup','cups','tbsp','tsp','teaspoon','tablespoon','grams','g','ml','oz','lb','lbs','pinch','slice'}
    for i in ingredients:
        if not isinstance(i, str):
            continue
        i = i.lower()
        i = ' '.join([w for w in i.split() if not w.replace('.', '', 1).isdigit() and w not in units])
        cleaned.append(i.strip())
    return cleaned

# ---------- Evaluation ----------
def evaluate_model(test_data, method='hybrid', top_n=TOP_N):
    hit_count = 0
    precision_sum = 0
    recall_sum = 0
    total = len(test_data)

    for ingredients, target_recipe in test_data:
        if method=='tfidf':
          recs = recommender.recommend(user_input=ingredients, top_n=top_n, method='tfidf')
        elif method=='sbert':
          recs = recommender.recommend(user_input=ingredients, top_n=top_n, method='sbert')
        elif method=='hybrid':
          recs = recommender.recommend(user_input=ingredients, top_n=top_n, method='hybrid')
        else:
            raise ValueError("Method must be 'tfidf','sbert','hybrid'")

        rec_names = recs['recipe_name'].fillna('').tolist()
        rec_ingredients = recs['ingredients'].fillna('').tolist()

        # Hit rate
        hit = any(fuzz.token_sort_ratio(target_recipe.lower(), r.lower()) >= FUZZY_THRESHOLD for r in rec_names)
        if hit:
            hit_count += 1

        # Flatten all ingredients from top-N recommended recipes
        rec_set_all = set()
        for r in rec_ingredients:
            rec_set_all.update(clean_ingredient_list(str(r).split(',')))

        target_set = set(clean_ingredient_list(ingredients))

        # Count matched ingredients
        matched = 0
        for t in target_set:
            if any(fuzz.partial_ratio(t, r) >= ING_FUZZY_THRESHOLD for r in rec_set_all):
                matched += 1

        precision_sum += matched / len(rec_set_all) if rec_set_all else 0
        recall_sum += matched / len(target_set) if target_set else 0

    hit_rate = hit_count / total
    avg_precision = precision_sum / total
    avg_recall = recall_sum / total

    print(f"\n📊 {method.upper()} | HitRate@{top_n}: {hit_rate:.2f} | Precision@{top_n}: {avg_precision:.2f} | Recall@{top_n}: {avg_recall:.2f}")
    return hit_rate, avg_precision, avg_recall

# ---------- Test ----------
if __name__ == "__main__":
    test_data = [
        (["egg", "pastry shells", "strawberry", "kiwis","honey","cream","sugar"], "Strawberry Kiwi Tartlets"),
        (["watermelon", "lemon juice", "mint", "honey"], "Watermelon Soup"),
        (["chicken", "lemon", "garlic"], "Lemon Garlic Chicken"),
        (["milk", "sugar", "egg"], "Custard"),
        (["lemon", "sugar", "water", "ice"], "Old-Fashioned Lemonade")
    ]

    for method in ['tfidf','sbert','hybrid']:
        evaluate_model(test_data, method=method, top_n=TOP_N)
