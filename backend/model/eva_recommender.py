import os
import pandas as pd
from fuzzywuzzy import fuzz
from recommender import recommend_tfidf, recommend_sbert, recommend_hybrid, _build_query_text, df

TOP_N = 10
RECIPE_MATCH_THRESHOLD = 75
ING_MATCH_THRESHOLD = 70

# ---------- Helper ----------
def clean_ingredient_list(ingredients):
    cleaned = []
    units = {'cup', 'cups', 'tbsp', 'tsp', 'teaspoon', 'tablespoon', 'grams', 'g', 'ml', 'oz', 'lb', 'lbs', 'pinch', 'slice'}
    for i in ingredients:
        if not isinstance(i, str):
            continue
        i = i.lower()
        i = ' '.join([w for w in i.split() if not w.replace('.', '', 1).isdigit() and w not in units])
        cleaned.append(i.strip())
    return cleaned

# ---------- Evaluation ----------
def evaluate_model(test_data, method='hybrid', top_n=TOP_N):
    hit_count, precision_sum, recall_sum = 0, 0, 0
    total = len(test_data)

    for ingredients, target_recipe in test_data:
        query = _build_query_text(ingredients)

        if method == 'tfidf':
            recs = recommend_tfidf(query, top_n=top_n)
        elif method == 'sbert':
            recs = recommend_sbert(query, top_n=top_n)
        elif method == 'hybrid':
            recs = recommend_hybrid(query, top_n=top_n)
        else:
            raise ValueError("Method must be 'tfidf','sbert','hybrid'")

        rec_names = recs['recipe_name'].fillna('').tolist()
        rec_ingredients = recs['ingredients'].fillna('').tolist()

        # ---- Hit Rate ----
        hit = any(fuzz.token_set_ratio(target_recipe.lower(), r.lower()) >= RECIPE_MATCH_THRESHOLD for r in rec_names)
        if hit:
            hit_count += 1

        # ---- Ingredient Matching ----
        target_set = set(clean_ingredient_list(ingredients))
        rec_set_all = set()
        for r in rec_ingredients:
            rec_set_all.update(clean_ingredient_list(str(r).split(',')))

        matched = 0
        for t in target_set:
            if any(fuzz.token_set_ratio(t, r) >= ING_MATCH_THRESHOLD for r in rec_set_all):
                matched += 1

        # Precision and Recall
        precision = matched / len(rec_set_all) if rec_set_all else 0
        recall = matched / len(target_set) if target_set else 0

        precision_sum += precision
        recall_sum += recall

        print(f"✔️ {target_recipe}: Precision={precision:.2f}, Recall={recall:.2f}, Hit={hit}")

    hit_rate = hit_count / total
    avg_precision = precision_sum / total
    avg_recall = recall_sum / total

    print(f"\n📊 {method.upper()} | HitRate@{top_n}: {hit_rate:.2f} | Precision@{top_n}: {avg_precision:.2f} | Recall@{top_n}: {avg_recall:.2f}")
    return hit_rate, avg_precision, avg_recall


# ---------- Test ----------
if __name__ == "__main__":
    test_data = [
        (["egg", "pastry shells", "strawberry", "kiwis", "honey", "cream", "sugar"], "Strawberry Kiwi Tartlets"),
        (["watermelon", "lemon juice", "mint", "honey"], "Watermelon Soup"),
        (["chicken", "lemon", "garlic"], "Lemon Garlic Chicken"),
        (["milk", "sugar", "egg"], "Custard"),
        (["lemon", "sugar", "water", "ice"], "Old-Fashioned Lemonade")
    ]

    for method in ['tfidf', 'sbert', 'hybrid']:
        evaluate_model(test_data, method=method, top_n=TOP_N)
