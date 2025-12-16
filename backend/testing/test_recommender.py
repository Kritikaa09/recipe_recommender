from backend.model.recommender import recommend_recipes

# Example ingredients
test_ingredients = ["chicken", "onion", "garlic"]

# TF-IDF + SBERT hybrid
results = recommend_recipes(test_ingredients, top_n=5)

print("Recommended Recipes:\n")
print(results[['recipe_name', 'level', 'total_time', 'ingredients']])
