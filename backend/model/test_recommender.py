# backend/model/test_recommender.py

import pandas as pd
from recommender import RecipeRecommender

recommender = RecipeRecommender()

def interactive_session():
    print("✨ Welcome to the Recipe Recommender!\n")

    user_id = input("Enter your user ID (for CF recommendations): ").strip()
    user_ingredients = input("Enter the ingredients you have (comma separated): ").strip().split(",")

    print("\n🔍 Finding recommendations...\n")

    # Use dataset from the class
    df = recommender.df

    # Run all recommendation models
    tfidf_rec = recommender.recommend(user_input=user_ingredients, top_n=5, method="tfidf")
    sbert_rec = recommender.recommend(user_input=user_ingredients, top_n=5, method="sbert")
    hybrid_rec = recommender.recommend(user_input=user_ingredients, top_n=5, method="hybrid")
    try:
        cf_rec = recommender.recommend(user_input=user_ingredients, top_n=5, method="cf", user_id=user_id)
    except Exception:
        cf_rec = pd.DataFrame()
    # Display recommendation names only
    def show_recs(title, recs):
        print(f"💡 {title} Recommendations:")
        if recs.empty:
            print("No recommendations found.\n")
            return []
        for i, row in enumerate(recs['recipe_name'], 1):
            print(f"{i}. {row}")
        print("\n" + "-" * 60 + "\n")
        return recs['recipe_name'].tolist()

    tfidf_list = show_recs("TF-IDF", tfidf_rec)
    sbert_list = show_recs("SBERT", sbert_rec)
    hybrid_list = show_recs("Hybrid", hybrid_rec)
    cf_list = show_recs("Collaborative Filtering", cf_rec)

    # Combine all recipe names for rating selection
    all_recipes = list(dict.fromkeys(tfidf_list + sbert_list + hybrid_list + cf_list))

    if not all_recipes:
        print("No recipes found to view or rate.\n")
        return

    print("⭐ Choose a recipe to view details (and optionally rate):")
    for i, r in enumerate(all_recipes, 1):
        print(f"{i}. {r}")
    choice = input("\nEnter the number of the recipe (or press Enter to skip): ").strip()
    if not choice:
        print("\n✨ Thank you for using the recipe recommender!")
        return

    try:
        idx = int(choice) - 1
        recipe_name = all_recipes[idx]
    except (ValueError, IndexError):
        print("⚠️ Invalid choice.")
        return

    # Fetch full recipe details
    recipe_data = df[df['recipe_name'] == recipe_name].iloc[0]
    print(f"\n🍽️ Recipe Details for: {recipe_name}")
    print("-" * 60)
    
    if 'ingredients' in recipe_data:
        print("🧾 Ingredients:\n", recipe_data['ingredients'])
    if 'directions' in recipe_data:
        print("\n👨‍🍳 Instructions:\n", recipe_data['directions'])
    if 'total_time' in recipe_data:
        print(f"\n🕒 Total Time: {recipe_data['total_time']}")
    if 'prep_time' in recipe_data:
        print(f"⏳ Prep Time: {recipe_data['prep_time']}")
    if 'cook_time' in recipe_data:
        print(f"🔥 Cook Time: {recipe_data['cook_time']}")
    if 'servings' in recipe_data:
        print(f"👥 Servings: {recipe_data['servings']}")
    if 'yield' in recipe_data:
        print(f"🍱 Yield: {recipe_data['yield']}")

    # Ask for rating
    rating = input(f"\nEnter your rating for '{recipe_name}' (0-5): ").strip()
    if rating:
        try:
            rating = float(rating)
            if 0 <= rating <= 5:
                recommender.add_user_rating(user_id, recipe_name, rating)
            else:
                print("⚠️ Rating must be between 0 and 5.")
        except ValueError:
            print("⚠️ Invalid rating entered.")
    else:
        print("Skipping rating.")

    print("\n✨ Thank you for using the recipe recommender!")

if __name__ == "__main__":
    interactive_session()
