from datasets import load_dataset

# Step 1: Load dataset from Hugging Face
dataset = load_dataset("SDMN2001/recipe_difficulties")

# Step 2: Convert the train split to pandas DataFrame
df = dataset["train"].to_pandas()

# Step 3: Save locally
df.to_csv("recipes.csv", index=False)

print("✅ Dataset saved successfully as recipes.csv")
