import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Load podcasts.csv
df = pd.read_csv("podcasts.csv")

# Step 2: Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words="english")

# Step 3: Generate TF-IDF embeddings for the podcast descriptions
print("Generating TF-IDF embeddings for podcast descriptions...")
tfidf_matrix = tfidf_vectorizer.fit_transform(df['description'].fillna(''))

# Step 4: Function to recommend similar podcasts
def recommend_podcasts(query, n_results=5):
    """Recommend podcasts similar to the given query."""
    print("Generating TF-IDF embedding for query...")
    query_vector = tfidf_vectorizer.transform([query])  # Transform the query into the TF-IDF space

    print("Calculating similarities...")
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()  # Compute cosine similarity

    # Get indices of the top n_results most similar items
    top_indices = similarities.argsort()[::-1][:n_results]

    # Fetch the top recommendations
    recommendations = df.iloc[top_indices]
    return recommendations

# Step 5: Test the recommendation engine
query = "A podcast about technology and innovation"
recommendations = recommend_podcasts(query)

print("\nRecommendations:")
for _, row in recommendations.iterrows():
    print(f"- {row['name']} by {row['publisher']} ({row['description']})")
