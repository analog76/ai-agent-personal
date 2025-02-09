import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
import chromadb

# ================================
# 1. Create a Sample Dataset
# ================================
# Our dataset contains the following fields: hostname, score, description, and duration.
data = [
    {"hostname": "host1", "score": 5, "description": "This is a complex task with many variables.", "duration": 120},
    {"hostname": "host2", "score": 3, "description": "A simple task with few steps.", "duration": 30},
    {"hostname": "host3", "score": 4, "description": "A moderately complex task requiring planning.", "duration": 60},
    {"hostname": "host4", "score": 2, "description": "Very simple and trivial task.", "duration": 15},
    {"hostname": "host5", "score": 5, "description": "A highly complex and challenging task.", "duration": 180},
]

# =====================================
# 2. Compute Embeddings with SentenceTransformer
# =====================================
# Load a pre-trained model (e.g., all-MiniLM-L6-v2)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Extract the descriptions and compute embeddings.
descriptions = [item['description'] for item in data]
embeddings = model.encode(descriptions, convert_to_tensor=False).tolist()  # list of vectors

# =====================================
# 3. Store Data in Chroma DB
# =====================================
# For simplicity, we are using the default in-memory Chroma client.
client = chromadb.Client()  # uses default settings (in-memory)
collection = client.create_collection(name="tasks_collection")

# Use the hostname as an identifier.
ids = [item["hostname"] for item in data]
# Store metadata (score, duration, and description) along with each embedding.
metadatas = [
    {"score": item["score"], "duration": item["duration"], "description": item["description"]}
    for item in data
]
# For convenience, we also store the description as the document.
documents = descriptions

# Add the items into the collection.
collection.add(
    ids=ids,
    embeddings=embeddings,
    metadatas=metadatas,
    documents=documents
)

# =====================================
# 4. Build a Recommendation Engine Class
# =====================================
class RecommendationEngine:
    def __init__(self, collection, model, top_k=3):
        """
        Initializes the recommendation engine with:
          - a Chroma DB collection
          - a SentenceTransformer model
          - a parameter top_k for the number of similar items to retrieve
        """
        self.collection = collection
        self.model = model
        self.top_k = top_k
        
    def recommend(self, query_description):
        """
        Given a query description, this method:
          1. Computes its embedding.
          2. Queries the Chroma DB collection for the top_k most similar items.
          3. Computes the average "score" and "duration" from the returned items.
          
        Returns a dictionary with the expected complexity (score),
        expected duration, and detailed query results.
        """
        # Compute the embedding for the query.
        query_embedding = self.model.encode(query_description, convert_to_tensor=False)
        
        # Query the collection for the top_k results.
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=self.top_k,
            include=["embeddings", "metadatas", "documents"]
        )
        
        # Extract the score and duration from the metadata of each returned item.
        scores = []
        durations = []
        for metadata in results["metadatas"][0]:
            scores.append(metadata["score"])
            durations.append(metadata["duration"])
        
        # Compute averages.
        expected_score = np.mean(scores) if scores else None
        expected_duration = np.mean(durations) if durations else None
        
        return {
            "expected_score": expected_score,
            "expected_duration": expected_duration,
            "results": results
        }

# Create an instance of the recommendation engine.
engine = RecommendationEngine(collection, model, top_k=3)
 

# =====================================
# 5. Save the Recommendation Engine Model
# =====================================
# For persistence, we can save the engine instance (which encapsulates our vector store and model) using pickle.
with open("recommendation_engine.pkl", "wb") as f:
    pickle.dump(engine, f)
    
with open("engine_config.pkl", "wb") as f:
    pickle.dump(engine_config, f)


# =====================================
# 6. Test the Recommendation Engine
# =====================================
# Here, we run a test query to calculate the expected complexity and duration.
test_description = "A challenging and intricate task that requires careful planning."
recommendation = engine.recommend(test_description)

print("Test Query Description:")
print(test_description)
print("\nRecommendation Results:")
print("Expected Complexity (Score):", recommendation["expected_score"])
print("Expected Duration:", recommendation["expected_duration"])
print("\nDetailed Results from Chroma DB:")
print(recommendation["results"])
