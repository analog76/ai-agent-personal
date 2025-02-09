import os
import openai
import numpy as np
import pickle
import chromadb

from dotenv import load_dotenv

# ----------------------------------------
# 1. Setup OpenAI API Key
# ----------------------------------------
# Option A: Make sure your environment variable OPENAI_API_KEY is set.
# Option B: Replace "your-api-key" with your actual key (not recommended for production).
OPENAI_API_KEY=""

openai.api_key = os.getenv("OPENAI_API_KEY", OPENAI_API_KEY)
  
# ----------------------------------------
# 2. Define a Sample Dataset
# ----------------------------------------
dataset = [
    {
        "hostname": "host1",
        "score": 5,
        "description": "This is a complex task with many variables.",
        "duration": 120,
    },
    {
        "hostname": "host2",
        "score": 3,
        "description": "A simple task with few steps.",
        "duration": 30,
    },
    {
        "hostname": "host3",
        "score": 4,
        "description": "A moderately complex task requiring planning.",
        "duration": 60,
    },
    {
        "hostname": "host4",
        "score": 2,
        "description": "Very simple and trivial task.",
        "duration": 15,
    },
    {
        "hostname": "host5",
        "score": 5,
        "description": "A highly complex and challenging task.",
        "duration": 180,
    },
]

# ----------------------------------------
# 3. Define a Function to Get OpenAI Embeddings
# ----------------------------------------
def get_embedding(text):
    """
    Given a text string, returns its embedding using OpenAI's API.
    """
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"  # General-purpose embedding model
    )
    embedding = response['data'][0]['embedding']
    return embedding

# Precompute embeddings for the descriptions.
descriptions = [item["description"] for item in dataset]

documents_all_columns = []
for record in dataset:
    # You can adjust the formatting as needed.
    doc_text = ". ".join([f"{key}: {value}" for key, value in record.items()])
    documents_all_columns.append(doc_text)


print(" documents**********",documents_all_columns)
print(" descriptions ",descriptions)
descriptions=documents_all_columns
embeddings = [get_embedding(text) for text in descriptions]

# ----------------------------------------
# 4. Store the Data in Chroma DB
# ----------------------------------------
# Create a Chroma client and a collection.
client = chromadb.Client()
# Attempt to create a new collection; if it already exists, get it.
try:
    collection = client.create_collection(name="tasks_collection")
except Exception:
    collection = client.get_collection(name="tasks_collection")

# Use hostnames as unique IDs.
ids = [item["hostname"] for item in dataset]

# Prepare metadata for each entry.
metadatas = [
    {
        "score": item["score"],
        "duration": item["duration"],
        "description": item["description"],
    }
    for item in dataset
]

# We'll also store the description as the document.
documents = descriptions

# Add the items to the Chroma collection.
collection.add(
    ids=ids,
    embeddings=embeddings,
    metadatas=metadatas,
    documents=documents
)

# ----------------------------------------
# 5. Build the Recommendation Engine Class
# ----------------------------------------
class RecommendationEngine:
    def __init__(self, collection, top_k=3, collection_name="tasks_collection"):
        """
        Initializes the recommendation engine.
          - collection: A Chroma DB collection containing the embeddings.
          - top_k: Number of similar items to retrieve.
          - collection_name: Name of the collection (used for reinitialization after unpickling).
        """
        self.collection = collection
        self.top_k = top_k
        self.collection_name = collection_name

    def recommend(self, query_description):
        """
        Given a query description, computes its embedding, queries the collection for the top_k
        similar items, and returns a dictionary containing:
          - expected_score: The average "score" from the retrieved metadata.
          - expected_duration: The average "duration" from the retrieved metadata.
          - results: Detailed results from the query.
        """
        query_embedding = get_embedding(query_description)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=self.top_k,
            include=["embeddings", "metadatas", "documents"]
        )
        scores = []
        durations = []
        for metadata in results["metadatas"][0]:
            scores.append(metadata["score"])
            durations.append(metadata["duration"])
        expected_score = np.mean(scores) if scores else None
        expected_duration = np.mean(durations) if durations else None
        return {
            "expected_score": expected_score,
            "expected_duration": expected_duration,
            "results": results
        }

    def __getstate__(self):
        """
        Custom method for pickling.
        Remove the Chroma collection (which contains non-picklable objects)
        from the instance dictionary.
        """
        state = self.__dict__.copy()
        if "collection" in state:
            del state["collection"]
        return state

    def __setstate__(self, state):
        """
        Custom method for unpickling.
        After restoring the state, reinitialize the Chroma collection.
        """
        self.__dict__.update(state)
        client = chromadb.Client()
        self.collection = client.get_collection(name=self.collection_name)

# ----------------------------------------
# 6. Create and Test the Recommendation Engine
# ----------------------------------------
engine = RecommendationEngine(collection, top_k=3)

# Test the engine with a sample query.
test_description = "A challenging and intricate task that requires careful planning."
result = engine.recommend(test_description)
print("Initial Recommendation Result:")
print("Expected Score:", result["expected_score"])
print("Expected Duration:", result["expected_duration"])
print("Detailed Results:", result["results"])

# ----------------------------------------
# 7. Pickle (Save) and Unpickle (Reload) the Engine
# ----------------------------------------
# Save the engine to a file.
with open("recommendation_engine.pkl", "wb") as f:
    pickle.dump(engine, f)
print("\nRecommendation engine pickled successfully.")

# Later (or in another session), load the engine.
with open("recommendation_engine.pkl", "rb") as f:
    loaded_engine = pickle.load(f)
print("Recommendation engine loaded successfully.")

# Test the loaded engine.
loaded_result = loaded_engine.recommend(test_description)
print("\nLoaded Engine Recommendation Result:")
print("Expected Score:", loaded_result["expected_score"])
print("Expected Duration:", loaded_result["expected_duration"])
print("Detailed Results:", loaded_result["results"])
