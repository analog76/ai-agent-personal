import chromadb
from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2 # this is how you import in Chroma 0.5.0+

chroma_client = chromadb.Client()
ef = ONNXMiniLM_L6_V2(preferred_providers=["CPUExecutionProvider"])

# switch `create_collection` to `get_or_create_collection` to avoid creating a new collection every time
collection = chroma_client.get_or_create_collection("myCollection", embedding_function=ef)

# switch `add` to `upsert` to avoid adding the same documents every time
collection.add(
    documents=[
        "This is a document about pineapple",
        "This is a document about oranges"
    ],
    ids=["id1", "id2"]
)

results = collection.query(
    query_texts=["This is a query document about florida"], # Chroma will embed this for you
    n_results=2 # how many results to return
)

print(results)
