import os                                                                                                                                                                                                          
from dotenv import load_dotenv, find_dotenv
import chromadb.utils.embedding_functions as embedding_functions
from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2 # this is how you import in Chroma 0.5.0+

from dotenv import load_dotenv
import openai
import pandas as pd
import chromadb

ef = ONNXMiniLM_L6_V2(preferred_providers=["CPUExecutionProvider"])

file_path = 'wiki_movie_plots_deduped.csv'  # Update with your dataset path
data = pd.read_csv(file_path)
data['Seq_Num'] = data.index + 1


N=2
data=data[:N]
#print(data)    
API_KEY=os.getenv("OPENAI_API_KEY") 
print("API_KEY ",API_KEY)

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key="",
                model_name="text-embedding-3-small"
            )

            
vectors = openai_ef(data['Wiki Page'])
chroma_client = chromadb.Client()

collection = chroma_client.get_or_create_collection("movie_plots1", embedding_function=ef)
wiki=data['Wiki Page'][0]
seq=data['Seq_Num'][0]

print(wiki)
print(seq)
print(vectors[0])

collection.add(
    documents=[wiki],
    ids=["id1"],
    embeddings=vectors[0]
)

query="Kansas_Saloon_Smasher"
query_embedding=openai_ef(query)
results = collection.query(query_embeddings=query_embedding ,n_results=1)
print(results)


 
