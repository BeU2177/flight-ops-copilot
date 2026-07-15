import main
import numpy as np

rag = main.LocalRAGEngine()

# Get the glossary chunk
doc_res = rag.collection.get(where={"source": "weather_glossary.md"}, include=["embeddings", "documents"], limit=1)
print("Doc text:", doc_res["documents"])
doc_emb = doc_res["embeddings"][0]

# Encode query
query = "What is CAVOK?"
query_emb = rag.embedding.encode([query])[0]

# Query with where to get the specific doc distance
res_query = rag.collection.query(
    query_embeddings=[query_emb],
    where={"source": "weather_glossary.md"},
    n_results=1
)
print("Query result for weather_glossary.md only:")
print("IDs:", res_query["ids"])
print("Distances:", res_query["distances"])
print("Documents:", res_query["documents"])

# Calculate L2 distance directly in numpy
l2_dist = np.sum((np.array(doc_emb) - np.array(query_emb))**2)
print("Direct numpy L2 distance:", l2_dist)
