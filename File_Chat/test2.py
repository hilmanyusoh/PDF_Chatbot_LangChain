import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

# Load data
input_file = "hadith_data.csv"  # Replace with your input file path
df = pd.read_csv(input_file)

# Ensure "Hadith Text" column exists
if "Hadith Text" not in df.columns:
    raise ValueError("The column 'Hadith Text' was not found in the CSV file.")

texts = df["Hadith Text"].tolist()  # Extract texts

# Generate embeddings using HuggingFace SentenceTransformer
embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")  # Replace with your preferred model
embeddings = embedding_model.encode(texts, normalize_embeddings=True)

# Create and add embeddings to FAISS index
dimension = embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(dimension)
faiss_index.add(embeddings)

# Save FAISS index
faiss.write_index(faiss_index, "hadith_faiss_index.index")

# Save associated texts (metadata)
with open("hadith_texts.txt", "w") as f:
    for text in texts:
        f.write(text + "\n")

print("FAISS index and texts saved!")
