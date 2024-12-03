import faiss
from sentence_transformers import SentenceTransformer

# Load FAISS index
faiss_index = faiss.read_index("hadith_faiss_index.index")

# Load associated texts
with open("hadith_texts.txt", "r") as f:
    texts = f.readlines()
texts = [text.strip() for text in texts]  # Remove newline characters

# Load the embedding model
embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")  # Same model used to create embeddings

# Function to ask questions
def ask_question(question, k=3):
    # Generate embedding for the question
    question_embedding = embedding_model.encode([question], normalize_embeddings=True)
    
    # Perform similarity search
    _, indices = faiss_index.search(question_embedding, k)  # Retrieve top-k results
    
    # Get the most relevant texts
    retrieved_texts = [texts[i] for i in indices[0]]
    
    # Combine the retrieved texts as context
    context = " ".join(retrieved_texts)
    return context

# Example usage
question = "tell me about praying"
context = ask_question(question)
print("Retrieved Context:", context)
