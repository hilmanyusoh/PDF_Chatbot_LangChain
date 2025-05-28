import faiss
import pickle
from sentence_transformers import SentenceTransformer
import torch

class HadithQA:
    def __init__(self):
        # Load FAISS index
        self.index = faiss.read_index("hadith_faiss_index/index.faiss")
        
        # Load chunks
        with open("hadith_chunks.pkl", "rb") as f:
            self.chunks = pickle.load(f)

        # Select device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Load embedding model
        self.embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)

    def ask_question(self, question, k=3):
        q_emb = self.embedding_model.encode([question], normalize_embeddings=True)
        D, I = self.index.search(q_emb, k)
        retrieved = [self.chunks[i] for i in I[0]]
        context = " ".join(retrieved)
        return context

if __name__ == "__main__":
    qa = HadithQA()
    question = "How should I pray?"
    context = qa.ask_question(question)
    print("Retrieved Context:", context)
