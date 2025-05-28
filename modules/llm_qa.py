import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

class SimpleHadithQA:
    def __init__(self):
        # โหลด FAISS index และ chunks
        self.index = faiss.read_index("/Users/hilmanyusoh/Desktop/PDF_Chatbot_LangChain/collect_data/hadith_faiss_index/index.faiss")
        with open("/Users/hilmanyusoh/Desktop/PDF_Chatbot_LangChain/collect_data/hadith_chunks.pkl", "rb") as f:
            self.chunks = pickle.load(f)
        self.embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

    def ask(self, question, k=1):
        # แปลงคำถามเป็น embedding
        q_emb = self.embedding_model.encode([question], normalize_embeddings=True)
        q_emb = np.array(q_emb, dtype='float32')
        # ค้นหาข้อความที่ใกล้เคียงที่สุด
        _, I = self.index.search(q_emb, k)
        return self.chunks[I[0][0]]

if __name__ == "__main__":
    qa = SimpleHadithQA()
    question = "How should I pray?"
    answer = qa.ask(question)
    print("ตอบ:", answer)
