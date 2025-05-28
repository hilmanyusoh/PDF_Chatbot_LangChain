import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class HadithQA:
    def __init__(self):
        self.index = faiss.read_index("hadith_faiss_index/index.faiss")
        with open("hadith_chunks.pkl", "rb") as f:
            self.chunks = pickle.load(f)
        self.embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        # Load tokenizer and model manually
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

    def summarize(self, text, max_length=50, min_length=10):
        inputs = self.tokenizer([text], max_length=1024, truncation=True, return_tensors="pt")
        summary_ids = self.model.generate(
            inputs["input_ids"],
            num_beams=4,
            max_length=max_length,
            min_length=min_length,
            early_stopping=True
        )
        summary = self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn", use_fast=False)

        return summary

    def ask_question(self, question, k=3):
        q_emb = self.embedding_model.encode([question], normalize_embeddings=True)
        q_emb = np.array(q_emb, dtype='float32')
        _, I = self.index.search(q_emb, k)
        retrieved = [self.chunks[i] for i in I[0]]
        context = " ".join(retrieved)
        summary = self.summarize(context, max_length=50, min_length=10)
        return summary

if __name__ == "__main__":
    qa = HadithQA()
    question = "ละหมาดคืออะไร"
    answer = qa.ask_question(question)
    print("ตอบ:", answer)
