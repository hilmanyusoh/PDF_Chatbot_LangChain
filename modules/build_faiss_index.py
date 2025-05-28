import pandas as pd
import logging
import pickle
import torch

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

logging.basicConfig(level=logging.INFO)

def build_index(
    csv_path="/Users/hilmanyusoh/Desktop/PDF_Chatbot_LangChain/collect_data/hadith_data.csv",
    index_path="hadith_faiss_index",
    chunk_path="hadith_chunks.pkl"
):
    try:
        # Load data
        df = pd.read_csv(csv_path)
        hadith_texts = df["Hadith Text"].dropna().tolist()
        logging.info(f"Loaded {len(hadith_texts)} hadith entries.")

        # Split texts into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = []
        for text in hadith_texts:
            chunks.extend(text_splitter.split_text(text))
        logging.info(f"Created {len(chunks)} text chunks.")

        # Create Document objects
        docs = [Document(page_content=chunk) for chunk in chunks]

        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Using device: {device}")

        # Load embedding model
        embeddings = HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True}
        )

        # Build FAISS vectorstore
        vectorstore = FAISS.from_documents(docs, embedding=embeddings)
        vectorstore.save_local(index_path)
        logging.info(f"Saved FAISS index to: {index_path}")

        # Save chunks separately for retrieval
        with open(chunk_path, "wb") as f:
            pickle.dump(chunks, f)
        logging.info(f"Saved chunks to: {chunk_path}")

        print("✅ FAISS index and chunks saved successfully.")

    except Exception as e:
        logging.error(f"Error during indexing: {e}")

if __name__ == "__main__":
    build_index()
