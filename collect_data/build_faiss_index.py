import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
import pickle

def build_index(csv_path="/Users/hilmanyusoh/Desktop/PDF_Chatbot_LangChain/collect_data/hadith_data.csv"):
    # Load data
    df = pd.read_csv(csv_path)
    hadith_texts = df["Hadith Text"].dropna().tolist()

    # Split texts into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = []
    for text in hadith_texts:
        chunks.extend(text_splitter.split_text(text))

    # Create Document objects
    docs = [Document(page_content=chunk) for chunk in chunks]

    # Load embedding model
    embeddings = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    # Build FAISS vectorstore
    vectorstore = FAISS.from_documents(docs, embedding=embeddings)
    vectorstore.save_local("hadith_faiss_index")

    # Save chunks separately for retrieval
    with open("hadith_chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print("FAISS index and chunks saved.")

if __name__ == "__main__":
    build_index()
