# PDF Chatbot with LangChain

A chatbot system for querying Hadith content using semantic search based on vector similarity. This project loads Hadith data, splits it into chunks, embeds it using HuggingFace Transformers, and searches similar content via FAISS. The UI is built with Streamlit.

---

## Features

* Load Hadith content from a CSV file
* Clean and split content into chunks
* Convert chunks into embeddings using Sentence Transformers
* Store embeddings in a FAISS vector index
* Enable semantic question answering with a user interface

---

## Technologies Used

1. **FAISS**
   A library for efficient similarity search of vector data, suitable for semantic search over large datasets.

2. **Sentence Transformers**
   Uses the `BAAI/bge-small-en-v1.5` model to convert text (e.g., questions) into embeddings for semantic comparison.

3. **LangChain**
   Leverages several LangChain utilities:

   * `RecursiveCharacterTextSplitter` for text chunking
   * `Document` for managing text data
   * `langchain_community.vectorstores.FAISS` to interface with FAISS
   * `HuggingFaceBgeEmbeddings` to generate embeddings from HuggingFace models

4. **PyTorch**
   Used to accelerate embedding computation and support GPU/CPU configuration.

5. **Pandas**
   For loading and manipulating Hadith data from CSV files.

6. **Pickle**
   Serializes and deserializes preprocessed data (e.g., text chunks).

7. **NumPy**
   Handles numerical arrays and ensures embeddings are in `float32` format compatible with FAISS.

8. **Streamlit**
   Provides a simple and interactive web UI for user question-answering.

9. **Logging**
   For debug logs and tracking the system’s operations.

---

## Project Structure

```
PDF_CHATBOT_LANGCHAIN/
│
├── app/
│   ├── app.py                  # Streamlit web app
│   └── qa_system.py           # Backend QA system logic
│
├── data/
│   ├── hadith_data.csv        # Raw Hadith data
│   ├── hadith_chunks.pkl      # Preprocessed and chunked data
│   ├── scraping.py            # Web scraping script
│   └── hadith_faiss_index/    # Directory storing FAISS index
│
├── modules/
│   ├── build_faiss_index.py   # Code to build FAISS index
│   └── llm_qa.py              # Handles LLM-based QA pipeline
│
└── README.md
```

---

## How to Run

1. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Build FAISS index:**

   ```bash
   python modules/build_faiss_index.py
   ```

3. **Run the app:**

   ```bash
   streamlit run app/app.py
   ```

---



