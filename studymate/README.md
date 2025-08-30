### StudyMate: An AI-Powered PDF-Based Q&A System for Students

StudyMate lets you upload one or more PDFs (textbooks, lecture notes, papers), ask natural-language questions, and receive grounded answers with citations to the source pages.

#### Features
- Conversational Q&A grounded in your PDFs
- Accurate PDF text extraction and chunking (PyMuPDF)
- Semantic search with SentenceTransformers embeddings and FAISS
- LLM answer generation via IBM watsonx Mixtral-8x7B-Instruct
- Streamlit UI for local use

#### Quickstart
1) Create and populate an environment file:

```bash
cp .env.example .env
# Fill in IBM creds and project/space id
```

2) Create a virtual environment and install dependencies:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

3) Run the app:

```bash
streamlit run studymate/app/app.py
```

4) In the browser, upload PDFs, click Build Index, and start asking questions.

#### IBM watsonx configuration
Set the following environment variables (or edit `.env`):

```bash
IBM_WATSONX_API_KEY=your_api_key
IBM_WATSONX_URL=https://us-south.ml.cloud.ibm.com
# Provide ONE of the following identifiers
IBM_WATSONX_PROJECT_ID=your_project_id
# or
IBM_WATSONX_SPACE_ID=your_space_id
```

If watsonx credentials are not provided, the app will still perform retrieval and return the top matching chunks.

#### Data and indexes
- Built FAISS index and metadata are stored under `studymate/data/` by default.
- Rebuild the index after uploading a different set of PDFs.

#### Notes
- This app uses cosine similarity with FAISS (IndexFlatIP) over normalized embeddings from `sentence-transformers/all-MiniLM-L6-v2` by default.
- You can customize chunk sizes and retrieval parameters in the sidebar.
