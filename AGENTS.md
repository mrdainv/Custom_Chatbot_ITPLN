# Agent Guidelines for PMB ITPLN Chatbot

Welcome! If you are an AI agent or a developer contributing to this repository, please follow these guidelines to ensure consistency, maintainability, and quality of the codebase.

## Project Overview
This is a Retrieval-Augmented Generation (RAG) Chatbot application designed to answer questions about the PMB (Penerimaan Mahasiswa Baru) at ITPLN.
- **Frontend/Framework:** Streamlit
- **LLM & Embeddings:** Google Gemini (`gemini-1.5-flash` for generation, `models/embedding-001` for embeddings)
- **Orchestration:** LangChain
- **Vector Database:** FAISS (Local)

## Coding Conventions
1. **Caching:** Always use Streamlit's caching decorators (`@st.cache_data` for data/text extraction, `@st.cache_resource` for connections/models like the vector store and LangChain chains) to prevent unnecessary re-computations on every user interaction.
2. **Environment Variables:** Never hardcode API keys. Always use `.env` files and `os.getenv`.
3. **Error Handling:** Wrap file loading, vector store initialization, and LLM calls in `try-except` blocks. Use `st.error` or `st.warning` to communicate issues to the user gracefully.
4. **LangChain Usage:** Prefer the newer LangChain LCEL (LangChain Expression Language) and builder methods (e.g., `create_history_aware_retriever`, `create_retrieval_chain`) over deprecated chains (`load_qa_chain`).
5. **UI Updates:** When modifying the UI, adhere to Streamlit's conversational elements (`st.chat_message`, `st.chat_input`).

## Testing & Verification
Before submitting any changes, you must ensure the application runs correctly:
1. **Install Dependencies:** `pip install -r requirements.txt`
2. **Syntax Check:** Run `python -m py_compile main.py` to catch obvious syntax errors.
3. **Smoke Test:** You can run `streamlit run main.py --server.headless true` to ensure the app starts without immediate crashes.
4. **Dependencies:** If you add a new library, ensure it is added to `requirements.txt`. Do not add unused libraries.

## Document Handling
- Place test PDFs inside the `docs/` folder.
- Ensure the extraction logic in `get_pdf_text` robustly handles potential errors when reading corrupted or empty PDFs.
