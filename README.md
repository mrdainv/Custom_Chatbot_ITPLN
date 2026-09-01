# PMB ITPLN Chatbot (Elena)

This repository contains a RAG (Retrieval-Augmented Generation) based Chatbot application built using **Streamlit**, **LangChain**, and **Google Gemini (gemini-1.5-flash)**. The chatbot, named Elena, is specifically designed to answer questions related to the admission process (PMB) of Institut Teknologi PLN (ITPLN) based on provided PDF documents.

## Features

- **Document-Based Q&A**: Answers questions strictly based on the content of the provided PDF documents.
- **Context-Aware Conversations**: Maintains chat history to handle follow-up questions contextually.
- **Fast and Efficient**: Utilizes Streamlit caching to avoid reprocessing documents on every interaction.
- **Vector Search**: Uses FAISS for efficient similarity search over document embeddings.

## Prerequisites

- Python 3.8+
- Google API Key (for Gemini and Embeddings)

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. **Install dependencies:**
   It is recommended to use a virtual environment.
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Environment Variables:**
   Create a `.env` file in the root directory and add your Google API Key:
   ```env
   GOOGLE_API_KEY=your_google_api_key_here
   ```

## Usage

1. **Add Documents:**
   Place your reference PDF files in the `docs/` folder. The application will read and process all `.pdf` files in this directory.

2. **Run the Application:**
   ```bash
   streamlit run main.py
   ```

3. **Interact with Elena:**
   Open your browser to the local URL provided by Streamlit (usually `http://localhost:8501`) and start asking questions about PMB ITPLN.

## Project Structure

- `main.py`: The main Streamlit application script containing the UI, RAG logic, and document processing.
- `requirements.txt`: List of Python dependencies.
- `docs/`: Directory where source PDF documents should be placed.
- `faiss_index/`: (Generated automatically) Directory where the FAISS vector database is saved locally.
- `.env`: Environment variables file (not included in version control).
