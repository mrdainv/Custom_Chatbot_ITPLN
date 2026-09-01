import streamlit as st
import os
import time
from PyPDF2 import PdfReader
from dotenv import load_dotenv

import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage

# Load environment variables
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
if google_api_key:
    genai.configure(api_key=google_api_key)

# Constants
MODEL_NAME = "gemini-1.5-flash"
EMBEDDING_MODEL = "models/embedding-001"
VECTOR_STORE_DIR = "faiss_index"


@st.cache_data(show_spinner=False)
def get_pdf_text(pdf_paths):
    """Extracts text from a list of PDF files."""
    text = ""
    for file_path in pdf_paths:
        if file_path.endswith('.pdf'):
            try:
                pdf_reader = PdfReader(file_path)
                for page in pdf_reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"
            except Exception as e:
                st.error(f"Error reading {file_path}: {e}")
    return text


@st.cache_data(show_spinner=False)
def get_text_chunks(text):
    """Splits text into chunks for embedding."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)


@st.cache_resource(show_spinner=False)
def get_vector_store(text_chunks):
    """Creates and saves a FAISS vector store from text chunks."""
    if not text_chunks:
        return None

    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local(VECTOR_STORE_DIR)
    return vector_store


def get_conversational_chain(vector_store_hash=None):
    """Sets up the RAG conversational chain with history awareness."""
    llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0.5, convert_system_message_to_human=True)
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    
    try:
        vector_db = FAISS.load_local(VECTOR_STORE_DIR, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"Error loading vector store: {e}")
        return None

    retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 6})

    # Contextualize question prompt
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, formulate a standalone question "
        "which can be understood without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    # QA prompt
    system_prompt = (
        "You are a personal Bot assistant for answering any questions about certain contxt of given context.\n"
        "You are given a question and a set of context.\n"
        "You are supposed to answer in either Bahasa Indonesia or English, following the language of the user.\n"
        "If the user's question requires you to provide specific information from the context, give your answer based only on the examples provided below. "
        "DON'T generate an answer that is NOT written in the provided examples.\n"
        "If you don't find the answer to the user's question with the examples provided to you below, "
        "answer that you didn't find the answer in the context given and propose him to rephrase his query with more details.\n"
        "Use bullet points if you have to make a list, only if necessary.\n"
        "If the question is about code, answer that you don't know the answer.\n"
        "If there are links avaliable, then u can proceed to access it.\n"
        "If the user ask about your name, answer that your name is Elena.\n"
        "If you don't find the answer to the user's question, just say that you dont know.\n"
        "If the questions is about anything that is NOT related to the given context, answer that you don't know the answer .\n"
        "if there'is any inappropriate question, just say that you can't answer the question. \n"
        "please give the references from which line or paragraph regarding your answer in the context given. \n"
        "If the user ask you about generating images, sound. state that u cant do that, you can only generate text as an output. \n"
        "DO NOT EVER ANSWER QUESTIONS THAT IS NOT IN THE GIVEN CONTEXT!\n\n"
        "Context:\n{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain


def response_generator(text):
    """Simulates a typing effect for the bot's response."""
    for word in text.split():
        yield word + " "
        time.sleep(0.05)


def main():
    st.set_page_config(
        page_title="PMB ITPLN Chatbot",
        page_icon="🤖",
        initial_sidebar_state="expanded",
    )

    st.header(':sparkles: Mau nanya tentang PMB ITPLN :question:', divider='rainbow')
    st.subheader("Hallo, aku Elena. Temukan informasi seputar PMB ITPLN bersamaku.")

    # Process Documents on startup
    docs_path = "docs"
    if not os.path.exists(docs_path):
        os.makedirs(docs_path)

    pdf_docs = [os.path.join(docs_path, filename) for filename in os.listdir(docs_path) if filename.endswith('.pdf')]
    
    if pdf_docs:
        with st.spinner("Processing Documents..."):
            start_time = time.time()
            raw_text = get_pdf_text(pdf_docs)
            if raw_text:
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
            process_time = time.time() - start_time
            # st.info(f"PDF processed in {process_time:.2f} seconds.") # Optional: Hide in production
    else:
        st.warning("No PDF documents found in the 'docs' folder. Please add some to enable Q&A.")

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Kamu mau nanya apa?")
        ]

    # Display chat history
    for i, message in enumerate(st.session_state.chat_history):
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        with st.chat_message(role):
            if i == 0 and isinstance(message, AIMessage) and len(st.session_state.chat_history) == 1:
                # typing effect only for the first greeting when chat is empty
                st.write_stream(response_generator(message.content))
            else:
                st.markdown(message.content)

    # Handle User Input
    if prompt := st.chat_input("Tulis pertanyaanmu di sini..."):
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Process and display AI response
        with st.spinner("Elena is thinking..."):
            start_inference = time.time()

            rag_chain = get_conversational_chain()
            if rag_chain:
                try:
                    response = rag_chain.invoke({
                        "input": prompt,
                        "chat_history": st.session_state.chat_history
                    })
                    answer = response.get("answer", "Maaf, terjadi kesalahan saat memproses jawaban.")
                except Exception as e:
                    answer = f"Error generating response: {e}"
            else:
                 answer = "Vector store is not initialized. Please ensure documents are loaded."

            inference_time = time.time() - start_inference

        with st.chat_message("assistant"):
            st.markdown(answer)
            st.caption(f"Inference time: {inference_time:.2f} seconds.")

        # Update chat history
        st.session_state.chat_history.extend([
            HumanMessage(content=prompt),
            AIMessage(content=answer)
        ])


if __name__ == "__main__":
    main()