import streamlit as st
from langchain_community.document_loaders import PyPDFLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
import os, traceback, tempfile

# --- 1. RESET LOGIC ---
# This function clears the cache and the chat history whenever a new PDF is uploaded.
def reset_state():
    st.session_state.messages = []
    st.cache_resource.clear()

st.set_page_config(page_title="PDF Chatbot", page_icon="🤖")

# --- 2. API KEY SETUP ---
if "GOOGLE_API_KEY" not in st.secrets:
    st.error("ERROR: GOOGLE_API_KEY secret is not set in Streamlit Secrets.")
    st.stop()
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# --- 3. RAG CHAIN (CACHED) ---
@st.cache_resource(show_spinner="Processing your PDF...")
def get_rag_chain(_uploaded_file):
    if _uploaded_file is None:
        return None
    try:
        # Save uploaded file to temp
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(_uploaded_file.getvalue())
            pdf_path = tmp_file.name

        loader = PyPDFLoader(pdf_path)
        chunks = loader.load_and_split(RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200))
        
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") 
        db = FAISS.from_documents(chunks, embedding=embeddings) 
        
        # --- MODEL UPDATED TO 2026 STABLE VERSION ---
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.3)
        
        prompt = ChatPromptTemplate.from_template("""
        You are a helpful pdf assistant.
        Answer the question using the 'Context' provided.
        Context: {context}
        Question: {input}
        Answer:
        """)
        
        combine_docs_chain = create_stuff_documents_chain(llm, prompt)
        return create_retrieval_chain(db.as_retriever(), combine_docs_chain)

    except Exception as e:
        st.error(f"Processing Error: {e}")
        return None
    finally:
        if 'pdf_path' in locals() and os.path.exists(pdf_path):
            os.remove(pdf_path)

# --- 4. SIDEBAR ---
with st.sidebar:
    st.header("1. Upload Your PDF File")
    # THE FIX: 'on_change' calls reset_state when a new file is uploaded
    uploaded_file = st.file_uploader(
        "Upload PDF", 
        type="pdf", 
        on_change=reset_state, 
        key="pdf_uploader"
    )
    
    st.write("---")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- 5. CHAT INTERFACE ---
st.title("📚 PDF Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Show history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if uploaded_file:
    rag_chain = get_rag_chain(uploaded_file)
    
    if rag_chain:
        if user_question := st.chat_input("Ask a question about this PDF:"):
            st.session_state.messages.append({"role": "user", "content": user_question})
            with st.chat_message("user"):
                st.markdown(user_question)
            
            with st.chat_message("assistant"):
                with st.spinner("Gemini is thinking..."):
                    try:
                        response = rag_chain.invoke({"input": user_question})
                        answer = response.get("answer", "⚠️ No answer found.")
                        st.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        st.error(f"Chain Error: {e}")
else:
    st.info("Please upload a PDF file in the sidebar to begin.")
