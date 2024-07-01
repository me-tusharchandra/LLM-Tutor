import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile

# Load environment variables
load_dotenv()

# Initialize the Google PaLM model
llm = GoogleGenerativeAI(model="models/text-bison-001", temperature=0.2)

# Load embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load the vector store
@st.cache_resource
def load_vectorstore():
    return Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

loaded_vectorstore = load_vectorstore()

# Create a custom prompt template
prompt_template = """You are an AI tutor specialized in large language models and related AI topics. 
Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}

Question: {question}
AI Tutor:"""

PROMPT = PromptTemplate.from_template(prompt_template)

# Create a retrieval QA chain
@st.cache_resource
def create_qa_chain(_vectorstore):
    retriever = _vectorstore.as_retriever()
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )
    return rag_chain

qa = create_qa_chain(loaded_vectorstore)

# Function to process uploaded PDF
def process_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    loader = PyPDFLoader(tmp_file_path)
    data = loader.load()

    # Split the documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(data)

    # Add documents to the vectorstore
    loaded_vectorstore.add_documents(docs)

    # Remove the temporary file
    os.unlink(tmp_file_path)

    return f"Added {len(docs)} document chunks to the knowledge base."

# Streamlit app
st.title("LLM Tutor Chatbot")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF to add to the knowledge base", type="pdf")
if uploaded_file is not None:
    with st.spinner("Processing PDF..."):
        result = process_pdf(uploaded_file)
    st.success(result)
    # Recreate the QA chain with the updated vectorstore
    qa = create_qa_chain(loaded_vectorstore)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What would you like to know about LLMs?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get response from QA chain
    with st.spinner("Thinking..."):
        response = qa.invoke(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
        st.markdown("**Note:** Source documents are not automatically provided in this version.")

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Optional: Add a way to view the contents of the vector store
if st.button("View Vector Store Contents"):
    st.write("Vector Store Contents:")
    results = loaded_vectorstore.similarity_search("", k=5)  # Get 5 random documents
    for doc in results:
        st.write(f"Content: {doc.page_content[:100]}...")
        st.write(f"Metadata: {doc.metadata}")
        st.write("---")