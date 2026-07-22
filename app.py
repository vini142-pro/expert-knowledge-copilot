"""
Expert Knowledge Copilot - Industrial Knowledge Intelligence
ET AI Hackathon 2026 - Problem Statement #8

A RAG-powered conversational AI that ingests industrial documents
(safety procedures, maintenance manuals, inspection reports) and answers
operational queries with source citations.
"""

import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()

st.set_page_config(page_title="Industrial Knowledge Copilot", page_icon="🏭", layout="wide")

# ---------- SESSION STATE ----------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "doc_names" not in st.session_state:
    st.session_state.doc_names = []

# ---------- SIDEBAR: API KEY + DOC UPLOAD ----------
with st.sidebar:
    st.title("🏭 Knowledge Copilot")
    st.caption("AI for Industrial Knowledge Intelligence")

    api_key = st.text_input("Groq API Key", type="password", value=os.getenv("GROQ_API_KEY", ""))
    st.caption("Get a free key instantly at console.groq.com — no card needed")
    if api_key:
        os.environ["GROQ_API_KEY"] = api_key

    st.divider()
    st.subheader("📄 Upload Documents")
    st.caption("Safety SOPs, maintenance manuals, inspection reports, P&IDs (as text/PDF)")

    uploaded_files = st.file_uploader(
        "Upload PDF documents", type=["pdf"], accept_multiple_files=True
    )

    if st.button("⚙️ Build Knowledge Base", use_container_width=True, type="primary"):
        if not api_key:
            st.error("Please enter your Groq API key first.")
        elif not uploaded_files:
            st.error("Please upload at least one PDF.")
        else:
            with st.spinner("Ingesting documents, chunking, and building vector index..."):
                all_docs = []
                doc_names = []
                for uploaded_file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name

                    loader = PyPDFLoader(tmp_path)
                    docs = loader.load()
                    for d in docs:
                        d.metadata["source"] = uploaded_file.name
                    all_docs.extend(docs)
                    doc_names.append(uploaded_file.name)
                    os.unlink(tmp_path)

                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                chunks = splitter.split_documents(all_docs)

                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                vectorstore = FAISS.from_documents(chunks, embeddings)

                st.session_state.vectorstore = vectorstore
                st.session_state.doc_names = doc_names
                st.session_state.chat_history = []

            st.success(f"Knowledge base built from {len(doc_names)} document(s), {len(chunks)} chunks indexed.")

    if st.session_state.doc_names:
        st.divider()
        st.subheader("📚 Indexed Documents")
        for name in st.session_state.doc_names:
            st.write(f"• {name}")

# ---------- MAIN AREA ----------
st.title("Expert Knowledge Copilot")
st.caption("Ask operational, maintenance, or safety questions across your document corpus — with source citations.")

if not st.session_state.vectorstore:
    st.info("👈 Upload documents and build the knowledge base from the sidebar to get started.")
    st.markdown("""
    **Try uploading:**
    - A safety SOP / permit-to-work procedure
    - An equipment maintenance manual
    - An inspection or compliance report

    **Then ask questions like:**
    - "What is the procedure for confined space entry?"
    - "What are the maintenance intervals for this equipment?"
    - "What safety precautions apply near hazardous gas zones?"
    """)
else:
    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📎 Sources"):
                    for s in msg["sources"]:
                        st.markdown(f"**{s['source']}** (page {s['page']})")
                        st.caption(s["snippet"])

    query = st.chat_input("Ask a question about your documents...")

    if query:
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant context and generating answer..."):
                retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})

                prompt_template = PromptTemplate(
                    template="""You are an Industrial Knowledge Copilot. Answer the operational/safety/maintenance
question using ONLY the context provided below. Be precise and factual. If the context doesn't
contain the answer, say so clearly instead of guessing.

Context:
{context}

Question: {question}

Answer:""",
                    input_variables=["context", "question"],
                )

                llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    retriever=retriever,
                    chain_type="stuff",
                    chain_type_kwargs={"prompt": prompt_template},
                    return_source_documents=True,
                )

                result = qa_chain.invoke({"query": query})
                answer = result["result"]
                source_docs = result["source_documents"]

                sources = []
                for doc in source_docs:
                    sources.append({
                        "source": doc.metadata.get("source", "unknown"),
                        "page": doc.metadata.get("page", "?"),
                        "snippet": doc.page_content[:200] + "...",
                    })

                st.markdown(answer)
                if sources:
                    with st.expander("📎 Sources"):
                        for s in sources:
                            st.markdown(f"**{s['source']}** (page {s['page']})")
                            st.caption(s["snippet"])

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })
