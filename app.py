import streamlit as st
import tempfile
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CourseMate AI",
    page_icon="🎓",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0f0c29, #1a1a3e, #0f0c29);
    min-height: 100vh;
}

header[data-testid="stHeader"] {
    background: transparent;
}

/* Hero */
.hero {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
}
.hero-badge {
    display: inline-block;
    background: rgba(139, 92, 246, 0.15);
    border: 1px solid rgba(139, 92, 246, 0.4);
    color: #c4b5fd;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    padding: 0.3rem 0.9rem;
    border-radius: 999px;
    margin-bottom: 1rem;
}
.hero-title {
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #ffffff 30%, #a78bfa 80%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.15;
    margin: 0 0 0.6rem;
}
.hero-sub {
    color: #94a3b8;
    font-size: 1rem;
    font-weight: 300;
    margin: 0;
}

/* Divider */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(139,92,246,0.4), transparent);
    margin: 1.5rem 0;
}

/* PDF status pill */
.pdf-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.35);
    color: #6ee7b7;
    font-size: 0.82rem;
    font-weight: 600;
    padding: 0.35rem 0.9rem;
    border-radius: 999px;
    margin-bottom: 1rem;
}

/* Chat messages */
.msg-user {
    display: flex;
    justify-content: flex-end;
    margin: 0.75rem 0;
}
.msg-ai {
    display: flex;
    justify-content: flex-start;
    margin: 0.75rem 0;
}
.bubble-user {
    background: linear-gradient(135deg, #7c3aed, #5b21b6);
    color: #fff;
    padding: 0.75rem 1.1rem;
    border-radius: 18px 18px 4px 18px;
    max-width: 78%;
    font-size: 0.93rem;
    line-height: 1.55;
    box-shadow: 0 4px 15px rgba(124, 58, 237, 0.35);
}
.bubble-ai {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    color: #e2e8f0;
    padding: 0.75rem 1.1rem;
    border-radius: 18px 18px 18px 4px;
    max-width: 78%;
    font-size: 0.93rem;
    line-height: 1.55;
    backdrop-filter: blur(8px);
}
.avatar {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.85rem;
    flex-shrink: 0;
    margin-top: 2px;
}
.avatar-ai {
    background: linear-gradient(135deg, #7c3aed, #4f46e5);
    margin-right: 0.6rem;
}
.avatar-user {
    background: rgba(255,255,255,0.1);
    margin-left: 0.6rem;
}

/* Chat container */
.chat-container {
    max-height: 420px;
    overflow-y: auto;
    padding: 0.5rem 0.2rem;
    scrollbar-width: thin;
    scrollbar-color: rgba(139,92,246,0.3) transparent;
}

/* Welcome card */
.welcome-card {
    background: rgba(139, 92, 246, 0.07);
    border: 1px solid rgba(139, 92, 246, 0.2);
    border-radius: 14px;
    padding: 1.5rem 1.8rem;
    text-align: center;
    margin: 1rem 0 1.5rem;
}
.welcome-card h4 {
    color: #c4b5fd;
    font-size: 1.05rem;
    font-weight: 600;
    margin: 0 0 0.5rem;
}
.welcome-card p {
    color: #94a3b8;
    font-size: 0.88rem;
    margin: 0;
    line-height: 1.6;
}

/* File uploader */
[data-testid="stFileUploaderDropzone"] {
    background: rgba(139, 92, 246, 0.06) !important;
    border: 2px dashed rgba(139, 92, 246, 0.35) !important;
    border-radius: 16px !important;
    color: #94a3b8 !important;
}

/* Input */
.stTextInput > div > div > input {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(139,92,246,0.35) !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 0.93rem !important;
    padding: 0.65rem 1rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: rgba(139,92,246,0.7) !important;
    box-shadow: 0 0 0 3px rgba(139,92,246,0.15) !important;
}
.stTextInput > div > div > input::placeholder {
    color: #64748b !important;
}
.stTextInput > div > div > input:disabled {
    opacity: 0.4 !important;
    cursor: not-allowed !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #5b21b6) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Sora', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 0.65rem 1.6rem !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 15px rgba(124, 58, 237, 0.3) !important;
    width: 100% !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(124, 58, 237, 0.45) !important;
}
.stButton > button:disabled {
    opacity: 0.4 !important;
    transform: none !important;
    cursor: not-allowed !important;
}

/* Clear / secondary button */
.clear-btn > button {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    box-shadow: none !important;
    font-size: 0.8rem !important;
    font-weight: 400 !important;
    color: #94a3b8 !important;
    padding: 0.4rem 1rem !important;
}
.clear-btn > button:hover {
    background: rgba(255,255,255,0.09) !important;
    transform: none !important;
    box-shadow: none !important;
}

/* Footer */
.footer {
    text-align: center;
    color: #475569;
    font-size: 0.78rem;
    padding: 1.5rem 0 0.5rem;
    font-family: 'JetBrains Mono', monospace;
}
</style>
""", unsafe_allow_html=True)


# ── Shared LLM & prompt (cached — never change between uploads) ───────────────
@st.cache_resource
def load_llm_and_prompt():
    llm = ChatMistralAI(model="mistral-small-2506")
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
"""),
        ("human", """Context:
{context}

Question:
{question}
""")
    ])
    embedding_model = MistralAIEmbeddings(model="mistral-embed")
    return llm, prompt, embedding_model


# ── Build retriever from an uploaded PDF bytes ────────────────────────────────
def build_retriever_from_pdf(uploaded_file, embedding_model):
    # Write to a temp file so PyPDFLoader can read it
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
    finally:
        os.unlink(tmp_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(docs)

    # In-memory Chroma — no persist_directory, scoped to this session
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5},
    )
    return retriever


# ── Session state defaults ────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None
if "input_key" not in st.session_state:
    st.session_state.input_key = 0


# ── Load shared components ────────────────────────────────────────────────────
llm, prompt, embedding_model = load_llm_and_prompt()


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">🎓 RAG-Powered Learning Assistant</div>
    <div class="hero-title">CourseMate AI</div>
    <p class="hero-sub">Upload your PDF and ask anything — get precise, document-grounded answers.</p>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)


# ── PDF Upload ────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    label="Upload your PDF",
    type=["pdf"],
    label_visibility="collapsed",
    help="Upload any PDF — lecture notes, textbooks, research papers, etc.",
)

if uploaded_file is not None:
    # Only reprocess if it's a new file
    if uploaded_file.name != st.session_state.pdf_name:
        with st.spinner(f"Processing **{uploaded_file.name}**… chunking & embedding, please wait."):
            st.session_state.retriever = build_retriever_from_pdf(uploaded_file, embedding_model)
            st.session_state.pdf_name = uploaded_file.name
            st.session_state.messages = []  # clear chat for new doc
        st.success(f"✅ **{uploaded_file.name}** is ready — start asking questions below!")

# Show active PDF pill
if st.session_state.pdf_name:
    st.markdown(
        f'<div style="text-align:center"><span class="pdf-pill">📄 {st.session_state.pdf_name}</span></div>',
        unsafe_allow_html=True,
    )

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)


# ── Chat history ──────────────────────────────────────────────────────────────
pdf_ready = st.session_state.retriever is not None

if not pdf_ready:
    st.markdown("""
    <div class="welcome-card">
        <h4>👆 Upload a PDF to get started</h4>
        <p>Once you upload your course material, lecture notes, or any PDF document, you can ask questions and CourseMate AI will answer using only that document.</p>
    </div>
    """, unsafe_allow_html=True)
elif not st.session_state.messages:
    st.markdown("""
    <div class="welcome-card">
        <h4>📖 Your document is ready!</h4>
        <p>Ask me anything — concepts, summaries, definitions, or specific topics from your uploaded PDF.</p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="msg-user">
                <div class="bubble-user">{msg["content"]}</div>
                <div class="avatar avatar-user">👤</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="msg-ai">
                <div class="avatar avatar-ai">🤖</div>
                <div class="bubble-ai">{msg["content"]}</div>
            </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ── Input row ─────────────────────────────────────────────────────────────────
col1, col2 = st.columns([5, 1])

with col1:
    user_input = st.text_input(
        label="query",
        placeholder="Ask a question about your PDF…" if pdf_ready else "Upload a PDF first…",
        label_visibility="collapsed",
        key=f"user_query_{st.session_state.input_key}",
        disabled=not pdf_ready,
    )

with col2:
    send = st.button("Send", use_container_width=True, disabled=not pdf_ready)


# ── Handle query ──────────────────────────────────────────────────────────────
if send and user_input.strip() and pdf_ready:
    query = user_input.strip()
    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.input_key += 1  # forces input widget to re-render empty

    with st.spinner("Thinking…"):
        docs = st.session_state.retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in docs])
        final_prompt = prompt.invoke({"context": context, "question": query})
        response = llm.invoke(final_prompt)
        answer = response.content

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()


# ── Clear conversation ────────────────────────────────────────────────────────
if st.session_state.messages:
    st.markdown("")
    st.markdown('<div class="clear-btn">', unsafe_allow_html=True)
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="footer">CourseMate AI · Powered by MistralAI + LangChain + ChromaDB</div>',
    unsafe_allow_html=True,
)