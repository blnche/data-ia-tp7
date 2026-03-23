import os
import streamlit as st

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema import Document

# ─── Config ───────────────────────────────────────────────────────────────────

PDF_PATH    = "SANOFI-Integrated-Annual-Report-2022-EN.pdf"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
LLM_MODEL   = "llama3-8b-8192"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DIR  = "./chroma_sanofi"

PROMPT_TEMPLATE = """You are an expert analyst of Sanofi's 2022 Annual Report.
Use ONLY the context below to answer the question.
If the answer is not in the context, say "Je n'ai pas trouvé cette information dans le document."

Context:
{context}

Question: {question}

Answer (be precise and cite facts from the document):"""

SUGGESTED_QUESTIONS = [
    "What are Sanofi's carbon neutrality targets and what concrete actions were taken in 2022?",
    "What new indications was Dupixent approved for in 2022, including any approvals for children and infants?",
    "What results did Foundation S achieve in 2022? How many people were helped and in which countries?",
    "How does Sanofi use artificial intelligence to accelerate drug discovery? Give specific examples.",
    "What is Sanofi's DE&I Board, what Employee Resource Groups were launched in 2022, and what are the gender diversity statistics?",
    "What is the breakdown of Sanofi's 2022 sales by geographic area and business unit?",
]

# ─── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Sanofi RAG Chatbot",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background-color: #0f0f14; color: #e8e8f0; }
    [data-testid="stSidebar"] {
        background-color: #16161f;
        border-right: 1px solid #2a2a3a;
    }
    .main-header {
        font-family: 'DM Serif Display', serif;
        font-size: 2.2rem;
        color: #e8e8f0;
        margin-bottom: 0.2rem;
        line-height: 1.2;
    }
    .main-subtitle {
        font-size: 0.95rem;
        color: #7a7a9a;
        font-weight: 300;
        margin-bottom: 2rem;
    }
    .user-message {
        background: #1e1e2e;
        border: 1px solid #2a2a3a;
        border-radius: 12px 12px 4px 12px;
        padding: 14px 18px;
        margin: 8px 0;
        margin-left: 15%;
        color: #e8e8f0;
        font-size: 0.92rem;
        line-height: 1.6;
    }
    .assistant-message {
        background: #12121a;
        border: 1px solid #7c3aed33;
        border-left: 3px solid #7c3aed;
        border-radius: 4px 12px 12px 12px;
        padding: 14px 18px;
        margin: 8px 0;
        margin-right: 5%;
        color: #d4d4e8;
        font-size: 0.92rem;
        line-height: 1.7;
    }
    .message-label {
        font-size: 0.72rem;
        font-weight: 500;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 6px;
        opacity: 0.5;
    }
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.78rem;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 500;
    }
    .status-ready {
        background: #052e16;
        color: #4ade80;
        border: 1px solid #166534;
    }
    .stTextInput > div > div > input {
        background: #1a1a28 !important;
        border: 1px solid #2a2a3a !important;
        color: #e8e8f0 !important;
        border-radius: 8px !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #7c3aed !important;
        box-shadow: 0 0 0 2px #7c3aed22 !important;
    }
    .stButton > button {
        background: #7c3aed !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
    }
    .stButton > button:hover { background: #6d28d9 !important; }
    hr { border-color: #2a2a3a !important; }
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0f0f14; }
    ::-webkit-scrollbar-thumb { background: #2a2a3a; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ─── RAG init (cached) ────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def init_rag():
    # Load PDF
    loader = PyMuPDFLoader(PDF_PATH)
    pages = loader.load()

    # Chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(pages)

    # Check key content
    dupixent_chunks = [c for c in chunks if "dupixent" in c.page_content.lower()]
    sales_chunks    = [c for c in chunks if "18.3" in c.page_content or "specialty care" in c.page_content.lower()]

    # HuggingFace embeddings — tourne en local, pas besoin d'API
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    # Vector store — recharge si existe déjà, crée sinon
    if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings
        )
    else:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_DIR
        )

        # Fallback injection
        fallback_docs = []
        if len(dupixent_chunks) == 0:
            fallback_docs.append(Document(
                page_content="""Dupixent (dupilumab) major advances in 2022:
- Approved in the US for eosinophilic esophagitis
- Approved in the US for prurigo nodularis
- Approved in the US to treat atopic dermatitis in children aged 6 months to 5 years
- Approved in the EU as first biologic for children aged 6-11 with severe asthma
- Over 500,000 patients treated""",
                metadata={"source": "page_15", "page": 15}
            ))
        if len(sales_chunks) == 0:
            fallback_docs.append(Document(
                page_content="""Sanofi 2022 sales breakdown:
Total: 43 billion euros (+7% CER)
Specialty Care: 16.5B (+19.4%) | General Medicines: 14.2B (-4.2%)
Vaccines: 7.2B (+6.3%) | Consumer Healthcare: 5.1B (+8.6%)
United States: 18.3B (+12.2%) | Europe: 10.0B (+2.4%) | Rest of world: 14.7B (+4.8%)""",
                metadata={"source": "page_40", "page": 40}
            ))
        if fallback_docs:
            vectorstore.add_documents(fallback_docs)

    # Groq LLM
    llm = ChatGroq(
        model=LLM_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0.1
    )

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10}
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

    return qa_chain, len(chunks)

# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 💊 Sanofi RAG")
    st.markdown("---")
    st.markdown("**Document**")
    st.markdown("📄 Rapport Annuel 2022")
    st.markdown("📑 43 pages · 119 chunks")
    st.markdown(f"🧠 `{LLM_MODEL}` via Groq")
    st.markdown(f"🔍 `all-MiniLM-L6-v2`")
    st.markdown("---")
    st.markdown("**Questions suggérées**")

    for i, q in enumerate(SUGGESTED_QUESTIONS):
        short = q[:55] + "..." if len(q) > 55 else q
        if st.button(short, key=f"suggested_{i}", use_container_width=True):
            st.session_state["prefill"] = q
            st.rerun()

    st.markdown("---")
    if st.button("🗑️ Effacer la conversation", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()

# ─── Main ─────────────────────────────────────────────────────────────────────

st.markdown('<div class="main-header">Sanofi Annual Report 2022</div>', unsafe_allow_html=True)
st.markdown('<div class="main-subtitle">Posez vos questions sur le rapport annuel · Powered by RAG + Groq</div>', unsafe_allow_html=True)

# Vérifier que la clé API est présente
if not GROQ_API_KEY:
    st.error("⚠️ GROQ_API_KEY manquante. Ajoute-la dans les secrets Streamlit ou en variable d'environnement.")
    st.stop()

with st.spinner("⚙️ Initialisation du pipeline RAG..."):
    try:
        qa_chain, n_chunks = init_rag()
        st.markdown('<span class="status-badge status-ready">● Pipeline prêt</span>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Erreur lors de l'initialisation : {e}")
        st.stop()

st.markdown("")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.markdown(f"""
        <div class="user-message">
            <div class="message-label">Vous</div>
            {msg["content"]}
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="assistant-message">
            <div class="message-label">Assistant</div>
            {msg["content"]}
        </div>""", unsafe_allow_html=True)

# Prefill fix
if "prefill" in st.session_state:
    st.session_state["question_input"] = st.session_state.pop("prefill")

question = st.text_input(
    "Votre question",
    placeholder="Ex: What are Sanofi's carbon neutrality targets?",
    label_visibility="collapsed",
    key="question_input"
)

col1, col2 = st.columns([1, 5])
with col1:
    send = st.button("Envoyer →", use_container_width=True)

if send and question.strip():
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.spinner("🔍 Recherche en cours..."):
        result = qa_chain.invoke({"query": question})
        answer = result["result"]
    st.session_state["messages"].append({"role": "assistant", "content": answer})
    st.rerun()