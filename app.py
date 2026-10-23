"""Athena AI Lite — Streamlit Web Application.

An educational, fully functional multi-agent RAG application designed to teach:
- PDF text extraction with PyMuPDF.
- Overlapping text chunking.
- Semantic vector embeddings with sentence-transformers.
- Vector search with FAISS CPU (Cosine Similarity).
- Grounded RAG answer synthesis with Google Gemini.
- Multi-agent coordination with DocumentAgent, WebAgent, and Supervisor.
- Transparent inspection via the "How the AI Works" learning panel.
"""

import time
from typing import List, Optional
import streamlit as st

import config
from agents.document_agent import DocumentAgent
from agents.supervisor import AgentSupervisor, SupervisorResult
from agents.web_agent import WebAgent
from core.chunker import TextChunk, chunk_document
from core.embeddings import LocalEmbeddings
from core.llm_client import GeminiLLMClient
from core.pdf_loader import PDFDocument, PDFExtractionError, extract_text_from_pdf
from core.retriever import Retriever
from core.vector_store import FAISSVectorStore
from core.web_search import WebSearchEngine

# Page configuration
st.set_page_config(
    page_title="Athena AI Lite — Educational Multi-Agent RAG",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --- Cached Resources for Performance ---
@st.cache_resource(show_spinner="Loading local embedding model (all-MiniLM-L6-v2)...")
def get_embeddings_model() -> LocalEmbeddings:
    """Loads and caches the local sentence-transformers embedding model."""
    try:
        return LocalEmbeddings(model_name=config.EMBEDDING_MODEL_NAME, mock_mode=False)
    except Exception as e:
        st.warning(f"Embedding model initialization note: {e}. Falling back to deterministic test mode.")
        return LocalEmbeddings(model_name=config.EMBEDDING_MODEL_NAME, mock_mode=True)


@st.cache_resource
def get_llm_client() -> GeminiLLMClient:
    """Initializes and caches the Gemini LLM client."""
    return GeminiLLMClient()


@st.cache_resource
def get_web_engine() -> WebSearchEngine:
    """Initializes and caches the Web Search Engine."""
    return WebSearchEngine()


# --- Session State Initialization ---
def init_session_state():
    """Initializes persistent application session state variables."""
    if "preferences" not in st.session_state:
        st.session_state.preferences = config.load_preferences()

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = FAISSVectorStore(
            embedding_model_name=config.EMBEDDING_MODEL_NAME,
            dimension=config.EMBEDDING_DIMENSION
        )

    if "retriever" not in st.session_state:
        embeddings = get_embeddings_model()
        st.session_state.retriever = Retriever(
            embeddings=embeddings,
            vector_store=st.session_state.vector_store
        )

    if "supervisor" not in st.session_state:
        doc_agent = DocumentAgent(retriever=st.session_state.retriever)
        web_agent = WebAgent(web_engine=get_web_engine())
        llm_client = get_llm_client()
        st.session_state.supervisor = AgentSupervisor(
            document_agent=doc_agent,
            web_agent=web_agent,
            llm_client=llm_client
        )

    if "active_doc" not in st.session_state:
        st.session_state.active_doc: Optional[PDFDocument] = None

    if "active_chunks" not in st.session_state:
        st.session_state.active_chunks: List[TextChunk] = []

    if "last_result" not in st.session_state:
        st.session_state.last_result: Optional[SupervisorResult] = None

    if "last_uploaded_name" not in st.session_state:
        st.session_state.last_uploaded_name = None


init_session_state()


# --- SIDEBAR: Preferences & System Controls ---
with st.sidebar:
    st.title("🎓 Athena AI Lite")
    st.caption("Educational Multi-Agent RAG Assistant")

    # API Mode Indicator Badge
    llm_configured = config.is_llm_configured()
    if llm_configured:
        st.success("🟢 **Live Gemini API Connected**", icon="✨")
    else:
        st.info("🟡 **Mock Mode (Educational)**\nNo `GEMINI_API_KEY` detected. Running with deterministic evidence synthesis.", icon="ℹ️")

    st.markdown("---")
    st.subheader("⚙️ Personalization Settings")

    prefs: config.UserPreferences = st.session_state.preferences

    # Learning Level
    learning_levels = ["Beginner", "Intermediate", "Advanced"]
    lvl_idx = learning_levels.index(prefs.learning_level) if prefs.learning_level in learning_levels else 0
    selected_level = st.selectbox(
        "Learning Level",
        learning_levels,
        index=lvl_idx,
        help="Adapts vocabulary, technical depth, and conceptual explanations."
    )

    # Answer Length
    length_options = ["Short", "Balanced", "Detailed"]
    len_idx = length_options.index(prefs.answer_length) if prefs.answer_length in length_options else 1
    selected_length = st.selectbox(
        "Answer Length",
        length_options,
        index=len_idx,
        help="Controls the verbosity and paragraph count of answers."
    )

    # Include Examples
    include_examples = st.checkbox(
        "Include Practical Examples",
        value=prefs.include_examples,
        help="Prompts the model to provide illustrative code or analogies."
    )

    # Live Web Search
    include_web_search = st.checkbox(
        "Include Live Web Search",
        value=prefs.include_web_search,
        help="Enables the supervisor to dispatch the Web Agent for live external research."
    )

    with st.expander("🛠️ Advanced Retrieval Hyperparameters"):
        top_k = st.slider(
            "Top-k Chunks to Retrieve",
            min_value=1,
            max_value=8,
            value=prefs.top_k,
            help="Number of nearest-neighbor chunks retrieved from FAISS."
        )
        chunk_size = st.slider(
            "Chunk Size (characters)",
            min_value=200,
            max_value=1000,
            value=prefs.chunk_size,
            step=50,
            help="Target length of each textual chunk."
        )
        chunk_overlap = st.slider(
            "Chunk Overlap (characters)",
            min_value=10,
            max_value=200,
            value=prefs.chunk_overlap,
            step=10,
            help="Overlapping characters between adjacent chunks."
        )

    # Update preferences in state and persist
    if (
        selected_level != prefs.learning_level or
        selected_length != prefs.answer_length or
        include_examples != prefs.include_examples or
        include_web_search != prefs.include_web_search or
        top_k != prefs.top_k or
        chunk_size != prefs.chunk_size or
        chunk_overlap != prefs.chunk_overlap
    ):
        st.session_state.preferences = config.UserPreferences(
            learning_level=selected_level,
            answer_length=selected_length,
            include_examples=include_examples,
            include_web_search=include_web_search,
            top_k=top_k,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        config.save_preferences(st.session_state.preferences)

    # Reset Preferences
    if st.button("🔄 Reset Preferences to Default", use_container_width=True):
        st.session_state.preferences = config.reset_preferences()
        st.rerun()

    st.markdown("---")
    st.caption("Athena AI Lite v1.0 • Pure Python & FAISS")


# --- MAIN PAGE LAYOUT ---
st.title("🎓 Athena AI Lite")
st.markdown(
    "A hands-on, transparent educational assistant demonstrating **PDF Extraction**, "
    "**Text Chunking**, **FAISS Vector Search**, **Grounded RAG**, and **Multi-Agent Coordination**."
)


# --- SECTION 1: PDF Upload and Ingestion ---
with st.container():
    st.subheader("1. Active Document Knowledge Base")

    col_up, col_stat = st.columns([1.5, 1])

    with col_up:
        uploaded_file = st.file_uploader(
            "Upload a university PDF (syllabus, lecture slides, course handbook)",
            type=["pdf"],
            help="Select a digital PDF with selectable text."
        )

        if uploaded_file is not None and uploaded_file.name != st.session_state.last_uploaded_name:
            file_bytes = uploaded_file.read()

            try:
                with st.spinner(f"Extracting text from '{uploaded_file.name}' using PyMuPDF..."):
                    pdf_doc = extract_text_from_pdf(file_bytes, filename=uploaded_file.name)

                if not pdf_doc.has_extractable_text:
                    st.error(pdf_doc.error_message)
                    st.session_state.active_doc = None
                    st.session_state.active_chunks = []
                    st.session_state.retriever.clear()
                else:
                    with st.spinner(f"Chunking and embedding into FAISS index..."):
                        chunks = st.session_state.retriever.index_document(
                            document=pdf_doc,
                            chunk_size=st.session_state.preferences.chunk_size,
                            chunk_overlap=st.session_state.preferences.chunk_overlap
                        )

                    st.session_state.active_doc = pdf_doc
                    st.session_state.active_chunks = chunks
                    st.session_state.last_uploaded_name = uploaded_file.name
                    st.success(
                        f"✅ Successfully ingested '{uploaded_file.name}'! "
                        f"Extracted {pdf_doc.total_pages} page(s) and generated {len(chunks)} chunks in FAISS."
                    )

            except PDFExtractionError as e:
                st.error(f"❌ PDF Ingestion Failed: {e}")
                st.session_state.active_doc = None
                st.session_state.active_chunks = []
                st.session_state.retriever.clear()

    with col_stat:
        if st.session_state.active_doc is not None:
            doc = st.session_state.active_doc
            st.markdown(
                f"""
                <div style="padding:15px; border-radius:8px; background-color:#f0f2f6; color:#1a1a1a;">
                    <b>Active Knowledge Base:</b> {doc.filename}<br>
                    <b>Total Pages:</b> {doc.total_pages}<br>
                    <b>Total Characters:</b> {doc.total_characters:,}<br>
                    <b>Vectors in FAISS:</b> {st.session_state.vector_store.total_vectors}<br>
                    <b>Status:</b> Ready for RAG queries
                </div>
                """,
                unsafe_allow_html=True
            )
            if st.button("🗑️ Unload Document", use_container_width=True):
                st.session_state.active_doc = None
                st.session_state.active_chunks = []
                st.session_state.last_uploaded_name = None
                st.session_state.retriever.clear()
                st.session_state.last_result = None
                st.rerun()
        else:
            st.info("No active document. Please upload a university PDF above.")


# --- SECTION 2: Chat & Inquiry ---
st.markdown("---")
st.subheader("2. Ask a Question")

question_input = st.text_input(
    "Enter your academic inquiry:",
    placeholder="e.g., What are the key topics in Week 3, or when is the midterm exam?",
    disabled=(st.session_state.active_doc is None and not st.session_state.preferences.include_web_search)
)

col_ask, col_clear = st.columns([1, 5])
with col_ask:
    submit_btn = st.button("🚀 Ask Athena", type="primary", use_container_width=True)

if submit_btn and question_input.strip():
    with st.spinner("Supervisor orchestrating specialist agents..."):
        result = st.session_state.supervisor.run(
            question=question_input.strip(),
            preferences=st.session_state.preferences
        )
        st.session_state.last_result = result


# --- SECTION 3: Answer & Evidence Citations ---
if st.session_state.last_result is not None:
    res: SupervisorResult = st.session_state.last_result

    st.markdown("---")
    st.subheader("3. Grounded Answer & Verifiable Citations")

    # Real Activity Trace
    st.markdown("**Agent Execution Trace:**")
    trace_cols = st.columns(len(res.activity_trace))
    for i, step in enumerate(res.activity_trace):
        with trace_cols[i]:
            badge = "✅" if step.status == "completed" else "⏭️"
            st.caption(f"{badge} **{step.step_name}**")

    # Main Answer Box
    st.markdown(f"### Answer ({res.decision.reasoning})")
    st.markdown(res.answer)

    # Citations
    col_pdf_cit, col_web_cit = st.columns(2)

    with col_pdf_cit:
        st.markdown("#### 📄 Document Citations (PDF)")
        if res.document_citations:
            for cit in res.document_citations:
                with st.container():
                    st.markdown(
                        f"""
                        <div style="border-left: 3px solid #2e7d32; padding-left: 10px; margin-bottom: 10px;">
                            <span style="background-color:#e8f5e9; color:#2e7d32; padding:2px 8px; border-radius:4px; font-weight:bold; font-size:12px;">
                                Page {cit.page_number}
                            </span>
                            <span style="font-size:12px; color:#666; margin-left:8px;">
                                Cosine Similarity: <code>{cit.score:.4f}</code>
                            </span>
                            <div style="font-size:13px; font-style:italic; margin-top:4px;">
                                "{cit.text_snippet}"
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
        else:
            st.caption("No document citations for this response.")

    with col_web_cit:
        st.markdown("#### 🌐 Web Citations (Live Search)")
        if res.web_citations:
            for item in res.web_citations:
                mock_label = " [MOCK FIXTURE]" if item.get("is_mock") == "True" else ""
                st.markdown(
                    f"- **[{item['title']}]({item['url']})**{mock_label}\n"
                    f"  <span style='font-size:12px; color:#555;'>{item['snippet'][:130]}...</span>",
                    unsafe_allow_html=True
                )
        else:
            st.caption("Web search was not invoked or returned 0 results.")


# --- SECTION 4: EDUCATIONAL LEARNING AND DEBUG PANEL ---
st.markdown("---")
with st.expander("🔍 **How the AI Works — Under the Hood (Educational Inspection)**", expanded=False):
    st.markdown(
        "This debug panel inspects intermediate data structures at every stage of the pipeline "
        "so you can understand how raw text becomes indexed vectors and grounded answers."
    )

    tab_pdf, tab_vectors, tab_agents = st.tabs([
        "📄 1. PDF Extraction & Chunks",
        "🧠 2. FAISS Vector Search & Cosine Metric",
        "🤖 3. Multi-Agent Supervisor Details"
    ])

    with tab_pdf:
        st.markdown("### Stage 1: Document Processing")
        if st.session_state.active_doc is not None:
            doc = st.session_state.active_doc
            st.write({
                "Document Name": doc.filename,
                "Total Pages": doc.total_pages,
                "Total Characters Extracted": doc.total_characters,
                "Has Extractable Text": doc.has_extractable_text,
                "Total Chunks Created": len(st.session_state.active_chunks)
            })

            st.markdown("**Raw Text Preview (First Page):**")
            st.code(doc.get_text_preview(max_chars=600), language="text")

            st.markdown("**Generated Text Chunks:**")
            st.caption("Notice how each chunk tracks its own chunk_id, page_number, and character span.")
            for c in st.session_state.active_chunks[:4]:
                st.text(f"ID: {c.chunk_id} | Page: {c.page_number} | Length: {c.char_count} chars")
                st.text(f"Text: {c.text[:120]}...")
                st.markdown("---")
        else:
            st.info("Upload a PDF to inspect raw page text and chunk boundaries.")

    with tab_vectors:
        st.markdown("### Stage 2: Embeddings & FAISS Index")
        st.markdown(
            """
            - **Embedding Model:** `all-MiniLM-L6-v2` (Sentence Transformers)
            - **Vector Dimensions:** `384` numbers per vector
            - **Similarity Metric:** **Cosine Similarity**
              - Vectors are $L_2$-normalized to length 1.0 ($\|v\|_2 = 1.0$).
              - Because vectors have unit length, the Dot Product (Inner Product) exactly equals the Cosine Similarity:
                $$\\text{Cosine}(A, B) = \\frac{A \\cdot B}{\|A\| \|B\|} = A \\cdot B$$
            - **FAISS Index Type:** `IndexFlatIP` (Flat Inner Product index)
            """
        )
        st.write({
            "Embedding Model": config.EMBEDDING_MODEL_NAME,
            "Dimensionality": config.EMBEDDING_DIMENSION,
            "Similarity Metric": config.SIMILARITY_METRIC,
            "Total Vectors Indexed": st.session_state.vector_store.total_vectors
        })

        if st.session_state.last_result and st.session_state.last_result.document_citations:
            st.markdown("**Latest Query Retrieved Vector Ranking:**")
            results_table = [
                {
                    "Rank": i + 1,
                    "Chunk ID": c.citation_id,
                    "Page": c.page_number,
                    "Cosine Similarity Score": f"{c.score:.4f}",
                    "Snippet": c.text_snippet[:80] + "..."
                }
                for i, c in enumerate(st.session_state.last_result.document_citations)
            ]
            st.table(results_table)

    with tab_agents:
        st.markdown("### Stage 3: Multi-Agent Supervisor Routing")
        if st.session_state.last_result is not None:
            res = st.session_state.last_result
            st.write({
                "Supervisor Decision": res.decision.reasoning,
                "Document Agent Activated": res.decision.use_document_agent,
                "Web Agent Activated": res.decision.use_web_agent,
                "Sufficient Evidence Found": res.sufficient_evidence,
                "Total Pipeline Execution Time": f"{res.execution_time_seconds:.3f} seconds",
                "LLM Model Name": res.model_name or "gemini-2.5-flash",
                "Is Mock Mode": res.is_mock
            })

            st.markdown("**Detailed Step-by-Step Activity Trace:**")
            for step in res.activity_trace:
                st.markdown(f"- **{step.step_name}** (`{step.status}`): {step.detail}")
        else:
            st.info("Submit a question to inspect multi-agent supervisor decisions.")
