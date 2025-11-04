
import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.callbacks.base import BaseCallbackHandler
from langchain_classic.schema import Document
from langchain_community.tools.tavily_search import TavilySearchResults

# ========================
# STREAMING HANDLER
# ========================

class StreamHandler(BaseCallbackHandler):
    """Stream tokens to Streamlit as they are generated."""

    def __init__(self, container):
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text)


# ========================
# CONFIG
# ========================

st.set_page_config(page_title="📄 Chat with Document (Streaming + Memory + Citations)", layout="wide")

# Load configuration from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "faiss_index")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "openai")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))
NUMBER_OF_RETRIEVAL = int(os.getenv("NUMBER_OF_RETRIEVAL", "3"))
NUMBER_OF_SEARCH = int(os.getenv("NUMBER_OF_SEARCH", "3"))

# ========================
# HELPERS
# ========================

def load_document(file):
    """Load a PDF or text file with page metadata preserved."""
    if file.name.endswith(".pdf"):
        loader = PyPDFLoader(file.name)
        pages = loader.load_and_split()
        docs = []
        for i, page in enumerate(pages):
            # Keep the original metadata and add page number
            meta = page.metadata or {}
            meta["page_number"] = i + 1
            docs.append(Document(page_content=page.page_content, metadata=meta))
        print(docs)
        return docs
    else:
        loader = TextLoader(file.name)
        docs = loader.load()
        for doc in docs:
            doc.metadata["page_number"] = 1
        return docs


def get_vectorstore(docs):
    """Create or load FAISS index."""
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    if os.path.exists(FAISS_INDEX_PATH):
        vectordb = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        vectordb = FAISS.from_documents(docs, embeddings, normalize_L2=True)  # cosine distance
        vectordb.save_local(FAISS_INDEX_PATH)
    return vectordb


def web_search(query):
    """Perform web search using Tavily."""
    try:
        search = TavilySearchResults(
            tavily_api_key=TAVILY_API_KEY,
            max_results=NUMBER_OF_SEARCH
        )
        results = search.invoke(query)
        return results
    except Exception as e:
        return [{"error": str(e)}]


# ========================
# UI
# ========================

st.title("💬 Chat with Your Document")
st.caption("Streaming responses, memory, citations, and web search powered by Tavily")

# Search mode selector
search_mode = st.radio(
    "Choose mode:",
    ["Document Only", "Document + Web Search", "Web Search Only"],
    horizontal=True
)

uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

# Initialize memory & history
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
if "history" not in st.session_state:
    st.session_state.history = []

# ========================
# MAIN LOGIC
# ========================

# Handle document processing
vectordb = None
if uploaded_file and OPENAI_API_KEY:
    with st.spinner("Processing document..."):
        # Save uploaded file temporarily
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())

        docs = load_document(uploaded_file)
        vectordb = get_vectorstore(docs)

    st.success("✅ Document processed! Start chatting below.")

# Chat interface
user_query = st.chat_input("Ask something...")

# Display chat history
for speaker, msg in st.session_state.history:
    if speaker == "You":
        st.chat_message("user").write(msg)
    else:
        st.chat_message("assistant").write(msg)

if user_query:
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        # Web Search Only mode
        if search_mode == "Web Search Only":
            with st.spinner("Searching the web..."):
                search_results = web_search(user_query)

                if search_results and not any("error" in r for r in search_results):
                    answer = "Here's what I found on the web:\n\n"
                    for i, result in enumerate(search_results, 1):
                        title = result.get("title", "No title")
                        content = result.get("content", "")
                        url = result.get("url", "")
                        answer += f"**{i}. {title}** \n\n{content[:100]}\n\n🔗 [Source]({url})\n\n"
                        answer += "###"*20 + '\n\n'

                    st.markdown(answer)
                    st.session_state.history.append(("You", user_query))
                    st.session_state.history.append(("Bot", answer))
                else:
                    error_msg = "Sorry, I couldn't retrieve web search results."
                    st.error(error_msg)
                    st.session_state.history.append(("You", user_query))
                    st.session_state.history.append(("Bot", error_msg))

        # Document modes
        elif vectordb and (search_mode == "Document Only" or search_mode == "Document + Web Search"):
            # Create streaming output container
            stream_container = st.empty()
            stream_handler = StreamHandler(stream_container)

            # Create model with streaming
            # llm = ChatOpenAI(
            #     model_name=MODEL_NAME,
            #     temperature=TEMPERATURE,
            #     streaming=True,
            #     callbacks=[stream_handler],
            #     openai_api_key=OPENAI_API_KEY,
            # )

            llm = init_chat_model(
                model_name=MODEL_NAME,
                model_provider=MODEL_PROVIDER,
                temperature=TEMPERATURE,
                streaming=True,
                callbacks=[stream_handler],
                openai_api_key=OPENAI_API_KEY,
            )

            # Create retriever & chain
            retriever = vectordb.as_retriever(search_kwargs={"k": NUMBER_OF_RETRIEVAL})
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                memory=st.session_state.memory,
                return_source_documents=True,
                verbose=True
            )

            # Get response from document
            response = qa_chain({"question": user_query})
            final_answer = response["answer"]
            source_docs = response.get("source_documents", [])

            # Show document sources
            if source_docs:
                with st.expander("📖 Supporting Evidence (Document)"):
                    for i, doc in enumerate(source_docs, start=1):
                        page = doc.metadata.get("page", "N/A")
                        snippet = doc.page_content[:200].strip().replace("\n", " ")
                        st.markdown(f"**Source {i} — Page {page}:** {snippet}...")

            # Add web search if enabled
            if search_mode == "Document + Web Search":
                with st.spinner("Also searching the web..."):
                    search_results = web_search(user_query)
                    if search_results and not any("error" in r for r in search_results):
                        with st.expander("🌐 Web Search Results"):
                            for i, result in enumerate(search_results, 1):
                                title = result.get("title", "No title")
                                content = result.get("content", "")
                                url = result.get("url", "")
                                st.markdown(f"**{i}. {title}**\n{content}\n🔗 [Source]({url})\n")

            # Save conversation to memory
            st.session_state.history.append(("You", user_query))
            st.session_state.history.append(("Bot", final_answer))

        else:
            st.warning("Please upload a document first for document-based modes.")

if not OPENAI_API_KEY or not TAVILY_API_KEY:
    st.warning("⚠️ Please set your API keys: `OPENAI_API_KEY` and `TAVILY_API_KEY`")
elif not uploaded_file and search_mode != "Web Search Only":
    st.info("👆 Upload a document to begin, or switch to 'Web Search Only' mode.")
