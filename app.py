import os
import warnings

# Configure warnings and environment variables before any other imports
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Configure Streamlit's logger to ignore specific warnings
logging.getLogger('streamlit.runtime.scriptrunner.magic_funcs').setLevel(logging.ERROR)
logging.getLogger('streamlit.watcher.local_sources_watcher').setLevel(logging.ERROR)

import streamlit as st
import tempfile
from typing import List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import LangChain components
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_ollama import OllamaLLM

# Import HuggingFace components last
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=UserWarning)
    from langchain_huggingface import HuggingFaceEmbeddings

# Configure Streamlit
st.set_page_config(
    page_title="PDF Q&A with RAG",
    page_icon="📚",
    layout="wide"
)

# Add custom CSS for styling
st.markdown("""
<style>
    .section-header {
        font-weight: 600;
        margin-bottom: 8px;
        color: #333;
    }
    .thinking-text {
        color: #666666;
        font-style: italic;
        background-color: #f5f5f5;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
        border-left: 4px solid #9e9e9e;
    }
    .answer-text {
        color: #1e88e5;
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 5px;
        border-left: 4px solid #1e88e5;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Disable PyTorch/Streamlit file watcher warnings
if not st.runtime.exists():
    import torch._classes
    def __getattr__(*args, **kwargs):
        return None
    torch._classes.__getattr__ = __getattr__

# Initialize embedding models
@st.cache_resource
def get_embeddings(provider: str = "sentence_transformer"):
    """Get the embedding model based on the selected provider.
    
    Args:
        provider (str): The embedding provider name ('sentence_transformer' or 'openai')
    
    Returns:
        Embeddings: The initialized embedding model
    """
    if provider.lower() == "sentence_transformer":
        model_name = os.getenv("SENTENCE_TRANSFORMER_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    elif provider.lower() == "openai":
        if "OPENAI_API_KEY" not in os.environ:
            raise ValueError("Please set your OPENAI_API_KEY environment variable")
        return OpenAIEmbeddings(
            model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        )
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")

# Cache the LLM initialization
@st.cache_resource
def get_llm(provider: str, model: Optional[str] = None, temperature: Optional[float] = None, top_p: Optional[float] = None):
    """Get the LLM based on the selected provider."""
    if provider.lower() == "openai":
        if "OPENAI_API_KEY" not in os.environ:
            raise ValueError("Please set your OPENAI_API_KEY environment variable")
        return ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0,
            streaming=True
        )
    elif provider.lower() == "ollama":
        # Changed from mistral to llama2 as it's more commonly available
        return OllamaLLM(
            model="deepseek-r1:latest",
            temperature=0.0,
            base_url="http://localhost:11434"
        )
    elif provider.lower() == "deepseek":
        model = model or st.session_state.get("DEEPSEEK_MODEL", "deepseek-r1:latest")
        temperature = temperature if temperature is not None else st.session_state.get("DEEPSEEK_TEMP", 0.3)
        top_p = top_p if top_p is not None else st.session_state.get("DEEPSEEK_TOP_P", 0.9)
        
        return OllamaLLM(
            model=model,
            temperature=temperature,
            base_url="http://localhost:11434",
            stop=["<|end|>"],
            streaming=True,
            model_kwargs={
                "top_p": top_p,
                "top_k": 10,
                "repeat_penalty": 1.1,
            }
        )
    elif provider.lower() == "groq":
        if "GROQ_API_KEY" not in os.environ:
            raise ValueError("Please set your GROQ_API_KEY environment variable")
        
        model = model or st.session_state.get("GROQ_MODEL", "mixtral-8x7b-32768")
        temperature = temperature if temperature is not None else st.session_state.get("GROQ_TEMP", 0.0)
        
        return ChatGroq(
            temperature=temperature,
            groq_api_key=os.environ["GROQ_API_KEY"],
            model_name=model,
            streaming=True  # Enable streaming for better UX
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

def get_groq_models():
    """Get available Groq models from environment variable."""
    models_str = os.getenv("GROQ_MODELS", "mixtral-8x7b-32768,llama2-70b-4096,deepseek-r1-distill-llama-70b,gemma-7b-it")
    return [model.strip() for model in models_str.split(",")]

@st.cache_data
def load_pdf(uploaded_file) -> List[Document]:
    logger.info("Starting PDF document loading process")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name
        logger.info(f"Created temporary file: {tmp_path}")

        try:
            loader = PyPDFLoader(tmp_path)
            documents = loader.load()
            logger.info(f"Successfully loaded PDF with {len(documents)} pages")
            return documents
        finally:
            os.remove(tmp_path)
            logger.info("Cleaned up temporary PDF file")

@st.cache_resource
def create_vector_store(_documents: List[Document], embedding_provider: str = "sentence_transformer") -> FAISS:
    logger.info(f"Creating vector store using {embedding_provider} embeddings")
    
    # Use RecursiveCharacterTextSplitter for better chunk splitting
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    docs = text_splitter.split_documents(_documents)
    logger.info(f"Split documents into {len(docs)} chunks")
    
    # Initialize embeddings: OpenAI or SentenceTransformer
    logger.info(f"Initializing {embedding_provider} embedding model")
    embeddings = get_embeddings(embedding_provider)
    
    # Initialize vector store using FAISS from documents and embeddings
    logger.info("Creating FAISS index and embedding documents")
    
    # The FAISS.from_documents() function:
    # - Takes each doc.page_content (plain text)
    # - Passes it to the embeddings model
    # - The embeddings model handles tokenization and embedding internally
    # - Embedding vectors are stored in FAISS for similarity search
    vector_store = FAISS.from_documents(docs, embeddings)
    logger.info("Vector store creation completed")
    return vector_store

def get_answer_from_llm(llm, context: str, question: str) -> str:
    logger.info("Generating answer from LLM")
    
    if isinstance(llm, ChatOpenAI):
        # Simpler prompt for OpenAI
        prompt = f"""Use the following context to answer the question. Be clear and concise.

Context:
{context}

Question: {question}

Answer: """
    else:
        # Keep the thinking template for other models
        prompt = f""""Use the following context to answer the question. Be clear and concise.

Context:
{context}

Question: {question}

Answer: """
    
    try:
        # Added logging for Groq model to print model name and temperature
        if isinstance(llm, ChatGroq):
            logger.info(f"Sending request to Groq model with parameters: model_name='{llm.model_name}', temperature={llm.temperature}")
        response = llm.invoke(prompt)
        
        # Handle different response formats based on the LLM type
        if isinstance(llm, ChatOpenAI):
            if hasattr(response, 'content'):
                return response.content
            return str(response)
        elif isinstance(llm, ChatGroq):
            if hasattr(response, 'content'):
                return response.content
            elif isinstance(response, str):
                return response.split('content="')[1].split('"')[0] if 'content="' in response else response
        
        return response
    except Exception as e:
        logger.error(f"Error in LLM response: {str(e)}")
        return f"Error generating response: {str(e)}"

# --- Streamlit UI ---
st.title("📚 PDF Q&A with RAG")
st.write("Upload a PDF and ask questions about its content using state-of-the-art language models!")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # LLM Provider Selection
    provider_selected = st.selectbox(
        "Select LLM Provider",
        ["openai", "ollama", "deepseek", "groq"],
        index=3, # Changed default to Groq
        help="Choose the AI model provider. DeepSeek-R1 is an efficient open-source alternative."
    )
    st.session_state["LLM_PROVIDER"] = provider_selected
    
    # Model Configuration (for DeepSeek)
    if provider_selected == "deepseek":
        model_size = st.selectbox(
            "Select DeepSeek Model",
            ["deepseek-r1:latest", "deepseek-r1:instruct"],
            index=0,
            help="Choose the DeepSeek model variant. The instruct model is optimized for following instructions."
        )
        
        # Advanced settings collapsible
        with st.expander("Advanced Settings"):
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.1,
                help="Higher values make the output more creative but less focused"
            )
            top_p = st.slider(
                "Top P",
                min_value=0.0,
                max_value=1.0,
                value=0.9,
                step=0.1,
                help="Controls diversity of token selection"
            )
        
        st.session_state["DEEPSEEK_MODEL"] = model_size
        st.session_state["DEEPSEEK_TEMP"] = temperature
        st.session_state["DEEPSEEK_TOP_P"] = top_p
    
    # Model Configuration (for Groq)
    elif provider_selected == "groq":
        groq_model = st.selectbox(
            "Select Groq Model",
            get_groq_models(),
            index=2, # Changed default index to select deepseek-r1-distill-llama-70b
            help="Choose the Groq model variant to use for generation."
        )
        
        # Advanced settings collapsible
        with st.expander("Advanced Settings"):
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.1,
                help="Higher values make the output more creative but less focused"
            )
        
        st.session_state["GROQ_MODEL"] = groq_model
        st.session_state["GROQ_TEMP"] = temperature
    
    # Embedding Provider Selection
    embedding_provider = st.selectbox(
        "Select Embedding Provider",
        ["sentence_transformer", "openai"],
        index=0,
        help="Choose the embedding model provider"
    )
    st.session_state["EMBEDDING_PROVIDER"] = embedding_provider

# Main content
uploaded_pdf = st.file_uploader(
    "Upload PDF file",
    type=["pdf"],
    help="Upload a PDF document to ask questions about"
)

if uploaded_pdf:
    with st.status("Processing document...", expanded=True) as status:
        st.write("Loading PDF...")
        documents = load_pdf(uploaded_pdf)
        
        st.write("Creating vector store...")
        vector_store = create_vector_store(
            documents,
            embedding_provider=st.session_state["EMBEDDING_PROVIDER"]
        )
        
        status.update(label="Ready!", state="complete", expanded=False)
    
    # Query interface
    query = st.text_input(
        "Ask a question about the document:",
        placeholder="What is the main topic of this document?"
    )
    
    if query:
        with st.spinner("Generating answer..."):
            # Create placeholder for streaming output
            answer_placeholder = st.empty()
            thinking_container = st.empty()
            answer_container = st.empty()
            
            try:
                # Initialize response variable
                response = None
                should_continue = True
                
                # 1. Get relevant documents from vector store
                logger.info("Searching for relevant documents")
                relevant_docs = vector_store.similarity_search(query, k=4)
                logger.info(f"Found {len(relevant_docs)} relevant documents")
                
                # 2. Combine relevant documents into context
                logger.info("Preparing context from relevant documents")
                context = "\n\n".join(doc.page_content for doc in relevant_docs)
                
                # 3. Get LLM instance
                try:
                    logger.info(f"Initializing {st.session_state['LLM_PROVIDER']} model")
                    provider = st.session_state["LLM_PROVIDER"]
                    if provider == "groq":
                        llm = get_llm(provider, model=st.session_state["GROQ_MODEL"], temperature=st.session_state["GROQ_TEMP"])
                    elif provider == "deepseek":
                        llm = get_llm(
                            provider,
                            model=st.session_state["DEEPSEEK_MODEL"],
                            temperature=st.session_state["DEEPSEEK_TEMP"],
                            top_p=st.session_state["DEEPSEEK_TOP_P"]
                        )
                    else:
                        llm = get_llm(provider)
                    logger.info("LLM initialized successfully")
                except Exception as e:
                    logger.error(f"Error initializing LLM: {str(e)}")
                    if "not found" in str(e):
                        st.info("Please make sure you have pulled the required model using Ollama. Run:\n```bash\nollama pull llama2\n```")
                    should_continue = False
                
                # 4. Get answer from LLM if we should continue
                if should_continue:
                    try:
                        logger.info("Generating response from LLM")
                        response = get_answer_from_llm(llm, context, query)
                        logger.info("Response generated successfully")
                    except Exception as e:
                        logger.error(f"Error getting response from LLM: {str(e)}")
                        should_continue = False
                
                if should_continue and not response:
                    st.error("No response received from the model.")
                    should_continue = False
                
                # Show relevant context (optional for debugging)
                with st.expander("View Retrieved Context"):
                    st.markdown("**Retrieved Passages:**")
                    for i, doc in enumerate(relevant_docs, 1):
                        st.markdown(f"**Passage {i}:**\n{doc.page_content}\n")
                
                # 5. Process and display response if we have one
                if should_continue and response:
                    # Clean the response by removing any metadata or system information
                    cleaned_response = response
                    if isinstance(response, str):
                        # Remove any metadata that might be present
                        if 'additional_kwargs=' in response:
                            cleaned_response = response.split('content="')[1].split('" additional_kwargs=')[0]
                        elif 'response_metadata=' in response:
                            cleaned_response = response.split('content="')[1].split('" response_metadata=')[0]
                    
                    # Check if we're using OpenAI
                    if isinstance(llm, ChatOpenAI):
                        # Direct display without thinking section for OpenAI
                        answer_container.markdown(f"""
                            <div class="section-header">🔍 Answer:</div>
                            <div class="answer-text">{cleaned_response}</div>
                        """, unsafe_allow_html=True)
                    else:
                        # Process thinking and answer sections for other models
                        thinking_part = ""
                        answer_part = cleaned_response
                        
                        # Try to extract thinking section if present
                        if "<think>" in cleaned_response and "</think>" in cleaned_response:
                            try:
                                parts = cleaned_response.split("<think>")
                                thinking_section = parts[1].split("</think>")[0].strip()
                                answer_section = parts[1].split("</think>")[1].strip()
                                
                                # Further clean up the answer section
                                if "Answer:" in answer_section:
                                    answer_part = answer_section.split("Answer:")[1].strip()
                                else:
                                    answer_part = answer_section.strip()
                                    
                                thinking_part = thinking_section
                            except IndexError:
                                # If splitting fails, keep the entire response as the answer
                                answer_part = cleaned_response
                        else:
                            # If no think tags, check for "Answer:" marker
                            if "Answer:" in cleaned_response:
                                answer_part = cleaned_response.split("Answer:")[1].strip()
                        
                        # Display thinking section if available
                        if thinking_part:
                            with thinking_container.expander("💭 View Thinking Process", expanded=False):
                                st.markdown(f'<div class="thinking-text">{thinking_part}</div>', 
                                           unsafe_allow_html=True)
                        
                        # Display answer section
                        answer_container.markdown(f"""
                            <div class="section-header">🔍 Answer:</div>
                            <div class="answer-text">{answer_part}</div>
                        """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                if response:  # Only show raw response if it exists
                    st.write("Raw response:", response) 