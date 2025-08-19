from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np
import openai
import os
import streamlit as st
import hashlib
import tempfile
from typing import List

# Import the chart generator
from chart_generator import ChartGenerator

load_dotenv()

# Configuration
api_key = os.getenv("OPENAI_API_KEY")
openai_client = openai.OpenAI(api_key=api_key)

# PostgreSQL connection parameters
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
    "database": os.getenv("DB_NAME", "pdf_chatbot"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "password")
}

# File upload limits - 1GB max file size
MAX_FILE_SIZE = 1024 * 1024 * 1024  # 1GB in bytes

# Streamlit UI setup with increased upload limit
st.set_page_config(page_title="PDF Chatbot with Charts - Up to 1GB", layout="wide")

# Initialize session state
if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = None
if "selected_document_ids" not in st.session_state:
    st.session_state.selected_document_ids = []
if "chat_mode" not in st.session_state:
    st.session_state.chat_mode = "single"  # "single" or "multi"
if "chart_generator" not in st.session_state:
    st.session_state.chart_generator = ChartGenerator(openai_client)

def get_db_connection():
    """Create and return a database connection."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except psycopg2.Error as e:
        st.error(f"Database connection error: {e}")
        return None

def init_database():
    """Initialize database tables and pgvector extension."""
    conn = get_db_connection()
    if not conn:
        return False
    
    try:
        with conn.cursor() as cur:
            # Enable pgvector extension
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # Create documents table with file size tracking
            cur.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    filename VARCHAR(255) NOT NULL,
                    file_hash VARCHAR(64) UNIQUE NOT NULL,
                    file_size BIGINT NOT NULL,
                    upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Create chunks table with vector column
            cur.execute("""
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id SERIAL PRIMARY KEY,
                    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
                    chunk_text TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    embedding vector(384),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Create index for vector similarity search
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding 
                ON document_chunks USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
            """)
            
            conn.commit()
            return True
    except psycopg2.Error as e:
        st.error(f"Database initialization error: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()

def calculate_file_hash(uploaded_file):
    """Calculate SHA-256 hash of uploaded file."""
    uploaded_file.seek(0)
    file_content = uploaded_file.read()
    uploaded_file.seek(0)
    return hashlib.sha256(file_content).hexdigest()

def check_file_size(uploaded_file):
    """Check if file size is within limits."""
    file_size = uploaded_file.size
    if file_size > MAX_FILE_SIZE:
        return False, f"File size ({file_size / (1024*1024*1024):.2f} GB) exceeds maximum limit of 1GB"
    return True, f"File size: {file_size / (1024*1024):.2f} MB"

def document_exists(file_hash):
    """Check if document with given hash already exists in database."""
    conn = get_db_connection()
    if not conn:
        return None
    
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, filename FROM documents WHERE file_hash = %s;", (file_hash,))
            result = cur.fetchone()
            return result[0] if result else None
    except psycopg2.Error as e:
        st.error(f"Error checking document existence: {e}")
        return None
    finally:
        conn.close()

def load_and_chunk_pdf(uploaded_file, chunk_size=800, chunk_overlap=200):
    """
    Loads a PDF file from Streamlit's file uploader, extracts text, and splits it into manageable chunks.
    
    Parameters:
    - uploaded_file (BytesIO): The uploaded PDF file.
    - chunk_size (int): The maximum number of characters per chunk.
    - chunk_overlap (int): Overlap between chunks to preserve context.
    
    Returns:
    - List of text chunks extracted from the PDF.
    """
    try:
        # Create a temporary file for large PDF processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            temp_file_path = temp_file.name
        
        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()
        
        # Use larger chunks for better context preservation
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            keep_separator=True
        )
        chunks = text_splitter.split_documents(documents)
        texts = [chunk.page_content for chunk in chunks]
        
        # Cleanup temporary file
        os.unlink(temp_file_path)
        
        print(f"PDF processed: {len(texts)} chunks created")
        return texts
    except Exception as e:
        st.error(f"Error loading PDF: {e}")
        return []

def generate_embeddings(texts, model_name="all-MiniLM-L6-v2"):
    """Generate embeddings for text chunks with progress tracking."""
    embedding_model = SentenceTransformer(model_name)
    
    # Process in batches for large files
    batch_size = 32
    embeddings = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_embeddings = embedding_model.encode(batch, batch_size=batch_size, convert_to_tensor=False)
        embeddings.extend(batch_embeddings)
        
        # Update progress
        progress = min((i + batch_size) / len(texts), 1.0)
        progress_bar.progress(progress)
        status_text.text(f"Processing embeddings: {i + len(batch)}/{len(texts)} chunks")
    
    progress_bar.empty()
    status_text.empty()
    
    print(f"Generated {len(embeddings)} embeddings")
    return embedding_model, embeddings

def store_document_and_chunks(filename, file_hash, file_size, texts, embeddings):
    """Store document and its chunks with embeddings in PostgreSQL."""
    conn = get_db_connection()
    if not conn:
        return None
    
    try:
        with conn.cursor() as cur:
            # Insert document with file size
            cur.execute("""
                INSERT INTO documents (filename, file_hash, file_size) 
                VALUES (%s, %s, %s) RETURNING id;
            """, (filename, file_hash, file_size))
            document_id = cur.fetchone()[0]
            
            # Insert chunks with embeddings in batches
            batch_size = 100
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_embeddings = embeddings[i:i + batch_size]
                
                for j, (text, embedding) in enumerate(zip(batch_texts, batch_embeddings)):
                    embedding_list = embedding.tolist()
                    cur.execute("""
                        INSERT INTO document_chunks (document_id, chunk_text, chunk_index, embedding)
                        VALUES (%s, %s, %s, %s);
                    """, (document_id, text, i + j, embedding_list))
                
                conn.commit()  # Commit in batches
            
            print(f"Stored document with ID {document_id} and {len(texts)} chunks")
            return document_id
    except psycopg2.Error as e:
        st.error(f"Error storing document: {e}")
        conn.rollback()
        return None
    finally:
        conn.close()

def search_similar_chunks_multi_docs(query, document_ids: List[int], embedding_model, top_k=7):
    """Search for similar chunks across multiple documents using pgvector cosine similarity."""
    conn = get_db_connection()
    if not conn:
        return []
    
    try:
        # Generate query embedding
        query_embedding = embedding_model.encode([query], convert_to_tensor=False)[0]
        query_embedding_list = query_embedding.tolist()
        
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Search across multiple documents
            placeholders = ','.join(['%s'] * len(document_ids))
            
            cur.execute(f"""
                SELECT dc.chunk_text, 
                       dc.chunk_index,
                       d.filename,
                       dc.embedding <=> %s::vector as similarity_score
                FROM document_chunks dc
                JOIN documents d ON dc.document_id = d.id
                WHERE dc.document_id IN ({placeholders})
                AND dc.embedding <=> %s::vector < 0.8
                ORDER BY dc.embedding <=> %s::vector
                LIMIT %s;
            """, [query_embedding_list] + document_ids + [query_embedding_list, query_embedding_list, top_k])
            
            results = cur.fetchall()
            
            if results:
                print(f"Found {len(results)} relevant chunks across {len(document_ids)} documents:")
                chunks_with_source = []
                for i, result in enumerate(results):
                    print(f"Chunk {i+1} from {result['filename']}: Score {result['similarity_score']:.3f}")
                    # Add source information to chunk
                    chunk_with_source = f"[Source: {result['filename']}]\n{result['chunk_text']}"
                    chunks_with_source.append(chunk_with_source)
                
                return chunks_with_source
            else:
                print("No chunks found within similarity threshold, trying broader search...")
                # Fallback: get top chunks without threshold
                cur.execute(f"""
                    SELECT dc.chunk_text, 
                           dc.chunk_index,
                           d.filename,
                           dc.embedding <=> %s::vector as similarity_score
                    FROM document_chunks dc
                    JOIN documents d ON dc.document_id = d.id
                    WHERE dc.document_id IN ({placeholders})
                    ORDER BY dc.embedding <=> %s::vector
                    LIMIT %s;
                """, [query_embedding_list] + document_ids + [query_embedding_list, top_k])
                
                fallback_results = cur.fetchall()
                if fallback_results:
                    chunks_with_source = []
                    for result in fallback_results:
                        chunk_with_source = f"[Source: {result['filename']}]\n{result['chunk_text']}"
                        chunks_with_source.append(chunk_with_source)
                    return chunks_with_source
                else:
                    return ["No relevant results found."]
                    
    except psycopg2.Error as e:
        st.error(f"Error searching chunks: {e}")
        return ["Error occurred during search."]
    finally:
        conn.close()

def search_similar_chunks_single_doc(query, document_id, embedding_model, top_k=5):
    """Search for similar chunks in a single document using pgvector cosine similarity."""
    return search_similar_chunks_multi_docs(query, [document_id], embedding_model, top_k)

def generate_response(query, retrieved_chunks, is_multi_doc=False):
    """Generate response using OpenAI API with retrieved chunks and smart formatting."""
    if not retrieved_chunks or retrieved_chunks == ["No relevant results found."] or retrieved_chunks == ["Error occurred during search."]:
        return "No relevant information found in the document(s) to answer your question."

    try:
        # Combine chunks into context
        context = "\n\n".join(retrieved_chunks)
        
        # Enhanced system prompt for better responses
        multi_doc_instruction = ""
        if is_multi_doc:
            multi_doc_instruction = """
IMPORTANT: You are searching across MULTIPLE documents. When answering:
- Mention which document(s) contain the relevant information
- If information comes from different documents, clearly indicate the sources
- Synthesize information from multiple sources when applicable
- If there are contradictions between documents, mention them
"""
        
        system_prompt = f"""You are an intelligent document assistant. Your job is to:

1. ALWAYS provide helpful answers based on the document content provided
2. If the user asks for information in a specific format (like tables, lists, comparisons), provide it in that format
3. If the user asks for a table, create a well-structured markdown table
4. If the user asks for comparisons, provide clear comparisons
5. Use the document content creatively to answer questions even if the exact format isn't in the document
6. If you can extract relevant information, always provide an answer - don't say "no information found" unless truly nothing is relevant
7. Be comprehensive but concise
8. Format your response clearly with proper structure
{multi_doc_instruction}
Remember: Your goal is to be helpful and extract maximum value from the provided content."""

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user", 
                    "content": f"""Based on the following document excerpts, please answer this question: "{query}"

Document Content:
{context}

Please provide a comprehensive and well-formatted answer. If the user requested a specific format (like a table), please provide it in that format using the available information."""
                }
            ],
            temperature=0.1,  # Lower temperature for more consistent responses
            max_tokens=1500   # Allow longer responses for detailed answers
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {e}"

def get_uploaded_documents():
    """Get list of uploaded documents from database with file sizes."""
    conn = get_db_connection()
    if not conn:
        return []
    
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT id, filename, file_size, upload_timestamp 
                FROM documents 
                ORDER BY upload_timestamp DESC;
            """)
            return cur.fetchall()
    except psycopg2.Error as e:
        st.error(f"Error fetching documents: {e}")
        return []
    finally:
        conn.close()

def delete_document(document_id):
    """Delete document and its chunks from database."""
    conn = get_db_connection()
    if not conn:
        return False
    
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM documents WHERE id = %s;", (document_id,))
            conn.commit()
            return True
    except psycopg2.Error as e:
        st.error(f"Error deleting document: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()

def format_file_size(size_bytes):
    """Format file size in human readable format."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024**2:
        return f"{size_bytes/1024:.1f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes/(1024**2):.1f} MB"
    else:
        return f"{size_bytes/(1024**3):.2f} GB"

# Initialize database on startup
if init_database():
    st.success("Database initialized successfully!", icon="✅")
else:
    st.error("Failed to initialize database. Please check your PostgreSQL connection.", icon="❌")

# Initialize embedding model
if st.session_state.embedding_model is None:
    with st.spinner("Loading embedding model..."):
        st.session_state.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Sidebar for file upload and document management
with st.sidebar:
    st.header("📄 Document Management")
    
    # File uploader with custom help text to show 1GB limit
    uploaded_file = st.file_uploader(
        "Upload a PDF", 
        type=["pdf"], 
        key="file_uploader",
        help="Maximum file size: 1GB"
    )
    
    if uploaded_file is not None:
        # Check file size
        size_ok, size_msg = check_file_size(uploaded_file)
        
        if not size_ok:
            st.error(size_msg)
        else:
            file_hash = calculate_file_hash(uploaded_file)
            existing_doc_id = document_exists(file_hash)
            
            if existing_doc_id:
                st.info(f"Document already exists in database!")
                if existing_doc_id not in st.session_state.selected_document_ids:
                    st.session_state.selected_document_ids = [existing_doc_id]
            else:
                with st.spinner("Processing large PDF... This may take a while..."):
                    texts = load_and_chunk_pdf(uploaded_file)
                    if texts:
                        embedding_model, embeddings = generate_embeddings(texts)
                        document_id = store_document_and_chunks(
                            uploaded_file.name, file_hash, uploaded_file.size, texts, embeddings
                        )
                        if document_id:
                            st.session_state.selected_document_ids = [document_id]
                            st.success(f"✅ {uploaded_file.name} processed successfully!")
                        else:
                            st.error("Failed to store document in database.")
    
    st.divider()
    
    # Chat mode selection
    st.subheader("🎯 Query Mode")
    chat_mode = st.radio(
        "Choose how to search:",
        ["Single Document", "Multiple Documents"],
        index=0 if st.session_state.chat_mode == "single" else 1
    )
    st.session_state.chat_mode = "single" if chat_mode == "Single Document" else "multi"
    
    # Document selector
    st.subheader("📚 Available Documents")
    documents = get_uploaded_documents()
    
    if documents:
        if st.session_state.chat_mode == "single":
            # Single document selection
            doc_options = {f"{doc['filename']} ({format_file_size(doc['file_size'])})": doc['id'] 
                          for doc in documents}
            
            selected_doc = st.selectbox("Select a document to chat with:", 
                                      options=list(doc_options.keys()),
                                      index=0)
            
            if selected_doc:
                selected_doc_id = doc_options[selected_doc]
                st.session_state.selected_document_ids = [selected_doc_id]
        else:
            # Multiple document selection
            st.write("Select multiple documents to search across:")
            selected_docs = []
            
            for doc in documents:
                doc_name = f"{doc['filename']} ({format_file_size(doc['file_size'])})"
                if st.checkbox(doc_name, value=doc['id'] in st.session_state.selected_document_ids, key=f"doc_{doc['id']}"):
                    selected_docs.append(doc['id'])
            
            st.session_state.selected_document_ids = selected_docs
            
            if selected_docs:
                st.success(f"Selected {len(selected_docs)} document(s) for multi-document search")
        
        # Delete document functionality
        st.divider()
        st.subheader("🗑️ Delete Documents")
        doc_to_delete = st.selectbox(
            "Select document to delete:",
            options=[""] + [f"{doc['filename']} ({format_file_size(doc['file_size'])})" for doc in documents],
            key="delete_selector"
        )
        
        if doc_to_delete and st.button("Delete Selected Document", type="secondary"):
            doc_id_to_delete = next(doc['id'] for doc in documents if f"{doc['filename']} ({format_file_size(doc['file_size'])})" == doc_to_delete)
            if delete_document(doc_id_to_delete):
                st.success("Document deleted successfully!")
                st.rerun()
    else:
        st.info("No documents uploaded yet.")
    
    # Chart generation info
    st.divider()
    st.subheader("📊 Chart Generation")
    st.info("Ask questions like:\n• 'Create a bar chart of sales data'\n• 'Show me a pie chart of market share'\n• 'Visualize the quarterly trends'\n• 'Make a graph of the results'")

# Main chat interface
st.title("💬 Chat with your PDF(s)")
if st.session_state.chat_mode == "multi":
    st.caption("🚀 Multi-document RAG with PostgreSQL, pgvector & Chart Generation")
else:
    st.caption("🚀 Single-document RAG with PostgreSQL, pgvector & Chart Generation")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Upload PDF documents (up to 1GB each) and start asking questions! You can search across single or multiple documents, and I can also create charts and visualizations from your data. Try asking me to 'create a chart' or 'visualize' your data!"}
    ]

# Display chat messages
for msg in st.session_state.messages:
    if msg["role"] == "assistant":
        st.chat_message("assistant").write(msg["content"])
        # Display chart if it exists in the message
        if "chart" in msg and msg["chart"] is not None:
            st.plotly_chart(msg["chart"], use_container_width=True)
    else:
        st.chat_message(msg["role"]).write(msg["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your document(s) or request a chart..."):
    if not st.session_state.selected_document_ids:
        st.warning("Please upload and select document(s) first.")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        # Search for relevant chunks
        is_multi_doc = len(st.session_state.selected_document_ids) > 1
        search_msg = f"Searching across {len(st.session_state.selected_document_ids)} document(s)..." if is_multi_doc else "Searching document..."
        
        with st.spinner(search_msg):
            if is_multi_doc:
                retrieved_chunks = search_similar_chunks_multi_docs(
                    prompt, 
                    st.session_state.selected_document_ids, 
                    st.session_state.embedding_model,
                    top_k=10  # More chunks for multi-doc search
                )
            else:
                retrieved_chunks = search_similar_chunks_single_doc(
                    prompt, 
                    st.session_state.selected_document_ids[0], 
                    st.session_state.embedding_model
                )
        
        print("=== SEARCH RESULTS ===")
        print(f"Query: {prompt}")
        print(f"Mode: {'Multi-doc' if is_multi_doc else 'Single-doc'}")
        print(f"Documents: {st.session_state.selected_document_ids}")
        print(f"Retrieved chunks: {len(retrieved_chunks)}")
        print("=====================")
        
        # Check if user is requesting a chart
        if st.session_state.chart_generator.detect_chart_request(prompt):
            # Generate chart and response
            with st.spinner("Creating chart and generating response..."):
                chart, response = st.session_state.chart_generator.generate_chart_response(prompt, retrieved_chunks)
                
                # Display chart immediately
                if chart is not None:
                    st.plotly_chart(chart, use_container_width=True)
                
                # Store message with chart
                assistant_message = {
                    "role": "assistant", 
                    "content": response,
                    "chart": chart
                }
                st.session_state.messages.append(assistant_message)
                st.chat_message("assistant").write(response)
        else:
            # Generate normal text response
            with st.spinner("Generating response..."):
                response = generate_response(prompt, retrieved_chunks, is_multi_doc)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.chat_message("assistant").write(response)

# Display current document info
if st.session_state.selected_document_ids:
    with st.expander("📋 Current Selection Info"):
        conn = get_db_connection()
        if conn:
            try:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    placeholders = ','.join(['%s'] * len(st.session_state.selected_document_ids))
                    cur.execute(f"""
                        SELECT d.filename, d.file_size, d.upload_timestamp, COUNT(c.id) as chunk_count
                        FROM documents d
                        LEFT JOIN document_chunks c ON d.id = c.document_id
                        WHERE d.id IN ({placeholders})
                        GROUP BY d.id, d.filename, d.file_size, d.upload_timestamp
                        ORDER BY d.upload_timestamp DESC;
                    """, st.session_state.selected_document_ids)
                    doc_infos = cur.fetchall()
                    
                    if doc_infos:
                        total_chunks = sum(doc['chunk_count'] for doc in doc_infos)
                        total_size = sum(doc['file_size'] for doc in doc_infos)
                        
                        st.write(f"**Mode:** {'Multi-document' if len(doc_infos) > 1 else 'Single-document'}")
                        st.write(f"**Total Documents:** {len(doc_infos)}")
                        st.write(f"**Total Size:** {format_file_size(total_size)}")
                        st.write(f"**Total Chunks:** {total_chunks}")
                        
                        st.write("**Documents:**")
                        for doc in doc_infos:
                            st.write(f"- {doc['filename']} ({format_file_size(doc['file_size'])}) - {doc['chunk_count']} chunks")
            except psycopg2.Error as e:
                st.error(f"Error fetching document info: {e}")
            finally:
                conn.close()

# Add footer with chart examples
st.markdown("---")
with st.expander("💡 Chart Generation Examples"):
    st.markdown("""
    **Try these chart requests:**
    
    📊 **Bar Charts:** "Create a bar chart showing sales by quarter"
    
    📈 **Line Charts:** "Show me a line graph of revenue trends over time"
    
    🥧 **Pie Charts:** "Make a pie chart of market share distribution"
    
    🔍 **Scatter Plots:** "Create a scatter plot of price vs performance"
    
    📋 **Data Tables:** "Show me the financial data in a table format"
    
    **Tips:**
    - Include keywords like "chart", "graph", "visualize", "plot"
    - Mention the type of chart you want (bar, line, pie, scatter)
    - Be specific about what data you want to see
    - The system will extract numerical data from your documents automatically
    """)