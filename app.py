import json
import streamlit as st
import boto3
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os
import uuid
from datetime import datetime
import tempfile
from typing import List, Dict
import hashlib
import re

load_dotenv()

# ✅ Initialize clients
bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")
s3_client = boto3.client('s3')

# ✅ Configuration
S3_BUCKET = os.getenv('S3_BUCKET_NAME', 'your-pdf-bucket')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME', 'pdf-embeddings')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT', 'us-east-1')  # Fixed region format

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# ──────────────────────────────────────────────
# ✅ Version Comparison Utility
def parse_version(version_str: str) -> tuple:
    """Parse version string into tuple for comparison (e.g., '2.1.3' -> (2, 1, 3))"""
    try:
        # Handle common version formats: 1.0, 2.1.3, v1.2, etc.
        clean_version = re.sub(r'^v', '', str(version_str).lower())
        parts = clean_version.split('.')
        return tuple(int(part) for part in parts)
    except:
        # If parsing fails, treat as version 0.0
        return (0, 0)

def compare_versions(v1: str, v2: str) -> int:
    """Compare two version strings. Returns 1 if v1 > v2, -1 if v1 < v2, 0 if equal"""
    tuple1 = parse_version(v1)
    tuple2 = parse_version(v2)
    
    if tuple1 > tuple2:
        return 1
    elif tuple1 < tuple2:
        return -1
    else:
        return 0

# ──────────────────────────────────────────────
# ✅ S3 Functions
def upload_to_s3(file_content: bytes, filename: str, doc_id: str) -> str:
    """Upload PDF to S3 and return the S3 key"""
    try:
        s3_key = f"pdfs/{doc_id}/{filename}"
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=s3_key,
            Body=file_content,
            ContentType='application/pdf'
        )
        return s3_key
    except Exception as e:
        st.error(f"Error uploading to S3: {str(e)}")
        return None

def download_from_s3(s3_key: str) -> bytes:
    """Download PDF from S3"""
    try:
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=s3_key)
        return response['Body'].read()
    except Exception as e:
        st.error(f"Error downloading from S3: {str(e)}")
        return None

# ──────────────────────────────────────────────
# ✅ Pinecone Functions
def initialize_pinecone_index():
    """Create Pinecone index if it doesn't exist"""
    try:
        # Validate API key
        if not PINECONE_API_KEY:
            st.error("❌ PINECONE_API_KEY not found in environment variables")
            return None
        
        # Check if index exists
        existing_indexes = [index.name for index in pc.list_indexes()]
        
        if PINECONE_INDEX_NAME not in existing_indexes:
            st.info(f"🔧 Creating new Pinecone index: {PINECONE_INDEX_NAME}")
            
            # Try different region configurations
            regions_to_try = [
                ('aws', 'us-east-1')
            ]
            
            index_created = False
            for cloud, region in regions_to_try:
                try:
                    pc.create_index(
                        name=PINECONE_INDEX_NAME,
                        dimension=1536,  # OpenAI text-embedding-3-small dimension
                        metric='cosine',
                        spec=ServerlessSpec(
                            cloud=cloud,
                            region=region
                        )
                    )
                    st.success(f"✅ Created index in {cloud}:{region}")
                    index_created = True
                    break
                except Exception as region_error:
                    st.warning(f"⚠️ Failed to create index in {cloud}:{region} - {str(region_error)}")
                    continue
            
            if not index_created:
                st.error("❌ Failed to create index in any available region")
                return None
        else:
            st.success(f"✅ Using existing Pinecone index: {PINECONE_INDEX_NAME}")
        
        return pc.Index(PINECONE_INDEX_NAME)
    except Exception as e:
        st.error(f"❌ Error initializing Pinecone: {str(e)}")
        st.info("💡 Available regions: us-east-1, us-west-2, us-central1-gcp, eastus-azure")
        return None

def embed_documents_to_pinecone(docs: List, doc_id: str, filename: str, s3_key: str, version: str = "1.0"):
    """Embed documents into Pinecone with metadata including version"""
    try:
        index = pc.Index(PINECONE_INDEX_NAME)
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        vectors_to_upsert = []
        
        for i, doc in enumerate(docs):
            # Generate embedding
            embedding = embeddings.embed_query(doc.page_content)
            
            # Create unique ID for this chunk
            chunk_id = f"{doc_id}_{i}"
            
            # Create metadata with version information
            # Convert version tuple to string for Pinecone compatibility
            version_tuple = parse_version(version)
            version_sortable = ".".join(str(x).zfill(3) for x in version_tuple)  # e.g., "002.001.003" for sorting
            
            metadata = {
                "doc_id": doc_id,
                "filename": filename,
                "s3_key": s3_key,
                "version": version,
                "version_sortable": version_sortable,  # String format for Pinecone, sortable
                "version_major": version_tuple[0] if len(version_tuple) > 0 else 0,
                "version_minor": version_tuple[1] if len(version_tuple) > 1 else 0,
                "version_patch": version_tuple[2] if len(version_tuple) > 2 else 0,
                "chunk_index": i,
                "page_number": doc.metadata.get('page', 0),
                "content": doc.page_content,
                "upload_timestamp": datetime.now().isoformat(),
                "content_hash": hashlib.md5(doc.page_content.encode()).hexdigest()
            }
            
            vectors_to_upsert.append({
                "id": chunk_id,
                "values": embedding,
                "metadata": metadata
            })
        
        # Upsert vectors in batches
        batch_size = 100
        for i in range(0, len(vectors_to_upsert), batch_size):
            batch = vectors_to_upsert[i:i + batch_size]
            index.upsert(vectors=batch)
        
        return len(vectors_to_upsert)
    except Exception as e:
        st.error(f"Error embedding to Pinecone: {str(e)}")
        return 0

def query_pinecone_with_version_priority(question: str, top_k: int = 10, final_results: int = 3) -> List[Dict]:
    """Query Pinecone and prioritize results by version, then by similarity score"""
    try:
        index = pc.Index(PINECONE_INDEX_NAME)
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # Generate query embedding
        query_embedding = embeddings.embed_query(question)
        
        # Query Pinecone with more results than needed
        results = index.query(
            vector=query_embedding,
            top_k=top_k,  # Get more results to sort by version
            include_metadata=True
        )
        
        # Group results by document/filename and find highest version for each
        doc_versions = {}
        for match in results.matches:
            filename = match.metadata.get('filename', 'unknown')
            version = match.metadata.get('version', '1.0')
            version_sortable = match.metadata.get('version_sortable', '001.000.000')
            
            if filename not in doc_versions:
                doc_versions[filename] = {
                    'version': version, 
                    'version_sortable': version_sortable,
                    'matches': []
                }
            else:
                # Compare versions using sortable string format
                if version_sortable > doc_versions[filename]['version_sortable']:
                    doc_versions[filename]['version'] = version
                    doc_versions[filename]['version_sortable'] = version_sortable
            
            doc_versions[filename]['matches'].append(match)
        
        # Filter to keep only chunks from the highest version of each document
        filtered_matches = []
        for filename, doc_info in doc_versions.items():
            highest_version = doc_info['version']
            for match in doc_info['matches']:
                if match.metadata.get('version') == highest_version:
                    # Add version priority boost to score for final sorting
                    match.version_priority_score = match.score
                    filtered_matches.append(match)
        
        # Sort by similarity score (descending) and return top results
        filtered_matches.sort(key=lambda x: x.version_priority_score, reverse=True)
        
        return filtered_matches[:final_results]
        
    except Exception as e:
        st.error(f"Error querying Pinecone: {str(e)}")
        return []

def query_pinecone(question: str, top_k: int = 3) -> List[Dict]:
    """Original query function - kept for backward compatibility"""
    return query_pinecone_with_version_priority(question, top_k * 3, top_k)

def get_document_versions(filename_pattern: str = None) -> Dict[str, List[str]]:
    """Get all versions of documents in the index"""
    try:
        index = pc.Index(PINECONE_INDEX_NAME)
        
        # Query with a dummy vector to get some results
        dummy_vector = [0.0] * 1536
        results = index.query(
            vector=dummy_vector,
            top_k=1000,  # Get many results to see all documents
            include_metadata=True
        )
        
        # Group by filename and collect versions with sortable format
        doc_versions = {}
        for match in results.matches:
            filename = match.metadata.get('filename', 'unknown')
            version = match.metadata.get('version', '1.0')
            version_sortable = match.metadata.get('version_sortable', '001.000.000')
            
            if filename not in doc_versions:
                doc_versions[filename] = {}
            doc_versions[filename][version] = version_sortable
        
        # Convert to sorted lists (highest version first)
        result = {}
        for filename, version_dict in doc_versions.items():
            # Sort by version_sortable in descending order
            sorted_versions = sorted(version_dict.items(), key=lambda x: x[1], reverse=True)
            result[filename] = [version for version, _ in sorted_versions]
        
        return result
        
    except Exception as e:
        st.error(f"Error getting document versions: {str(e)}")
        return {}

# ──────────────────────────────────────────────
# ✅ Document Processing Functions
def process_uploaded_pdf(uploaded_file, doc_id: str, version: str = "1.0"):
    """Process uploaded PDF: S3 upload + Pinecone embedding"""
    try:
        # Read file content
        file_content = uploaded_file.read()
        filename = uploaded_file.name
        
        # Upload to S3
        with st.spinner("📤 Uploading to S3..."):
            s3_key = upload_to_s3(file_content, filename, doc_id)
            if not s3_key:
                return False
        
        # Create temporary file for processing
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(file_content)
            tmp_file_path = tmp_file.name
        
        try:
            # Load and split document
            with st.spinner("📄 Processing PDF..."):
                loader = PyPDFLoader(tmp_file_path)
                documents = loader.load()
                
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500, 
                    chunk_overlap=50
                )
                docs = splitter.split_documents(documents)
            
            # Embed to Pinecone
            with st.spinner("🔗 Creating embeddings..."):
                num_embedded = embed_documents_to_pinecone(
                    docs, doc_id, filename, s3_key, version
                )
            
            if num_embedded > 0:
                st.success(f"✅ Successfully processed {filename} (v{version})")
                st.info(f"📊 Created {num_embedded} embeddings")
                st.info(f"🆔 Document ID: {doc_id}")
                st.info(f"📁 S3 Key: {s3_key}")
                return True
            else:
                st.error("Failed to create embeddings")
                return False
                
        finally:
            # Clean up temporary file
            os.unlink(tmp_file_path)
            
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return False

# ──────────────────────────────────────────────
# ✅ Stream LLaMA2 Response (unchanged)
def stream_llama2_response(context, question, container):
    formatted_prompt = f"""<s>[INST] Use the following context to answer the question briefly and clearly in under 100 words.
If you don't know the answer, say "I don't know".

Context:
{context}

Question: {question} [/INST]
"""

    body = {
        "prompt": formatted_prompt,
        "max_gen_len": 512,
        "temperature": 0.5,
        "top_p": 0.9,
    }

    try:
        response = bedrock.invoke_model_with_response_stream(
            modelId="meta.llama3-70b-instruct-v1:0",
            body=json.dumps(body)
        )

        full_response = ""
        for event in response["body"]:
            chunk = event.get("chunk")
            if chunk:
                decoded_chunk = chunk["bytes"].decode("utf-8").strip()
                for line in decoded_chunk.splitlines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        token = data.get("generation")
                        if token:
                            full_response += token
                            container.markdown(full_response)
                        if data.get("stop_reason") == "stop":
                            return
                    except json.JSONDecodeError:
                        continue

    except Exception as e:
        container.error(f"Streaming error: {str(e)}")

# ──────────────────────────────────────────────
# ✅ Main Streamlit App
def main():
    st.set_page_config(
        page_title="PDF Q&A with Version Priority",
        page_icon="📄",
        layout="wide"
    )
    
    st.title("📄 PDF Document Q&A with Version Priority")
    st.markdown("*Automatically prioritizes the highest version of each document in search results*")
    
    # Initialize Pinecone
    if 'pinecone_initialized' not in st.session_state:
        with st.spinner("🔧 Initializing Pinecone..."):
            index = initialize_pinecone_index()
            if index:
                st.session_state.pinecone_initialized = True
                st.success("✅ Pinecone initialized successfully!")
            else:
                st.error("❌ Failed to initialize Pinecone")
                return
    
    # Sidebar for document upload
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Check Pinecone configuration
        if st.button("🔍 Test Pinecone Connection"):
            try:
                if not PINECONE_API_KEY:
                    st.error("❌ PINECONE_API_KEY missing")
                else:
                    # Test connection
                    indexes = pc.list_indexes()
                    st.success(f"✅ Connected to Pinecone")
                    st.info(f"📊 Available indexes: {[idx.name for idx in indexes]}")
            except Exception as e:
                st.error(f"❌ Pinecone connection failed: {str(e)}")
        
        st.divider()
        st.header("📤 Upload Documents")
        
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            accept_multiple_files=False
        )
        
        if uploaded_file:
            # Generate unique document ID
            doc_id = str(uuid.uuid4())
            
            # Version input with validation
            version = st.text_input(
                "Version (e.g., 1.0, 2.1.3, v1.2)", 
                value="1.0",
                help="Higher versions will be prioritized in search results"
            )
            
            # Show version parsing preview
            parsed_version = parse_version(version)
            st.caption(f"Parsed as: {parsed_version}")
            
            if st.button("🚀 Process & Upload"):
                success = process_uploaded_pdf(uploaded_file, doc_id, version)
                if success:
                    st.rerun()
        
        st.divider()
        
        # Document management
        st.header("📚 Document Management")
        
        if st.button("📋 Show Document Versions"):
            with st.spinner("Loading document versions..."):
                doc_versions = get_document_versions()
                if doc_versions:
                    st.subheader("📊 Document Versions")
                    for filename, versions in doc_versions.items():
                        st.write(f"**{filename}**")
                        for i, version in enumerate(versions):
                            if i == 0:  # Highest version
                                st.write(f"  🔥 v{version} (highest - will be used)")
                            else:
                                st.write(f"  📄 v{version}")
                        st.write("")
                else:
                    st.info("No documents found in index")
        
        if st.button("🔄 Refresh Index Stats"):
            try:
                index = pc.Index(PINECONE_INDEX_NAME)
                stats = index.describe_index_stats()
                st.json(stats)
            except Exception as e:
                st.error(f"Error getting stats: {str(e)}")
    
    # Main content area
    st.header("❓ Ask Questions")
    st.markdown("*Questions will be answered using the highest version of relevant documents*")
    
    question = st.text_input("Enter your question about the documents:")
    
    if question:
        with st.spinner("🔍 Searching relevant documents (prioritizing latest versions)..."):
            matches = query_pinecone_with_version_priority(question, top_k=15, final_results=3)
            
            if matches:
                # Extract context from matches
                context_parts = []
                source_docs = []
                
                for match in matches:
                    metadata = match.metadata
                    context_parts.append(metadata['content'])
                    source_docs.append({
                        'content': metadata['content'],
                        'filename': metadata['filename'],
                        'version': metadata.get('version', '1.0'),
                        'page': metadata.get('page_number', 0),
                        'score': match.score,
                        'doc_id': metadata['doc_id']
                    })
                
                context = "\n\n".join(context_parts)
                
                # Generate response
                st.markdown("### 🤖 Answer:")
                response_container = st.empty()
                stream_llama2_response(context, question, response_container)
                
                # Show source documents with version information
                if st.checkbox("📚 Show source documents"):
                    st.markdown("#### 📄 Sources (ordered by relevance, highest versions only):")
                    for i, doc in enumerate(source_docs):
                        with st.expander(f"Source {i+1}: {doc['filename']} v{doc['version']} (Score: {doc['score']:.3f})"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**Document ID:** {doc['doc_id']}")
                                st.write(f"**Version:** {doc['version']} 🔥")
                            with col2:
                                st.write(f"**Page:** {doc['page']}")
                                st.write(f"**Similarity Score:** {doc['score']:.3f}")
                            
                            st.write(f"**Content:**")
                            st.write(doc['content'][:500] + "..." if len(doc['content']) > 500 else doc['content'])
            else:
                st.warning("No relevant documents found. Please upload some PDFs first.")

if __name__ == "__main__":
    main()