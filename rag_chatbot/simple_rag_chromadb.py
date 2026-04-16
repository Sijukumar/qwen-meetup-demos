#!/usr/bin/env python3
"""
RAG Chatbot with ChromaDB - Intelligent Enterprise Knowledge Assistant
Uses local ChromaDB vector database for document retrieval
"""

import os
import sys
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions

# Load environment variables from .env file
load_dotenv()


def get_api_key():
    """Get API key from environment variable."""
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("Error: DASHSCOPE_API_KEY not found in environment variables.")
        sys.exit(1)
    return api_key


def get_base_url():
    """Get base URL from environment variable."""
    return os.getenv("DASHSCOPE_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")


def get_model():
    """Get the LLM model name from environment variable."""
    return os.getenv("LLM_MODEL", "qwen-plus")


def get_embedding_model():
    """Get the embedding model name from environment variable."""
    return os.getenv("EMBEDDING_MODEL", "text-embedding-v3")


def create_client(api_key, base_url):
    """Create and return an OpenAI client configured for DashScope."""
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
    )


class ChromaDBManager:
    """Manages ChromaDB vector database operations."""
    
    def __init__(self, persist_directory="./chroma_db", collection_name="knowledge_base"):
        """Initialize ChromaDB with persistent storage."""
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Initialize ChromaDB client with persistent storage
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Use DashScope embeddings via OpenAI-compatible API
        api_key = get_api_key()
        base_url = get_base_url()
        
        # Create embedding function using DashScope
        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name=get_embedding_model(),
            api_base=base_url
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_documents(self, documents, metadatas=None, ids=None):
        """
        Add documents to the vector database in batches.
        
        Args:
            documents: List of text documents to add
            metadatas: Optional list of metadata dicts for each document
            ids: Optional list of unique IDs for each document
        """
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(documents))]
        
        # Batch size limit for embedding API
        BATCH_SIZE = 10
        total_added = 0
        
        for i in range(0, len(documents), BATCH_SIZE):
            batch_docs = documents[i:i + BATCH_SIZE]
            batch_ids = ids[i:i + BATCH_SIZE]
            batch_metadatas = metadatas[i:i + BATCH_SIZE] if metadatas else None
            
            self.collection.add(
                documents=batch_docs,
                metadatas=batch_metadatas,
                ids=batch_ids
            )
            total_added += len(batch_docs)
            print(f"  Progress: {total_added}/{len(documents)} chunks added...", end='\r')
        
        print(f"\n✓ Added {total_added} documents to ChromaDB")
    
    def add_from_file(self, file_path, chunk_size=1000, chunk_overlap=100):
        """
        Load and add documents from a text or PDF file.
        
        Args:
            file_path: Path to the text or PDF file
            chunk_size: Size of each chunk in characters (default 1000)
            chunk_overlap: Overlap between chunks (default 100)
        """
        text = ""
        
        print(f"\nProcessing file: {os.path.basename(file_path)}")
        
        # Check file extension
        if file_path.lower().endswith('.pdf'):
            try:
                import pypdf
                with open(file_path, 'rb') as f:
                    reader = pypdf.PdfReader(f)
                    total_pages = len(reader.pages)
                    print(f"PDF has {total_pages} pages, extracting text...")
                    for i, page in enumerate(reader.pages):
                        text += page.extract_text() + "\n"
                        if (i + 1) % 50 == 0:
                            print(f"  Extracted {i + 1}/{total_pages} pages...", end='\r')
                    print(f"\n✓ Extracted {len(text)} characters from PDF")
            except ImportError:
                print("PDF support requires pypdf. Install with: pip install pypdf")
                return 0
            except Exception as e:
                print(f"Error reading PDF: {e}")
                return 0
        else:
            # Treat as text file
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            print(f"✓ Read {len(text)} characters from text file")
        
        if not text.strip():
            print("No text content found in file.")
            return 0
        
        # Adjust chunk size for large documents
        if len(text) > 100000:
            chunk_size = 2000
            chunk_overlap = 200
            print(f"Large document detected, using chunk size: {chunk_size}")
        
        # Simple chunking
        chunks = []
        for i in range(0, len(text), chunk_size - chunk_overlap):
            chunk = text[i:i + chunk_size]
            if chunk.strip():
                chunks.append(chunk)
        
        print(f"Created {len(chunks)} chunks from document")
        
        ids = [f"{os.path.basename(file_path)}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"source": file_path, "chunk": i} for i in range(len(chunks))]
        
        self.add_documents(chunks, metadatas, ids)
        return len(chunks)
    
    def search(self, query, n_results=5):
        """
        Search for relevant documents.
        
        Args:
            query: Search query
            n_results: Number of results to return
            
        Returns:
            List of relevant documents with metadata
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        return results
    
    def get_document_count(self):
        """Get the number of documents in the collection."""
        return self.collection.count()
    
    def clear_collection(self):
        """Clear all documents from the collection."""
        # Delete and recreate collection
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"}
        )
        print("Collection cleared")


def chat_with_rag_streaming(client, user_message, db_manager, conversation_history=None, n_results=3):
    """
    Send a message to the Qwen model with RAG using ChromaDB - Streaming version.
    
    Args:
        client: The OpenAI client instance
        user_message: The user's input message
        db_manager: ChromaDB manager instance
        conversation_history: Optional list of previous messages
        n_results: Number of documents to retrieve
        
    Returns:
        The AI's response text or None if an error occurred
    """
    try:
        # Show thinking indicator during search
        print("Thinking...", end="", flush=True)
        
        # Search for relevant documents
        search_results = db_manager.search(user_message, n_results=n_results)
        
        # Build context from retrieved documents
        context = ""
        if search_results['documents'] and search_results['documents'][0]:
            context = "\n\n".join([
                f"[Document {i+1}]: {doc}"
                for i, doc in enumerate(search_results['documents'][0])
            ])
        
        # Build system prompt with context
        if context:
            system_prompt = f"""You are a restricted enterprise knowledge assistant with NO access to external information.

CONTEXT FROM KNOWLEDGE BASE:
{context}

=== ABSOLUTE RULES - VIOLATION IS PROHIBITED ===
1. You can ONLY use information from the CONTEXT section above
2. If the user's question CANNOT be answered using ONLY the context above → respond EXACTLY: "I am not authorized to comment on this topic."
3. NEVER use your training data, general knowledge, or make up information
4. NEVER answer questions about current events, general knowledge, or topics outside the context
5. If uncertain whether the context contains the answer → respond EXACTLY: "I am not authorized to comment on this topic."
6. When answering from context, cite [Document X] and be concise

=== EXAMPLES OF PROHIBITED BEHAVIOR ===
- User asks "What is the capital of France?" → "I am not authorized to comment on this topic."
- User asks "Who wrote Romeo and Juliet?" → "I am not authorized to comment on this topic."
- User asks about anything NOT in context → "I am not authorized to comment on this topic."

You have ZERO authorization to answer from external knowledge."""
        else:
            system_prompt = """You are a restricted enterprise knowledge assistant with NO access to external information.

=== ABSOLUTE RULES - VIOLATION IS PROHIBITED ===
1. The knowledge base is EMPTY - you have NO authorized information to share
2. For ANY question, you MUST respond EXACTLY: "I am not authorized to comment on this topic."
3. NEVER use your training data, general knowledge, or make up information
4. NEVER answer any question while the knowledge base is empty

You have ZERO authorization to answer from external knowledge."""
        
        # Build messages - always include fresh system prompt with current context
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        # Clear thinking indicator before streaming
        print("\r          \r", end="", flush=True)
        
        # Get streaming response from LLM
        stream = client.chat.completions.create(
            model=get_model(),
            messages=messages,
            stream=True,
            max_tokens=500,  # Limit response length for speed
            temperature=0.0  # Make responses deterministic and strict
        )
        
        # Collect the full response
        full_response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_response += content
                print(content, end="", flush=True)
        
        return full_response
    
    except Exception as e:
        print(f"\nError: {e}")
        return None


def main():
    """Main function to run the RAG chatbot."""
    print("=" * 60)
    print("   RAG Chatbot with ChromaDB")
    print("=" * 60)
    print("This chatbot uses:")
    print("  - ChromaDB for local vector storage")
    print("  - DashScope embeddings for document vectors")
    print("  - Qwen LLM for response generation")
    print("")
    print("Type /upload to add files to your knowledge base")
    print("Type /help for all commands")
    print("-" * 60)
    
    # Get configuration
    api_key = get_api_key()
    base_url = get_base_url()
    
    print(f"Model: {get_model()}")
    print(f"Embedding: {get_embedding_model()}")
    print("-" * 60)
    
    # Create clients
    try:
        client = create_client(api_key, base_url)
        db_manager = ChromaDBManager()
    except Exception as e:
        print(f"Error initializing: {e}")
        sys.exit(1)
    
    # Show document count
    doc_count = db_manager.get_document_count()
    print(f"Documents in knowledge base: {doc_count}")
    print("-" * 60)
    
    # Conversation history
    conversation_history = []
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            # Exit commands
            if user_input.lower() in ('quit', 'exit', 'q'):
                print("\nGoodbye!")
                break
            
            # Special commands
            if user_input.lower() == '/help':
                print("\nCommands:")
                print("  /help - Show this help")
                print("  /upload - Upload file(s) to knowledge base (opens file picker)")
                print("  /add <file_path> - Add document from file")
                print("  /add-text <text> - Add text directly")
                print("  /count - Show document count")
                print("  /clear - Clear knowledge base")
                print("  /clear-history - Clear conversation")
                print("  /status - Show settings")
                print("  quit - Exit")
                continue
            
            if user_input.lower() == '/upload':
                print("\nFile Upload Options:")
                print("  1. Enter file path manually")
                print("  2. Drag & drop file (paste the path)")
                print("  3. Use system file picker (macOS)")
                print("")
                upload_choice = input("Select option (1/2/3): ").strip()
                
                if upload_choice == '3':
                    # Use macOS file picker via osascript
                    import subprocess
                    try:
                        result = subprocess.run(
                            ['osascript', '-e', 'POSIX path of (choose file with prompt "Select a file to upload to knowledge base")'],
                            capture_output=True, text=True
                        )
                        if result.returncode == 0:
                            file_path = result.stdout.strip()
                            print(f"\nSelected: {file_path}")
                            if os.path.exists(file_path):
                                chunks = db_manager.add_from_file(file_path)
                                print(f"✓ Added {chunks} chunks from {os.path.basename(file_path)}")
                            else:
                                print(f"✗ File not found: {file_path}")
                        else:
                            print("File selection cancelled.")
                    except Exception as e:
                        print(f"File picker not available: {e}")
                        print("Please enter file path manually.")
                else:
                    print("\nEnter file path (or drag & drop a file):")
                    file_path = input("Path: ").strip()
                    # Remove quotes if present (from drag & drop)
                    file_path = file_path.strip('"').strip("'")
                    
                    if os.path.exists(file_path):
                        chunks = db_manager.add_from_file(file_path)
                        print(f"✓ Added {chunks} chunks from {os.path.basename(file_path)}")
                    else:
                        print(f"✗ File not found: {file_path}")
                continue
            
            if user_input.lower().startswith('/add '):
                file_path = user_input[5:].strip()
                if os.path.exists(file_path):
                    chunks = db_manager.add_from_file(file_path)
                    print(f"Added {chunks} chunks from {file_path}")
                else:
                    print(f"File not found: {file_path}")
                continue
            
            if user_input.lower().startswith('/add-text '):
                text = user_input[10:].strip()
                if text:
                    db_manager.add_documents([text])
                    print("Text added to knowledge base")
                continue
            
            if user_input.lower() == '/count':
                print(f"Documents in knowledge base: {db_manager.get_document_count()}")
                continue
            
            if user_input.lower() == '/clear':
                db_manager.clear_collection()
                continue
            
            if user_input.lower() == '/clear-history':
                conversation_history = []
                print("Conversation history cleared")
                continue
            
            if user_input.lower() == '/status':
                print(f"\nStatus:")
                print(f"  Model: {get_model()}")
                print(f"  Embedding: {get_embedding_model()}")
                print(f"  Documents: {db_manager.get_document_count()}")
                print(f"  DB Location: ./chroma_db")
                continue
            
            if not user_input:
                continue
            
            # Get response with streaming
            print("\nAssistant: ", end="", flush=True)
            response = chat_with_rag_streaming(client, user_input, db_manager, conversation_history)
            print()  # New line after streaming completes
            
            if response:
                conversation_history.append({"role": "user", "content": user_input})
                conversation_history.append({"role": "assistant", "content": response})
                if len(conversation_history) > 20:
                    conversation_history = conversation_history[-20:]
            else:
                print("Sorry, I couldn't process your request.")
                
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    main()
