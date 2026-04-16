#!/usr/bin/env python3
"""Upload documents to ChromaDB knowledge base."""

import os
import sys
from simple_rag_chromadb import ChromaDBManager, get_api_key, get_base_url, create_client

def upload_file(file_path):
    """Upload a file to the knowledge base."""
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return
    
    print(f"Uploading: {file_path}")
    
    # Initialize DB manager
    db_manager = ChromaDBManager()
    
    # Add file
    chunks = db_manager.add_from_file(file_path)
    print(f"✓ Added {chunks} chunks from {os.path.basename(file_path)}")
    print(f"Total documents in KB: {db_manager.get_document_count()}")

def upload_text(text, doc_id=None):
    """Upload raw text to the knowledge base."""
    if not text.strip():
        print("Error: Empty text")
        return
    
    # Initialize DB manager
    db_manager = ChromaDBManager()
    
    # Add text
    doc_id = doc_id or f"text_{db_manager.get_document_count()}"
    db_manager.add_documents([text], metadatas=[{"source": "manual_upload"}], ids=[doc_id])
    print(f"✓ Added text document")
    print(f"Total documents in KB: {db_manager.get_document_count()}")

def upload_directory(dir_path):
    """Upload all supported files from a directory."""
    if not os.path.isdir(dir_path):
        print(f"Error: Directory not found: {dir_path}")
        return
    
    supported_exts = ['.txt', '.md', '.pdf', '.py', '.js', '.html', '.json', '.csv']
    files = [f for f in os.listdir(dir_path) if any(f.lower().endswith(ext) for ext in supported_exts)]
    
    if not files:
        print(f"No supported files found in {dir_path}")
        return
    
    print(f"Found {len(files)} files to upload...")
    
    db_manager = ChromaDBManager()
    total_chunks = 0
    
    for filename in files:
        file_path = os.path.join(dir_path, filename)
        try:
            chunks = db_manager.add_from_file(file_path)
            total_chunks += chunks
            print(f"  ✓ {filename}: {chunks} chunks")
        except Exception as e:
            print(f"  ✗ {filename}: {e}")
    
    print(f"\nTotal: {total_chunks} chunks from {len(files)} files")
    print(f"Total documents in KB: {db_manager.get_document_count()}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Upload documents to ChromaDB")
    parser.add_argument("--file", "-f", help="Upload a single file")
    parser.add_argument("--dir", "-d", help="Upload all files from a directory")
    parser.add_argument("--text", "-t", help="Upload raw text")
    
    args = parser.parse_args()
    
    if args.file:
        upload_file(args.file)
    elif args.dir:
        upload_directory(args.dir)
    elif args.text:
        upload_text(args.text)
    else:
        print("Usage:")
        print("  python upload_to_kb.py --file document.pdf")
        print("  python upload_to_kb.py --dir ./my_documents")
        print("  python upload_to_kb.py --text 'Your text here'")
