#!/usr/bin/env python3
"""Reset the knowledge base - deletes all documents."""

import os
import shutil
import sys

def reset_knowledge_base():
    """Delete ChromaDB and start fresh."""
    chroma_db_path = "./chroma_db"
    
    if os.path.exists(chroma_db_path):
        print("Deleting knowledge base...")
        shutil.rmtree(chroma_db_path)
        print("✓ Knowledge base cleared!")
    else:
        print("No knowledge base found (already clean)")
    
    print("\nYou can now start fresh by running:")
    print("  python simple_rag_chromadb.py")

if __name__ == "__main__":
    confirm = input("This will DELETE all documents. Continue? (yes/no): ")
    if confirm.lower() == 'yes':
        reset_knowledge_base()
    else:
        print("Cancelled.")
