#!/usr/bin/env python3
"""
Improved loader script:
- Loads PDFs from a local folder using LangChain's PyPDFLoader
- Loads HTML pages (one-by-one) using LangChain's WebBaseLoader
- Produces a single JSON file with consistent entries for each document

Install required packages (if not already):
    pip install langchain langchain-community
    # langchain-community provides PyPDFLoader and WebBaseLoader in many setups
"""

import os
import json
from typing import List, Dict
from pathlib import Path

# LangChain community loaders
# Make sure you've installed langchain-community (pip install langchain-community)
try:
    from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
except Exception as e:
    raise RuntimeError(
        "Failed to import loaders from langchain_community. "
        "Install with: pip install langchain-community\nOriginal error: " + str(e)
    )


def load_pdfs_from_directory(directory_path: str) -> List[Dict]:
    """
    Scans a directory for PDFs, loads their content, and preserves metadata.
    Returns a list of dicts.
    """
    processed_docs: List[Dict] = []

    if not os.path.exists(directory_path):
        print(f"Warning: Directory '{directory_path}' not found.")
        return []

    print(f"--- Scanning '{directory_path}' for PDFs ---")

    for filename in sorted(os.listdir(directory_path)):
        if not filename.lower().endswith('.pdf'):
            continue

        file_path = os.path.join(directory_path, filename)
        try:
            loader = PyPDFLoader(file_path)
            pages = loader.load()  # returns list[Document]
            full_text = "\n".join([page.page_content for page in pages if getattr(page, "page_content", None)])
            num_pages = len(pages)

            doc_data = {
                "content": full_text.strip(),
                "source_type": "local_file",
                "source_location": os.path.abspath(file_path),
                "filename": filename,
                "num_pages": num_pages,
            }
            processed_docs.append(doc_data)
            print(f"[OK] Loaded PDF: {filename} ({num_pages} page(s))")

        except Exception as e:
            print(f"[ERROR] Loading PDF {filename}: {e}")

    return processed_docs


def load_html_from_urls(urls: List[str]) -> List[Dict]:
    """
    Loads content from specific HTML URLs, one-by-one.
    Returns a list of dicts.
    """
    processed_docs: List[Dict] = []

    print("\n--- Loading Data from HTML URLs ---")

    for url in urls:
        try:
            # instantiate loader per url for reliability
            loader = WebBaseLoader(url)
            docs = loader.load()  # list of Document objects

            # # WebBaseLoader may return multiple docs (e.g., if it splits by sections)
            # combined_text = "\n".join([d.page_content for d in docs if getattr(d, "page_content", None)])

            # metadata source fallback
            source_meta = None
            if docs:
                meta = getattr(docs[0], "metadata", {}) or {}
                source_meta = meta.get("source") or meta.get("url") or url

            # derive a filename-like label from URL
            safe_name = url.rstrip('/').split('/')[-1] or url.replace("https://", "").replace("/", "_")

            doc_data = {
                "content": combined_text.strip(),
                "source_type": "url",
                "source_location": source_meta or url,
                "filename": f"{safe_name}.html",
                "original_url": url,
            }
            processed_docs.append(doc_data)
            print(f"[OK] Loaded URL: {url} -> {doc_data['filename']} (chars: {len(doc_data['content'])})")

        except Exception as e:
            print(f"[ERROR] Loading URL {url}: {e}")

    return processed_docs


def main():
    # --- CONFIGURATION ---
    dataset_folder = r"D:\AI-Based Legal Reference and Case Retrieval System\data\pdf"
    # target_urls = [
    #     "https://indiankanoon.org/doc/1766147/",      # maneka_gandhi
    #     "https://indiankanoon.org/doc/127517806/",    # ks_puttaswamy
    #     "https://indiankanoon.org/doc/1031794/",      # vishaka
    #     "https://indiankanoon.org/doc/110813550/",    # shreya_singhal
    #     "https://indiankanoon.org/doc/168671544/",    # navtej_johar
    #     "https://indiankanoon.org/doc/823221/",       # shah_bano
    #     "https://indiankanoon.org/doc/25787660/",     # kesavananda_bharati
    #     "https://indiankanoon.org/doc/709776/",       # olga_tellis
    #     "https://indiankanoon.org/doc/193543754/",    # nalsa
    #     "https://indiankanoon.org/doc/1363234/",      # indira_sawhney
    # ]

    # --- EXECUTION ---
    all_documents: List[Dict] = []

    # 1. Load PDFs
    pdf_data = load_pdfs_from_directory(dataset_folder)
    all_documents.extend(pdf_data)

    # # 2. Load HTML pages (URLs)
    # html_data = load_html_from_urls(target_urls)
    # all_documents.extend(html_data)

    # --- OUTPUT ---
    output_filename = "loaded_documents.json"
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(all_documents, f, indent=4, ensure_ascii=False)

    print(f"\nProcess Complete!")
    print(f"Total documents loaded: {len(all_documents)}")
    print(f"Output saved to: {os.path.abspath(output_filename)}")


if __name__ == "__main__":
    main()
