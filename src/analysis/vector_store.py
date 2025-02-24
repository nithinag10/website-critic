from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import tiktoken
from ..config.setting import OPENAI_API_KEY
import os
from typing import Dict, List
from langchain.docstore.document import Document
from uuid import uuid4
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
import re

def create_vector_store(documents, persist_dir: str) -> FAISS:
    """
    Creates and persists a FAISS vector store from text chunks.
    """
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    uuids = [str(uuid4()) for _ in range(len(documents))]

    index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))

    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    vector_store.add_documents(documents=documents, ids=uuids)
    vector_store.save_local(persist_dir)
    return vector_store


def read_analysis_file(file_path: str, website: str) -> str:
    """Reads the entire text file content."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def split_into_segments(text: str) -> List[str]:
    """
    Splits the text file into segments based on the segment header.
    Expected header format: "Segment" followed by some identifier and colon.
    
    Returns a list of segment texts.
    """
    # Remove header info before the segments begin.
    # Assuming the first line(s) before a line starting with "Segment" is header.
    pattern = r"(?m)^Segment\s.*?:"
    segments = re.split(pattern, text)
    
    # The first chunk is header info, so remove it if empty or not a segment.
    if segments and segments[0].strip().startswith("Folder:"):
        segments = segments[1:]
    
    # Re-add the header line to each segment if desired (optional)
    # If you want to preserve the segment header, you can instead use re.finditer.
    # For simplicity, we assume each segment chunk now is the complete segment text.
    return [seg.strip() for seg in segments if seg.strip()]

def get_all_analyses(base_dirs: Dict[str, str]) -> List[Document]:
    """Read analyses from all results.txt files and create one document per segment."""
    all_documents = []
    
    for category, base_dir in base_dirs.items():
        if not os.path.exists(base_dir):
            continue
            
        for website_dir in os.listdir(base_dir):
            website_path = os.path.join(base_dir, website_dir)
            if not os.path.isdir(website_path):
                continue
                
            results_file = os.path.join(website_path, "results.txt")
            if os.path.exists(results_file):
                content = read_analysis_file(results_file, website_dir)
                
                # Split the content into segments using the segment headers.
                segments = split_into_segments(content)
                
                # Create one Document per segment.
                for i, seg_text in enumerate(segments):
                    doc = Document(
                        page_content=seg_text,
                        metadata={
                            "website": website_dir,
                            "category": category,
                            "source": results_file,
                            "segment_index": i + 1
                        }
                    )
                    all_documents.append(doc)


    
    return all_documents
