import os
import pypdf
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import re

def parse_text(file_path):
    """
    Parses the input file based on its extension and returns the text content.
    """
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext == '.txt':
        return _read_txt(file_path)
    elif ext == '.pdf':
        return _read_pdf(file_path)
    elif ext == '.epub':
        return _read_epub(file_path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

def _read_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def _read_pdf(file_path):
    text = []
    with open(file_path, 'rb') as f:
        reader = pypdf.PdfReader(f)
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text.append(content)
    return "\n".join(text)

def _read_epub(file_path):
    book = epub.read_epub(file_path)
    text = []
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            # Extract HTML content
            soup = BeautifulSoup(item.get_content(), 'html.parser')
            # Get text and clean it up
            content = soup.get_text(separator='\n')
            text.append(content)
    return "\n".join(text)

def chunk_text(text, max_chars=400):
    """
    Splits text into chunks roughly within the char limit.
    Using a simple regex split for sentences/paragraphs to avoid breaking mid-sentence.
    """
    # Simply splitting by double newlines or punctuation for now
    # A more robust approach might use NLTK or spacy, but keeping dependencies low
    
    # Split by paragraphs first
    paragraphs = text.split('\n')
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
            
        # VibeVoice/F5-TTS sweet spot is generally 300-500 chars
        if len(current_chunk) + len(para) > max_chars:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = para
        else:
            if current_chunk:
                current_chunk += " " + para # Use space instead of newline for VibeVoice cleaner joining
            else:
                current_chunk = para
    
    if current_chunk:
        chunks.append(current_chunk)
        
    return chunks
