"""
Chunk SEC 10-K MD&A text files for vector embedding.

Strategy:
- Split on paragraphs first (preserves semantic units)
- Then enforce max token length with overlap
- Each chunk gets metadata: source doc, company, chunk_id
"""

import json
import re
from pathlib import Path

from sentence_transformers import SentenceTransformer

# Approximate: 1 token ≈ 4 characters for English text
# We target 500 tokens ≈ 2000 chars, with 50-token overlap ≈ 200 chars
CHUNK_SIZE_CHARS = 2000
CHUNK_OVERLAP_CHARS = 200

FILINGS_DIR = Path(__file__).parent.parent.parent / "data" / "sec_filings"
OUTPUT_FILE = Path(__file__).parent.parent.parent / "data" / "sec_chunks" / "chunks.json"


def clean_text(text: str) -> str:
    """Normalize whitespace in text."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_into_sentences(text: str) -> list[str]:
    """
    Split text into sentences using simple punctuation rules.
    
    Works better than paragraph-splitting for dense financial text
    where paragraph breaks are inconsistent across filings.
    """
    # Normalize whitespace first
    text = re.sub(r"\s+", " ", text).strip()
    
    # Split on sentence-ending punctuation followed by space + capital letter
    # This preserves things like "$3.4 billion" or "Mr. Smith" without false splits
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    
    # Remove tiny fragments (likely artifacts)
    return [s.strip() for s in sentences if len(s.strip()) > 20]


def chunk_document(text: str, company: str, source_file: str) -> list[dict]:
    """
    Chunk a document into ~500-token pieces with sentence-level overlap.
    """
    sentences = split_into_sentences(text)
    
    chunks = []
    current_chunk_sentences = []
    current_chunk_len = 0
    chunk_num = 0
    
    for sentence in sentences:
        # Hard cap: if a single "sentence" is longer than CHUNK_SIZE_CHARS,
        # force-split it at chunk boundaries (rare in clean text, common in dense filings)
        if len(sentence) > CHUNK_SIZE_CHARS:
            # Flush current chunk first
            if current_chunk_sentences:
                chunk_text = " ".join(current_chunk_sentences)
                chunks.append({
                    "chunk_id": f"{company}_chunk_{chunk_num:03d}",
                    "company": company,
                    "source_file": source_file,
                    "text": chunk_text,
                    "char_count": len(chunk_text),
                })
                chunk_num += 1
                current_chunk_sentences = []
                current_chunk_len = 0
            
            # Force-split the giant sentence
            for start in range(0, len(sentence), CHUNK_SIZE_CHARS - CHUNK_OVERLAP_CHARS):
                piece = sentence[start:start + CHUNK_SIZE_CHARS]
                chunks.append({
                    "chunk_id": f"{company}_chunk_{chunk_num:03d}",
                    "company": company,
                    "source_file": source_file,
                    "text": piece,
                    "char_count": len(piece),
                })
                chunk_num += 1
            continue
        
        sent_len = len(sentence)
        
        # If adding this sentence would exceed chunk size, save current chunk
        if current_chunk_len + sent_len > CHUNK_SIZE_CHARS and current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            chunks.append({
                "chunk_id": f"{company}_chunk_{chunk_num:03d}",
                "company": company,
                "source_file": source_file,
                "text": chunk_text,
                "char_count": len(chunk_text),
            })
            chunk_num += 1
            
            # Overlap: keep last few sentences for context continuity
            overlap_sentences = []
            overlap_len = 0
            for sent in reversed(current_chunk_sentences):
                if overlap_len + len(sent) > CHUNK_OVERLAP_CHARS:
                    break
                overlap_sentences.insert(0, sent)
                overlap_len += len(sent)
            
            current_chunk_sentences = overlap_sentences + [sentence]
            current_chunk_len = sum(len(s) for s in current_chunk_sentences)
        else:
            current_chunk_sentences.append(sentence)
            current_chunk_len += sent_len
    
    # Don't forget the final chunk
    if current_chunk_sentences:
        chunk_text = " ".join(current_chunk_sentences)
        chunks.append({
            "chunk_id": f"{company}_chunk_{chunk_num:03d}",
            "company": company,
            "source_file": source_file,
            "text": chunk_text,
            "char_count": len(chunk_text),
        })
    
    return chunks

def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    all_chunks = []
    
    for filing_file in sorted(FILINGS_DIR.glob("*_10k_mda.txt")):
        company = filing_file.stem.split("_")[0]
        text = filing_file.read_text(encoding="utf-8")
        
        print(f"Chunking {filing_file.name} ({len(text):,} chars)...")
        
        chunks = chunk_document(text, company, filing_file.name)
        all_chunks.extend(chunks)
        
        print(f"  → {len(chunks)} chunks created")
    
    OUTPUT_FILE.write_text(json.dumps(all_chunks, indent=2, ensure_ascii=False))
    print(f"\n✅ Saved {len(all_chunks)} total chunks to {OUTPUT_FILE}")
    
    # Sanity check: print a sample chunk
    if all_chunks:
        sample = all_chunks[len(all_chunks) // 2]  # middle chunk
        print(f"\n📄 Sample chunk ({sample['chunk_id']}):")
        print(f"   Company: {sample['company']}")
        print(f"   Chars: {sample['char_count']}")
        print(f"   Preview: {sample['text'][:200]}...")


if __name__ == "__main__":
    main()