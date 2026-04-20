"""
Fetch MD&A sections from SEC 10-K filings.

SEC EDGAR requires a User-Agent header identifying the requester.
Reference: https://www.sec.gov/os/webmaster-faq#code-support
"""

import requests
from bs4 import BeautifulSoup
from pathlib import Path
import time

# SEC EDGAR requires User-Agent with contact email
HEADERS = {
    "User-Agent": "Alfiya Tamboli FIMEEN-Research alfiyaofficial94@gmail.com"
}

# 10-K filings (most recent full-year reports)
# Format: (ticker, filing_url)
FILINGS = [
    (
        "AAPL",
        "https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/aapl-20240928.htm"
    ),
    (
        "MSFT",
        "https://www.sec.gov/Archives/edgar/data/789019/000095017024087843/msft-20240630.htm"
    ),
    (
        "NVDA",
        "https://www.sec.gov/Archives/edgar/data/1045810/000104581025000023/nvda-20250126.htm"
    ),
]

OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "sec_filings"


def extract_mda_section(html: str) -> str:
    """
    Extract the MD&A section from a 10-K HTML filing.
    
    10-Ks have two "Item 7" references:
    1. In the Table of Contents (just the heading + page number)
    2. The actual MD&A section content
    
    We skip the first occurrence (TOC) and grab content between
    the second "Item 7" and the second "Item 7A" or "Item 8".
    """
    soup = BeautifulSoup(html, "lxml")
    full_text = soup.get_text(separator="\n")
    
    # Split into lines and find ALL occurrences of Item 7 markers
    lines = full_text.split("\n")
    
    item7_indices = []
    item7a_indices = []
    
    for i, line in enumerate(lines):
        clean = line.strip().lower()
        # Match "Item 7." or "Item 7 " at start of stripped line
        if clean.startswith("item 7.") or clean.startswith("item 7 "):
            # Distinguish Item 7 from Item 7A
            if "item 7a" in clean or clean.startswith("item 7a"):
                item7a_indices.append(i)
            else:
                item7_indices.append(i)
    
    # We need at least 2 occurrences of Item 7 (TOC + actual section)
    if len(item7_indices) < 2:
        print(f"  ⚠ Only {len(item7_indices)} 'Item 7' markers found, using first")
        start_idx = item7_indices[0] if item7_indices else 0
    else:
        # Use the SECOND occurrence — that's the actual MD&A start
        start_idx = item7_indices[1]
    
    # Find end: second occurrence of Item 7A (the real section, not TOC)
    if len(item7a_indices) >= 2:
        end_idx = item7a_indices[1]
    elif len(item7a_indices) == 1 and item7a_indices[0] > start_idx:
        end_idx = item7a_indices[0]
    else:
        # Fallback: take next 8000 lines or end of document
        end_idx = min(start_idx + 8000, len(lines))
    
    return "\n".join(lines[start_idx:end_idx])
    

def fetch_and_save(ticker: str, url: str) -> None:
    """Fetch a 10-K, extract MD&A, save to disk."""
    print(f"Fetching {ticker} from {url}")

    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()

    mda_text = extract_mda_section(response.text)

    # Clean up: collapse multiple whitespace into single spaces/newlines
    lines = [line.strip() for line in mda_text.split("\n") if line.strip()]
    cleaned = "\n".join(lines)

    output_path = OUTPUT_DIR / f"{ticker}_10k_mda.txt"
    output_path.write_text(cleaned, encoding="utf-8")

    char_count = len(cleaned)
    line_count = len(lines)
    print(f"  → Saved {output_path.name}: {char_count:,} chars, {line_count:,} lines")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for ticker, url in FILINGS:
        try:
            fetch_and_save(ticker, url)
            time.sleep(1)  # Be polite to SEC servers
        except Exception as e:
            print(f"  ✗ Failed for {ticker}: {e}")


if __name__ == "__main__":
    main()