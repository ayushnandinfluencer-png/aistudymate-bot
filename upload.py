# ============================================================
#   AIStudyMate → Pinecone Upload Script (FINAL)
# ============================================================

import os
import re
import time
import uuid
import PyPDF2
from tqdm import tqdm
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# ============================================================
#   🔑 STEP 1 — ADD YOUR API KEY
# ============================================================

PINECONE_API_KEY = "pcsk_5okcB_SBiv2hHFw5Bo67XDw74jcufr7QJotbYkwigvqfNGDR9voAtJP2fxiSHJdozCesc"

# ============================================================
#   ⚙️ SETTINGS
# ============================================================

PDF_FOLDER      = r"F:\Merged Books"
INDEX_NAME      = "aistudymate"
BOARD           = "NCERT"
CHUNK_SIZE      = 500
CHUNK_OVERLAP   = 50
BATCH_SIZE      = 50

# ============================================================
#   🚀 LOAD EMBEDDING MODEL (384 DIMENSION)
# ============================================================

print("\n🔄 Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Model loaded (384 dimension)")

# ============================================================
#   📌 HELPER: Extract class & subject
# ============================================================

def parse_filename(filename):
    name = os.path.splitext(filename)[0]
    class_match = re.search(r'[Cc]lass(\d+)', name)
    class_num = class_match.group(1) if class_match else "Unknown"

    parts = name.split('_', 1)
    subject = parts[1] if len(parts) > 1 else name
    subject = subject.replace('Combined', '').strip()

    return class_num, subject

# ============================================================
#   📌 HELPER: Extract text from PDF
# ============================================================

def extract_pages(pdf_path):
    pages = []
    try:
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for i, page in enumerate(reader.pages, start=1):
                text = page.extract_text()
                if text and text.strip():
                    pages.append({
                        "page_number": i,
                        "text": text.strip()
                    })
    except Exception as e:
        print(f"❌ Error reading {pdf_path}: {e}")

    return pages

# ============================================================
#   📌 HELPER: Chunk text
# ============================================================

def chunk_text(text, page_number, class_num, subject, book_name):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + CHUNK_SIZE
        chunk = ' '.join(words[start:end])

        if len(chunk.strip()) > 50:
            chunks.append({
                "text": chunk,
                "page_number": page_number,
                "class": class_num,
                "subject": subject,
                "book_name": book_name,
                "board": BOARD
            })

        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks

# ============================================================
#   📌 EMBEDDING FUNCTION
# ============================================================

def get_embedding(text):
    try:
        return model.encode(text).tolist()
    except Exception as e:
        print("❌ Embedding error:", e)
        return None

# ============================================================
#   🚀 MAIN FUNCTION
# ============================================================

def main():
    print("\n" + "="*60)
    print("🚀 AIStudyMate → Pinecone Upload Started")
    print("="*60)

    # Connect Pinecone
    print("\n🔌 Connecting to Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)
    print("✅ Connected")

    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]
    print(f"\n📚 Found {len(pdf_files)} PDFs")

    total_uploaded = 0

    for pdf_file in tqdm(pdf_files, desc="📄 Processing PDFs"):

        class_num, subject = parse_filename(pdf_file)
        book_name = os.path.splitext(pdf_file)[0].replace('_', ' ')
        pdf_path = os.path.join(PDF_FOLDER, pdf_file)

        print(f"\n➡️ {pdf_file}")
        print(f"   Class: {class_num} | Subject: {subject}")

        pages = extract_pages(pdf_path)

        if not pages:
            print("   ⚠️ No text found")
            continue

        all_chunks = []

        for page in pages:
            all_chunks.extend(
                chunk_text(
                    page["text"],
                    page["page_number"],
                    class_num,
                    subject,
                    book_name
                )
            )

        print(f"   🔹 Total chunks: {len(all_chunks)}")

        batch = []

        for chunk in tqdm(all_chunks, desc="   🔄 Uploading", leave=False):
            emb = get_embedding(chunk["text"])
            if emb is None:
                continue

            batch.append({
                "id": str(uuid.uuid4()),
                "values": emb,
                "metadata": chunk
            })

            if len(batch) >= BATCH_SIZE:
                index.upsert(vectors=batch)
                total_uploaded += len(batch)
                batch = []
                time.sleep(0.2)

        if batch:
            index.upsert(vectors=batch)
            total_uploaded += len(batch)

    print("\n" + "="*60)
    print("🎉 UPLOAD COMPLETED SUCCESSFULLY!")
    print(f"📊 Total vectors uploaded: {total_uploaded}")
    print("="*60)

# ============================================================

if __name__ == "__main__":
    main()