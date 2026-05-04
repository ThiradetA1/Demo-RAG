import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

DOCUMENTS_DIR = "documents"
VECTOR_DB_DIR = "./hr_vector_db"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def build_database():
    print("🚀 เริ่มกระบวนการสร้าง Vector Database...")

    # ==========================================
    # 1. ค้นหาไฟล์ PDF ทั้งหมดในโฟลเดอร์ documents/
    # ==========================================
    docs_path = Path(DOCUMENTS_DIR)
    if not docs_path.exists():
        print(f"❌ ไม่พบโฟลเดอร์ '{DOCUMENTS_DIR}' กรุณาสร้างและนำไฟล์ PDF ใส่ก่อนครับ")
        return

    pdf_files = sorted(docs_path.glob("*.pdf"))
    if not pdf_files:
        print(f"❌ ไม่พบไฟล์ PDF ในโฟลเดอร์ '{DOCUMENTS_DIR}'")
        return

    print(f"📂 พบไฟล์ PDF ทั้งหมด {len(pdf_files)} ไฟล์:")
    for f in pdf_files:
        print(f"   • {f.name}")

    # ==========================================
    # 2. โหลด Libraries
    # ==========================================
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma

    # ==========================================
    # 3. โหลด Embeddings
    # ==========================================
    print("\n⏳ กำลังโหลดโมเดล Embeddings...")
    embeddings_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    print("✅ โหลดโมเดลสำเร็จ")

    # ==========================================
    # 4. FIX: เช็คไฟล์ที่มีใน DB แล้ว → ข้ามไฟล์ซ้ำ
    # ==========================================
    db_exists = Path(VECTOR_DB_DIR).exists()
    existing_sources: set[str] = set()

    if db_exists:
        vector_db = Chroma(
            persist_directory=VECTOR_DB_DIR,
            embedding_function=embeddings_model,
        )
        metadata = vector_db.get()["metadatas"]
        existing_sources = {
            str(Path(m.get("source", "")).name)
            for m in metadata
            if m.get("source")
        }
        if existing_sources:
            print(f"\n📋 ไฟล์ที่มีใน DB แล้ว ({len(existing_sources)} ไฟล์):")
            for s in sorted(existing_sources):
                print(f"   • {s}")

    # กรองเฉพาะไฟล์ใหม่
    new_pdf_files = [f for f in pdf_files if f.name not in existing_sources]

    if not new_pdf_files:
        print("\n✅ ไม่มีไฟล์ใหม่ที่ต้องเพิ่ม — DB เป็นปัจจุบันแล้ว")
        return

    print(f"\n🆕 ไฟล์ใหม่ที่จะเพิ่ม ({len(new_pdf_files)} ไฟล์):")
    for f in new_pdf_files:
        print(f"   • {f.name}")

    # ==========================================
    # 5. โหลดเอกสารใหม่
    # ==========================================
    all_documents = []
    for pdf_file in new_pdf_files:
        try:
            loader = PyPDFLoader(str(pdf_file))
            docs = loader.load()
            all_documents.extend(docs)
            print(f"   ✅ {pdf_file.name} ({len(docs)} หน้า)")
        except Exception as e:
            print(f"   ⚠️ ข้ามไฟล์ {pdf_file.name} — {e}")

    if not all_documents:
        print("❌ ไม่สามารถโหลดเอกสารได้เลย")
        return

    total_pages = len(all_documents)
    print(f"\n📄 รวม {total_pages} หน้า จาก {len(new_pdf_files)} ไฟล์ใหม่")

    # ==========================================
    # 6. หั่นข้อความ
    # ==========================================
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=250,
        separators=["\n\n", "\n", "。", ".", " ", ""],
    )
    chunks = text_splitter.split_documents(all_documents)
    print(f"✂️  หั่นข้อความสำเร็จ ({len(chunks)} ท่อน)")

    # ==========================================
    # 7. บันทึกลง ChromaDB
    # ==========================================
    print(f"\n⏳ กำลัง{'อัปเดต' if db_exists else 'สร้าง'} Vector DB...")

    if db_exists:
        vector_db.add_documents(chunks)
    else:
        Chroma.from_documents(
            documents=chunks,
            embedding=embeddings_model,
            persist_directory=VECTOR_DB_DIR,
        )

    # ==========================================
    # 8. สรุปผล
    # ==========================================
    print("\n" + "=" * 45)
    print("🎉 เสร็จสิ้น! สรุปการทำงาน:")
    print(f"   📁 ไฟล์ที่เพิ่ม    : {len(new_pdf_files)} ไฟล์")
    print(f"   📄 จำนวนหน้า      : {total_pages} หน้า")
    print(f"   ✂️  จำนวน chunks   : {len(chunks)} ท่อน")
    print(f"   💾 บันทึกที่       : {VECTOR_DB_DIR}")
    print("=" * 45)


if __name__ == "__main__":
    build_database()