from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import warnings

# ปิดข้อความเตือนยิบย่อยจะได้ดูผลลัพธ์ง่ายๆ
warnings.filterwarnings("ignore")

print("🔍 กำลังเชื่อมต่อฐานข้อมูล...")

# 1. โหลดตัวแปลง Embeddings ตัวเดิม (ต้องใช้กุญแจดอกเดิมถึงจะเปิดดูได้)
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# 2. เชื่อมต่อไปยังโฟลเดอร์ฐานข้อมูลของเรา
vector_db = Chroma(persist_directory="./hr_vector_db", embedding_function=embeddings_model)

# 3. ใช้คำสั่ง .get() เพื่อดึงข้อมูลทั้งหมดออกมาดู
data = vector_db.get()

total_chunks = len(data['documents'])
print(f"📊 จำนวนข้อมูลที่ถูกหั่นและเก็บไว้ทั้งหมด: {total_chunks} ท่อน (Chunks)\n")

if total_chunks > 0:
    print("👀 ลองสุ่มดูข้อมูล 3 ท่อนแรก:\n")
    # วนลูปปรินต์ดูข้อมูล 3 อันดับแรก (หรือเปลี่ยนตัวเลขตรง min(3, ...) เป็นจำนวนที่อยากดู)
    for i in range(min(6, total_chunks)):
        print(f"📌 ไอดี (ID): {data['ids'][i]}")
        print(f"📄 แหล่งที่มา (Metadata): {data['metadatas'][i]}")
        print(f"💬 เนื้อหาข้อความ:")
        print(data['documents'][i])
        print("=" * 50)
else:
    print("⚠️ ยังไม่มีข้อมูลในฐานข้อมูลครับ ลองรันไฟล์ create_db.py ดูก่อนนะ")