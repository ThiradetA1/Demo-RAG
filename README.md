# DEMO Copilot

AI Chatbot สำหรับตอบคำถามในไฟล์ PDF 

---

## 🏗️ โครงสร้างโปรเจกต์

```
project/
├── app.html              #  Web App (ตัวหลัก)
├── create_db.py        # สร้าง/อัปเดต Vector Database จาก PDF
├── view_db.py          # ดูข้อมูลใน Database (สำหรับ debug)
├── requirements.txt    # Python dependencies (pinned versions)
├── .env.example        # ตัวอย่างไฟล์ config (คัดลอกเป็น .env)
├── .gitignore          # ป้องกัน API Key และ DB หลุดขึ้น Git
├── api.py              # สร้าง API เชื่อม HTML
└── documents/          # 📁 วางไฟล์ PDF ที่นี่ (สร้างโฟลเดอร์เอง)
```

---

## 🚀 วิธีติดตั้งและรัน

### 1. ติดตั้ง Dependencies

```bash
pip install -r requirements.txt
```

### 2. ตั้งค่า API Key

```bash
cp .env.example .env
```

แล้วแก้ไขไฟล์ `.env`:
```
GOOGLE_API_KEY=your_google_api_key_here
```

> 🔑 สร้าง API Key ได้ที่ [Google AI Studio](https://aistudio.google.com/app/apikey) (ฟรี)

### 3. เพิ่มเอกสาร PDF

สร้างโฟลเดอร์ `documents/` แล้วนำไฟล์ PDF ของบริษัทมาใส่:

```bash
mkdir documents
# วาง hr_policy.pdf, it_manual.pdf ฯลฯ ลงในโฟลเดอร์นี้
```

### 4. สร้าง Vector Database

```bash
python create_db.py
```

> ✅ รันซ้ำได้เมื่อเพิ่มไฟล์ PDF ใหม่ — ระบบจะเพิ่มเฉพาะไฟล์ที่ยังไม่มีใน DB โดยอัตโนมัติ

### 5. รัน Web App

```bash
uvicorn api:app --reload
```

เปิดเบราว์เซอร์ไปที่ `http://localhost:8501`

---

## 🛠️ คำสั่งที่ใช้บ่อย

| คำสั่ง | ผลลัพธ์ |
|--------|---------|
| `python create_db.py` | เพิ่มไฟล์ PDF ใหม่เข้า DB |
| `python view_db.py` | ดูว่า DB มีข้อมูลอะไรบ้าง |
| `uvicorn api:app --reload` | เปิด Web App |

---

## ⚠️ หมายเหตุสำคัญ

- **ห้าม** commit ไฟล์ `.env` และโฟลเดอร์ `hr_vector_db/` ขึ้น Git (มี `.gitignore` ป้องกันแล้ว)
- PDF ที่เป็นไฟล์สแกน (รูปภาพ) จะดึงข้อความไม่ได้ — ต้องใช้ไฟล์ PDF ที่มี text layer เท่านั้น
- ถ้าต้องการลบ DB แล้วสร้างใหม่ทั้งหมด ให้ลบโฟลเดอร์ `hr_vector_db/` ก่อนรัน `create_db.py`
