import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
# โหลด API Key จากไฟล์ .env ก่อนสิ่งอื่นทุกอย่าง
load_dotenv()

# ==========================================
# 1. ตั้งค่าหน้าจอ Streamlit
# ==========================================
st.set_page_config(
    page_title="Enterprise AI Copilot",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ---- Sidebar ----
with st.sidebar:
    st.header("🤖 Enterprise AI Copilot")
    st.markdown("**เวอร์ชัน:** 1.2.0")
    st.divider()

    st.markdown("### 📌 ตัวอย่างคำถาม")
    example_questions = [
        "ลาป่วยต้องใช้ใบรับรองแพทย์ไหม?",
        "สวัสดิการประกันสุขภาพมีอะไรบ้าง?",
        "วิธีขอรีเซ็ตรหัสผ่าน IT?",
        "วันหยุดพักร้อนได้กี่วันต่อปี?",
    ]
    for q in example_questions:
        if st.button(q, use_container_width=True, key=q):
            st.session_state["prefill_input"] = q

    st.divider()

    # ปุ่มล้างแชท
    if st.button("🗑️ ล้างประวัติการแชท", use_container_width=True, type="secondary"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("ข้อมูลทั้งหมดมาจากคู่มือและนโยบายของบริษัท")

# ---- Header ----
st.title("🤖 HR & IT Helpdesk Copilot")
st.caption("พิมพ์คำถามของคุณเกี่ยวกับกฎระเบียบหรือคู่มือบริษัทได้เลยครับ")

# ==========================================
# 2. Validate environment ก่อนโหลด pipeline
# ==========================================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
VECTOR_DB_PATH = "./hr_vector_db"

def check_environment() -> str | None:
    """คืนค่า error message ถ้ามีปัญหา, None ถ้าปกติ"""
    if not GOOGLE_API_KEY:
        return "❌ ไม่พบ `GOOGLE_API_KEY` ในไฟล์ `.env` — กรุณาเพิ่ม key แล้วรีสตาร์ทแอป"
    if not Path(VECTOR_DB_PATH).exists():
        return f"❌ ไม่พบโฟลเดอร์ฐานข้อมูล `{VECTOR_DB_PATH}` — กรุณารัน ingest script ก่อน"
    return None

env_error = check_environment()
if env_error:
    st.error(env_error)
    st.stop()

# ==========================================
# 3. โหลดโมเดลและฐานข้อมูล (Cache ไม่โหลดซ้ำ)
# ==========================================
@st.cache_resource(show_spinner="⚙️ กำลังโหลดระบบ AI...")
def load_rag_pipeline():
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough, RunnableParallel
    from langchain_core.output_parsers import StrOutputParser

    embeddings_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    vector_db = Chroma(
        persist_directory=VECTOR_DB_PATH,
        embedding_function=embeddings_model,
    )
    retriever = vector_db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 10},
    )
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
        google_api_key=GOOGLE_API_KEY,
    )
    system_prompt = (
        "คุณคือ AI ผู้ช่วย (Enterprise Assistant) ประจำบริษัท\n"
        "หน้าที่ของคุณคือตอบคำถามพนักงานอย่างสุภาพ เป็นมืออาชีพ และอ่านง่าย\n"
        "จงใช้ 'ข้อมูลอ้างอิง' ด้านล่างนี้ในการตอบคำถามเท่านั้น\n"
        "หากในข้อมูลอ้างอิงไม่มีคำตอบ ให้ตอบว่า "
        "'ขออภัยครับ ไม่พบข้อมูลในระบบ กรุณาติดต่อ HR โดยตรง'\n"
        "ห้ามแต่งข้อมูลหรือคิดตัวเลขขึ้นมาเองเด็ดขาด\n\n"
        "ข้อมูลอ้างอิง:\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # FIX: ใช้ RunnableLambda แยก string ออกจาก dict ก่อนส่งให้ retriever และ prompt
    get_input_str = RunnableLambda(lambda x: x["input"])

    rag_chain = RunnableParallel(
        context=get_input_str | retriever,   # retriever รับ string ✅
        input=get_input_str,                 # เก็บ string ไว้ใช้ใน prompt ✅
    ).assign(
        answer=(
            RunnableLambda(lambda x: {
                "context": format_docs(x["context"]),
                "input": x["input"],
            })
            | prompt | llm | StrOutputParser()
        )
    )

    return rag_chain


try:
    rag_chain = load_rag_pipeline()
except Exception as e:
    st.error(f"❌ โหลด pipeline ไม่สำเร็จ: {e}")
    st.stop()

# ==========================================
# 4. ระบบจัดการแชท
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# Welcome message ครั้งแรกที่เปิดแอป
if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown(
            "สวัสดีครับ! 👋 ผมคือ AI Copilot ประจำบริษัท\n\n"
            "สามารถถามเรื่อง **HR** (ลา, สวัสดิการ, เงินเดือน) "
            "หรือ **IT** (รหัสผ่าน, ซอฟต์แวร์, อุปกรณ์) ได้เลยครับ"
        )

# แสดงแชทเก่า
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # แสดง sources ที่บันทึกไว้ (ถ้ามี)
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("📄 ดูเอกสารอ้างอิง", expanded=False):
                for i, src in enumerate(message["sources"], 1):
                    st.caption(f"**[{i}]** {src}")

# รับค่าจากปุ่มตัวอย่างใน sidebar (ถ้ามี)
prefill = st.session_state.pop("prefill_input", None)

# กล่องรับ input
user_input = st.chat_input("ตัวอย่าง: ลาป่วยต้องใช้ใบรับรองแพทย์ไหม?") or prefill

if user_input:
    # แสดงข้อความ User
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # เรียก RAG
    with st.chat_message("assistant"):
        with st.spinner("🔍 กำลังค้นหาข้อมูลจากคู่มือบริษัท..."):
            try:
                response = rag_chain.invoke({"input": user_input})
                answer = response["answer"]

                # ดึง sources จาก context documents
                sources = []
                for doc in response.get("context", []):
                    meta = doc.metadata
                    label = meta.get("source") or meta.get("file_name") or "ไม่ระบุแหล่งที่มา"
                    page = meta.get("page")
                    sources.append(f"{label}" + (f" (หน้า {page + 1})" if page is not None else ""))
                # ลบ duplicate
                sources = list(dict.fromkeys(sources))

            except Exception as e:
                answer = f"⚠️ เกิดข้อผิดพลาด: `{e}`\nกรุณาลองใหม่อีกครั้ง หรือติดต่อทีม IT"
                sources = []

        st.markdown(answer)

        if sources:
            with st.expander("📄 ดูเอกสารอ้างอิง", expanded=False):
                for i, src in enumerate(sources, 1):
                    st.caption(f"**[{i}]** {src}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })