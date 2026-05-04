import os
import logging
import json
from pathlib import Path
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import time

# ─── 1. ตั้งค่า Logging ───────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─── 2. โหลด Environment Variables ──────────────────────────────────────────
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# เปลี่ยนโฟลเดอร์เริ่มต้นให้เป็นชื่อกลางๆ สำหรับเอกสารทั่วไป
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./hr_vector_db")
MAX_INPUT_LENGTH = int(os.getenv("MAX_INPUT_LENGTH", "2000"))

# ─── 3. Global State ──────────────────────────────────────────────────────────
rag_chain = None
retriever = None  # แยก retriever ไว้ใช้กับ streaming

# ─── 4. โหลด RAG Pipeline ────────────────────────────────────────────────────
def load_rag_pipeline():
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableParallel, RunnableLambda
    from langchain_core.output_parsers import StrOutputParser

    logger.info("กำลังโหลด Embedding Model...")
    embeddings_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    logger.info("กำลังเชื่อมต่อ Vector Database...")
    vector_db = Chroma(
        persist_directory=VECTOR_DB_PATH,
        embedding_function=embeddings_model,
    )

    retr = vector_db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 10},
    )

    logger.info("กำลังเชื่อมต่อ Gemini LLM...")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
        google_api_key=GOOGLE_API_KEY,
    )

    # ปรับ System Prompt ให้เป็น AI ตอบคำถามเอกสารทั่วไป
    system_prompt = (
        "คุณคือ AI ผู้ช่วยวิเคราะห์และตอบคำถามจากเอกสาร (Document Q&A Assistant)\n"
        "หน้าที่ของคุณคือตอบคำถามของผู้ใช้อย่างตรงไปตรงมา เข้าใจง่าย และมีความเป็นมืออาชีพ\n"
        "จงใช้ 'ข้อมูลอ้างอิง' ด้านล่างนี้ในการตอบคำถามเท่านั้น\n"
        "หากในข้อมูลอ้างอิงไม่มีคำตอบ ให้ตอบอย่างชัดเจนว่า "
        "'ขออภัยครับ ไม่พบข้อมูลนี้ในเอกสารอ้างอิง'\n"
        "ห้ามแต่งข้อมูลเพิ่มเติมหรือคิดตัวเลขขึ้นมาเองโดยเด็ดขาด\n"
        "ตอบเป็นภาษาเดียวกับที่ผู้ใช้ถาม\n\n"
        "ข้อมูลอ้างอิง:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    get_input_str = RunnableLambda(lambda x: x["input"])

    chain = RunnableParallel(
        context=get_input_str | retr,
        input=get_input_str,
    ).assign(
        answer=(
            RunnableLambda(lambda x: {
                "context": format_docs(x["context"]),
                "input": x["input"],
            })
            | prompt | llm | StrOutputParser()
        )
    )

    return chain, retr

# ─── 5. Lifespan ──────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_chain, retriever

    if not GOOGLE_API_KEY:
        raise RuntimeError("ไม่พบ GOOGLE_API_KEY ในไฟล์ .env")
    if not Path(VECTOR_DB_PATH).exists():
        raise RuntimeError(f"ไม่พบโฟลเดอร์ฐานข้อมูล: {VECTOR_DB_PATH}")

    logger.info("⚙️  กำลังโหลด RAG Pipeline...")
    start = time.time()
    rag_chain, retriever = load_rag_pipeline()
    elapsed = time.time() - start
    logger.info(f"✅ โหลดระบบสำเร็จ ใช้เวลา {elapsed:.1f}s — พร้อมให้บริการ")

    yield

    logger.info("🛑 ปิดระบบเรียบร้อย")

# ─── 6. FastAPI App ───────────────────────────────────────────────────────────
# ปรับชื่อและคำอธิบาย API ให้เป็นกลาง
app = FastAPI(
    lifespan=lifespan,
    title="PDF Document Assistant API",
    version="2.0.0",
    description="RAG-powered API for answering questions from PDF documents.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── 7. Pydantic Models ───────────────────────────────────────────────────────
class Message(BaseModel):
    role: str  # "user" หรือ "assistant"
    content: str

class ChatRequest(BaseModel):
    input: str = Field(..., min_length=1, max_length=MAX_INPUT_LENGTH)
    history: Optional[List[Message]] = Field(default_factory=list)
    stream: bool = False

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    elapsed_ms: int

# ─── 8. Endpoints ─────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """ตรวจสอบสถานะระบบ"""
    return {
        "status": "ok" if rag_chain else "loading",
        "version": "2.0.0",
        "vector_db": VECTOR_DB_PATH,
    }

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, req: Request):
    """ตอบคำถามแบบปกติ (non-streaming)"""
    if not rag_chain:
        raise HTTPException(status_code=503, detail="ระบบ RAG ยังไม่พร้อม กรุณารอสักครู่")

    client_ip = req.client.host
    logger.info(f"[{client_ip}] Q: {request.input[:80]}...")

    start = time.time()
    try:
        response = rag_chain.invoke({"input": request.input})
        answer = response["answer"]
        sources = _extract_sources(response.get("context", []))
        elapsed_ms = int((time.time() - start) * 1000)

        logger.info(f"[{client_ip}] ตอบสำเร็จ ใช้เวลา {elapsed_ms}ms — {len(sources)} แหล่งอ้างอิง")
        return ChatResponse(answer=answer, sources=sources, elapsed_ms=elapsed_ms)

    except Exception as e:
        logger.error(f"[{client_ip}] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"เกิดข้อผิดพลาด: {str(e)}")


@app.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest, req: Request):
    """ตอบคำถามแบบ Streaming (Server-Sent Events)"""
    if not rag_chain or not retriever:
        raise HTTPException(status_code=503, detail="ระบบ RAG ยังไม่พร้อม")

    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    client_ip = req.client.host
    logger.info(f"[{client_ip}] STREAM Q: {request.input[:80]}...")

    async def event_generator():
        try:
            # ดึง context จาก retriever
            docs = retriever.invoke(request.input)
            context_text = "\n\n".join(doc.page_content for doc in docs)
            sources = _extract_sources(docs)

            # ส่ง sources ก่อน
            yield f"data: {json.dumps({'type': 'sources', 'sources': sources}, ensure_ascii=False)}\n\n"

            # Setup LLM + prompt
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0.2,
                google_api_key=GOOGLE_API_KEY,
                streaming=True,
            )
            # ปรับ System Prompt ในฝั่ง Stream ด้วยเช่นกัน
            system_prompt = (
                "คุณคือ AI ผู้ช่วยวิเคราะห์และตอบคำถามจากเอกสาร (Document Q&A Assistant)\n"
                "หน้าที่ของคุณคือตอบคำถามของผู้ใช้อย่างตรงไปตรงมา เข้าใจง่าย และมีความเป็นมืออาชีพ\n"
                "จงใช้ 'ข้อมูลอ้างอิง' ด้านล่างนี้ในการตอบคำถามเท่านั้น\n"
                "หากไม่มีคำตอบในข้อมูลอ้างอิง ให้ตอบว่า 'ขออภัยครับ ไม่พบข้อมูลนี้ในเอกสารอ้างอิง'\n"
                "ห้ามแต่งข้อมูลขึ้นมาเองเด็ดขาด\n\n"
                f"ข้อมูลอ้างอิง:\n{context_text}"
            )
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}"),
            ])
            chain = prompt | llm | StrOutputParser()

            # Stream ทีละ chunk
            async for chunk in chain.astream({"input": request.input}):
                yield f"data: {json.dumps({'type': 'token', 'token': chunk}, ensure_ascii=False)}\n\n"

            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            logger.info(f"[{client_ip}] STREAM เสร็จสิ้น")

        except Exception as e:
            logger.error(f"[{client_ip}] STREAM Error: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )

# ─── 9. Helper Functions ──────────────────────────────────────────────────────
def _extract_sources(docs) -> List[str]:
    sources = []
    for doc in docs:
        meta = doc.metadata
        label = meta.get("source") or meta.get("file_name") or "ไม่ระบุแหล่งที่มา"
        page = meta.get("page")
        entry = label + (f" (หน้า {page + 1})" if page is not None else "")
        sources.append(entry)
    return list(dict.fromkeys(sources))  # deduplicate