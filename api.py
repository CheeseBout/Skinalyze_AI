"""
FastAPI server cho Cosmetic RAG Chatbot
Cung cấp REST API endpoints để tích hợp vào các ứng dụng khác
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import os
from pathlib import Path
import time
from datetime import datetime
import base64
from PIL import Image
import io

# Import từ RAG_cosmetic.py
from RAG_cosmetic import (
    setup_api_key,
    load_or_create_vectorstore,
    setup_rag_chain,
    analyze_skin_image,
    CHAT_HISTORY_DIR
)

# =============================================================================
# KHỞI TẠO FASTAPI
# =============================================================================
app = FastAPI(
    title="Cosmetic RAG Chatbot API",
    description="API tư vấn mỹ phẩm sử dụng RAG (Retrieval-Augmented Generation) và Vision AI",
    version="1.0.0"
)

# Cấu hình CORS để cho phép gọi từ frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Trong production, thay bằng domain cụ thể
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# GLOBAL VARIABLES - Khởi tạo 1 lần khi server start
# =============================================================================
rag_chain = None
conversation_sessions = {}  # Lưu conversation context cho mỗi session_id

# =============================================================================
# PYDANTIC MODELS - Định nghĩa request/response schemas
# =============================================================================
class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = None  # Để duy trì context conversation

class ChatResponse(BaseModel):
    answer: str
    response_time: float
    session_id: str
    timestamp: str

class ImageAnalysisRequest(BaseModel):
    image_base64: str  # Ảnh dạng base64
    additional_text: Optional[str] = None  # Text bổ sung kèm ảnh
    session_id: Optional[str] = None

class ImageAnalysisResponse(BaseModel):
    skin_analysis: str
    product_recommendation: str
    severity_warning: Optional[str] = None
    response_time: float
    session_id: str
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    message: str
    vectorstore_status: str
    timestamp: str

# =============================================================================
# STARTUP EVENT - Khởi tạo RAG Chain khi server start
# =============================================================================
@app.on_event("startup")
async def startup_event():
    """Khởi tạo RAG chain khi server khởi động"""
    global rag_chain
    
    print("\n" + "=" * 80)
    print("🚀 KHỞI ĐỘNG COSMETIC RAG CHATBOT API SERVER")
    print("=" * 80)
    
    try:
        # 1. Setup API Key
        setup_api_key()
        
        # 2. Load/Create Vector Store
        db, embeddings = load_or_create_vectorstore()
        
        if db is None:
            print("\n❌ CẢNH BÁO: Không thể khởi tạo Vector Store!")
            print("   Server sẽ chạy nhưng các endpoint sẽ trả về lỗi.")
            return
        
        # 3. Setup RAG Chain
        rag_chain = setup_rag_chain(db)
        
        print("\n✅ Server đã sẵn sàng phục vụ!")
        print("📚 API Docs: http://localhost:8000/docs")
        print("=" * 80 + "\n")
        
    except Exception as e:
        print(f"\n❌ LỖI khi khởi động server: {e}")
        print("   Server sẽ chạy nhưng các endpoint sẽ trả về lỗi.\n")

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return HealthResponse(
        status="online",
        message="Cosmetic RAG Chatbot API đang hoạt động",
        vectorstore_status="ready" if rag_chain is not None else "not_initialized",
        timestamp=datetime.now().isoformat()
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Kiểm tra trạng thái server"""
    return HealthResponse(
        status="healthy" if rag_chain is not None else "unhealthy",
        message="RAG chain sẵn sàng" if rag_chain is not None else "RAG chain chưa được khởi tạo",
        vectorstore_status="ready" if rag_chain is not None else "not_initialized",
        timestamp=datetime.now().isoformat()
    )

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Endpoint chat với RAG chatbot
    
    - **question**: Câu hỏi của người dùng
    - **session_id**: ID phiên (optional) để duy trì context conversation
    """
    if rag_chain is None:
        raise HTTPException(
            status_code=503,
            detail="RAG chain chưa được khởi tạo. Vui lòng kiểm tra logs server."
        )
    
    try:
        start_time = time.time()
        
        # Tạo hoặc lấy session_id
        session_id = request.session_id or f"session_{int(time.time() * 1000)}"
        
        # Lấy conversation context của session
        if session_id not in conversation_sessions:
            conversation_sessions[session_id] = []
        
        conversation_context = conversation_sessions[session_id]
        
        # Tạo query với context
        if conversation_context:
            recent_context = conversation_context[-3:]  # Lấy 3 cặp gần nhất
            context_str = "\n".join([
                f"User đã hỏi: {ctx[0]}\nBot đã trả lời: {ctx[1][:200]}..." 
                for ctx in recent_context
            ])
            
            query_with_context = f"""LỊCH SỬ HỘI THOẠI GẦN ĐÂY:
{context_str}

CÂU HỎI HIỆN TẠI: {request.question}

Hãy trả lời dựa trên LỊCH SỬ và câu hỏi hiện tại."""
            response = rag_chain.invoke(query_with_context)
        else:
            response = rag_chain.invoke(request.question)
        
        # Lưu vào conversation context
        conversation_context.append((request.question, response))
        conversation_sessions[session_id] = conversation_context
        
        elapsed_time = time.time() - start_time
        
        return ChatResponse(
            answer=response,
            response_time=round(elapsed_time, 2),
            session_id=session_id,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý: {str(e)}")

@app.post("/analyze-image", response_model=ImageAnalysisResponse)
async def analyze_image_endpoint(
    image: UploadFile = File(...),
    additional_text: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None)
):
    """
    Endpoint phân tích ảnh da và tư vấn sản phẩm
    
    - **image**: File ảnh da (jpg, png, webp, etc.)
    - **additional_text**: Text bổ sung (optional)
    - **session_id**: ID phiên (optional)
    """
    if rag_chain is None:
        raise HTTPException(
            status_code=503,
            detail="RAG chain chưa được khởi tạo. Vui lòng kiểm tra logs server."
        )
    
    try:
        start_time = time.time()
        
        # Tạo session_id
        session_id = session_id or f"session_{int(time.time() * 1000)}"
        
        # Đọc file ảnh
        image_bytes = await image.read()
        
        # Lưu tạm file ảnh
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        temp_image_path = temp_dir / f"{int(time.time() * 1000)}_{image.filename}"
        
        with open(temp_image_path, "wb") as f:
            f.write(image_bytes)
        
        try:
            # Bước 1: Phân tích ảnh da
            skin_analysis = analyze_skin_image(str(temp_image_path))
            
            if not skin_analysis:
                raise HTTPException(status_code=400, detail="Không thể phân tích ảnh")
            
            # Kiểm tra mức độ nghiêm trọng
            analysis_upper = skin_analysis.upper()
            is_very_severe = 'RẤT NẶNG' in analysis_upper or 'RẤT NGHIÊM TRỌNG' in analysis_upper
            
            # Bước 2: Tạo query RAG
            if additional_text:
                if is_very_severe:
                    rag_query = f"""Tình trạng da (RẤT NGHIÊM TRỌNG - CẦN GẶP BÁC SĨ):
{skin_analysis}

Yêu cầu: {additional_text}

Gợi ý 1-2 sản phẩm HỖ TRỢ NHẸ NHÀNG (không thay thế điều trị y khoa). 
NHẤN MẠNH: Cần gặp bác sĩ da liễu."""
                else:
                    rag_query = f"""Tình trạng da (từ phân tích ảnh):
{skin_analysis}

Yêu cầu: {additional_text}

Tư vấn 2-3 sản phẩm CỤ THỂ phù hợp với MỨC ĐỘ."""
            else:
                if is_very_severe:
                    rag_query = f"""Tình trạng da (RẤT NGHIÊM TRỌNG - CẦN GẶP BÁC SĨ):
{skin_analysis}

Gợi ý 1-2 sản phẩm HỖ TRỢ NHẸ NHÀNG (không thay thế điều trị y khoa).
NHẤN MẠNH: Cần gặp bác sĩ da liễu."""
                else:
                    rag_query = f"""Tình trạng da (từ phân tích ảnh):
{skin_analysis}

Tư vấn 2-3 sản phẩm CỤ THỂ phù hợp với MỨC ĐỘ."""
            
            product_recommendation = rag_chain.invoke(rag_query)
            
            elapsed_time = time.time() - start_time
            
            severity_warning = None
            if is_very_severe:
                severity_warning = "⚠️ CẢNH BÁO: Tình trạng da RẤT NGHIÊM TRỌNG! Vui lòng đặt lịch gặp bác sĩ da liễu NGAY. Sản phẩm gợi ý chỉ mang tính HỖ TRỢ, KHÔNG THAY THẾ điều trị y khoa!"
            
            return ImageAnalysisResponse(
                skin_analysis=skin_analysis,
                product_recommendation=product_recommendation,
                severity_warning=severity_warning,
                response_time=round(elapsed_time, 2),
                session_id=session_id,
                timestamp=datetime.now().isoformat()
            )
            
        finally:
            # Xóa file tạm
            if temp_image_path.exists():
                temp_image_path.unlink()
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý ảnh: {str(e)}")

@app.post("/analyze-image-base64", response_model=ImageAnalysisResponse)
async def analyze_image_base64_endpoint(request: ImageAnalysisRequest):
    """
    Endpoint phân tích ảnh da từ base64 string
    
    - **image_base64**: Ảnh dạng base64
    - **additional_text**: Text bổ sung (optional)
    - **session_id**: ID phiên (optional)
    """
    if rag_chain is None:
        raise HTTPException(
            status_code=503,
            detail="RAG chain chưa được khởi tạo."
        )
    
    try:
        start_time = time.time()
        
        # Decode base64
        image_bytes = base64.b64decode(request.image_base64)
        
        # Lưu tạm file
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        temp_image_path = temp_dir / f"{int(time.time() * 1000)}.jpg"
        
        with open(temp_image_path, "wb") as f:
            f.write(image_bytes)
        
        try:
            # Phân tích ảnh
            skin_analysis = analyze_skin_image(str(temp_image_path))
            
            if not skin_analysis:
                raise HTTPException(status_code=400, detail="Không thể phân tích ảnh")
            
            # Kiểm tra mức độ
            is_very_severe = 'RẤT NẶNG' in skin_analysis.upper()
            
            # Tạo query RAG
            if request.additional_text:
                rag_query = f"""Tình trạng da: {skin_analysis}
Yêu cầu: {request.additional_text}
Tư vấn sản phẩm phù hợp."""
            else:
                rag_query = f"""Tình trạng da: {skin_analysis}
Tư vấn sản phẩm phù hợp."""
            
            product_recommendation = rag_chain.invoke(rag_query)
            
            elapsed_time = time.time() - start_time
            
            return ImageAnalysisResponse(
                skin_analysis=skin_analysis,
                product_recommendation=product_recommendation,
                severity_warning="⚠️ CẦN GẶP BÁC SĨ DA LIỄU!" if is_very_severe else None,
                response_time=round(elapsed_time, 2),
                session_id=request.session_id or f"session_{int(time.time() * 1000)}",
                timestamp=datetime.now().isoformat()
            )
            
        finally:
            if temp_image_path.exists():
                temp_image_path.unlink()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi: {str(e)}")

@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """Xóa conversation context của một session"""
    if session_id in conversation_sessions:
        del conversation_sessions[session_id]
        return {"message": f"Đã xóa session {session_id}", "status": "success"}
    else:
        raise HTTPException(status_code=404, detail="Session không tồn tại")

# =============================================================================
# RUN SERVER
# =============================================================================
if __name__ == "__main__":
    import uvicorn
    
    print("\n🚀 Khởi động FastAPI Server...")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("🔗 Alternative Docs: http://localhost:8000/redoc\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",  # Cho phép truy cập từ mọi IP
        port=8000,
        reload=False  # Tắt auto-reload để tránh load lại model nhiều lần
    )
