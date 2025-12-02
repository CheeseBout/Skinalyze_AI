from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from app.model.schemas import *
from app.services.ai_engine import ai_engine
from app.services.rag_engine import rag_engine
from PIL import Image
import io
import time
import json
from datetime import datetime

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy" if rag_engine.rag_chain else "degraded",
        message="AI Dermatology API",
        vectorstore_status="ready" if rag_engine.db else "not_initialized",
        classification_model_status="loaded" if ai_engine.classification_model else "not_loaded",
        segmentation_model_status="loaded" if ai_engine.segmentation_model else "not_loaded",
        skin_condition_model_status="loaded" if ai_engine.skin_condition_model else "not_loaded",
        timestamp=datetime.now().isoformat()
    )

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    question: str = Form(...),
    conversation_history: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None)
):
    if not rag_engine.rag_chain:
        raise HTTPException(503, "RAG not initialized")
    
    start = time.time()
    
    # 1. Parse History
    hist_list = []
    if conversation_history:
        try: hist_list = json.loads(conversation_history)
        except: pass
        
    # 2. VLM
    vlm_context = ""
    if image:
        content = await image.read()
        analysis = rag_engine.analyze_image(content, note=question)
        vlm_context = f"\n[IMAGE ANALYSIS]: {analysis}\n"

    # 3. Condition Detection
    cond, types = rag_engine.detect_condition(question)
    cond_context = f"\n[DETECTED]: {cond} (Types: {types})\n" if cond else ""
    
    # 4. History Context
    hist_str = ""
    if hist_list:
        recent = hist_list[-3:]
        hist_str = "HISTORY:\n" + "\n".join([f"{m.get('role')}: {m.get('content')}" for m in recent])

    # 5. RAG Invoke
    full_query = f"{hist_str}\n{vlm_context}\n{cond_context}\nQUESTION: {question}"
    answer = rag_engine.rag_chain.invoke(full_query)
    
    return ChatResponse(answer=answer, response_time=time.time()-start, timestamp=datetime.now().isoformat())

@router.post("/analyze-image", response_model=ImageAnalysisResponse)
async def analyze_image_file(image: UploadFile = File(...), additional_text: Optional[str] = Form(None)):
    start = time.time()
    content = await image.read()
    analysis = rag_engine.analyze_image(content)
    
    is_severe = rag_engine.check_severity(analysis)
    query = f"Analysis: {analysis}. User Note: {additional_text}. {'URGENT: SEVERE.' if is_severe else ''} Recommend products."
    
    rec = rag_engine.rag_chain.invoke(query)
    
    return ImageAnalysisResponse(
        skin_analysis=analysis,
        product_recommendation=rec,
        severity_warning="⚠️ SEVERE" if is_severe else None,
        response_time=time.time()-start,
        timestamp=datetime.now().isoformat()
    )

@router.post("/api/classification-disease")
async def classify_disease(file: UploadFile = File(...), notes: Optional[str] = Form(None)):
    if not ai_engine.classification_model: 
        raise HTTPException(503, "Model not loaded")
    
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    
    try:
        check_face = (notes == 'facial')
        cls, conf, all_preds = ai_engine.predict_disease(img, check_face)
        
        # Gọi RAG để gợi ý sản phẩm dựa trên bệnh da
        product_suggestions = []
        rag_response = ""
        fallback_used = False
        
        if rag_engine.rag_chain:
            try:
                # Bước 1: Thử tìm sản phẩm điều trị bệnh da
                query = f"Tôi bị bệnh da {cls}. Gợi ý sản phẩm điều trị."
                rag_response = rag_engine.rag_chain.invoke(query)
                product_suggestions = rag_engine.extract_product_names(rag_response)
                
                # Bước 2: Nếu không tìm thấy sản phẩm → fallback sang loại da
                if not product_suggestions or "KHÔNG TÌM THẤY" in rag_response:
                    fallback_used = True
                    # Suy ra loại da từ bệnh
                    skin_types = rag_engine.get_skin_types_from_disease(cls)
                    skin_types_str = ", ".join(skin_types)
                    
                    # Tạo query mới dựa trên loại da
                    query = f"Tôi có loại da {skin_types_str}. Gợi ý sản phẩm chăm sóc da phù hợp."
                    rag_response = rag_engine.rag_chain.invoke(query)
                    product_suggestions = rag_engine.extract_product_names(rag_response)
                    
            except Exception as e:
                print(f"RAG Error: {e}")
                product_suggestions = []
        
        return {
            "predicted_class": cls,
            "confidence": conf,
            "all_predictions": all_preds,
            "product_suggestions": product_suggestions,
            "suggestion_note": f"Sản phẩm cho bệnh {cls}" if not fallback_used else f"Sản phẩm cho loại da (từ {cls})"
        }
    except ValueError as e:
        raise HTTPException(400, str(e))

@router.post("/api/classification-condition")
async def classify_condition(file: UploadFile = File(...)):
    if not ai_engine.skin_condition_model: 
        raise HTTPException(503, "Model not loaded")
    
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    
    try:
        # Dự đoán tình trạng da (Dry, Normal, Oily)
        cond, conf, all_preds = ai_engine.predict_condition(img)
        
        # Kiểm tra khuôn mặt
        has_face = ai_engine.detect_face(img)

        # Gọi RAG để gợi ý sản phẩm dựa trên tình trạng da
        product_suggestions = []
        if rag_engine.rag_chain:
            # Mapping condition sang từ khóa tiếng Việt
            condition_mapping = {
                "Oily": "Dầu",
                "Dry": "Khô", 
                "Normal": "Thường"
            }
            
            skin_type_vn = condition_mapping.get(cond, cond)
            
            # Tạo query cho RAG dựa trên loại da
            query = f"Tôi có loại da {skin_type_vn}. Gợi ý sản phẩm chăm sóc da phù hợp."
            
            try:
                rag_response = rag_engine.rag_chain.invoke(query)
                product_suggestions = rag_engine.extract_product_names(rag_response)
            except Exception as e:
                print(f"RAG Error: {e}")

        return {
            "predicted_condition": cond, 
            "confidence": conf, 
            "all_predictions": all_preds, 
            "face_detected": has_face,
            "product_suggestions": product_suggestions,
            "suggestion_note": f"Sản phẩm cho da {cond}"
        }
    except ValueError as e:
        raise HTTPException(400, str(e))

@router.post("/api/segmentation-disease")
async def segment_lesion(file: UploadFile = File(...)):
    if not ai_engine.segmentation_model: raise HTTPException(503, "Model not loaded")
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    
    res = ai_engine.segment_lesion(img)
    if not res: raise HTTPException(500, "Segmentation failed")
    res["format"] = "base64_png"
    res["original_size"] = img.size
    return res

@router.post("/api/face-detection")
async def face_detection(file: UploadFile = File(...)):
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    return {"has_face": ai_engine.detect_face(img)}

@router.post("/api/analyze-skin-image-vlm", response_model=VLMAnalysisResponse)
async def analyze_vlm_only(file: UploadFile = File(...), note: Optional[str] = Form(None)):
    start = time.time()
    content = await file.read()
    analysis = rag_engine.analyze_image(content, note)
    if not analysis: raise HTTPException(500, "Analysis failed")
    return VLMAnalysisResponse(skin_analysis=analysis, response_time=time.time()-start, timestamp=datetime.now().isoformat())