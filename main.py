from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import base64
import numpy as np
from typing import Dict, Optional
import os
from pathlib import Path
import time
from datetime import datetime

# Import from RAG_cosmetic.py for chatbot functionality
from RAG_cosmetic import (
    setup_api_key,
    load_or_create_vectorstore,
    setup_rag_chain,
    analyze_skin_image,
    CHAT_HISTORY_DIR
)

app = FastAPI(
    title="AI Dermatology & Cosmetic Consultant API",
    description="Unified API for skin disease classification, segmentation, and cosmetic consultation using RAG and Vision AI",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# GLOBAL VARIABLES - For RAG Chatbot
# =============================================================================
rag_chain = None
conversation_sessions = {}  # Store conversation context for each session_id

# =============================================================================
# PYDANTIC MODELS - Request/Response Schemas for RAG Chatbot
# =============================================================================
class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    response_time: float
    session_id: str
    timestamp: str

class ImageAnalysisRequest(BaseModel):
    image_base64: str
    additional_text: Optional[str] = None
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
    classification_model_status: str
    segmentation_model_status: str
    skin_condition_model_status: str
    timestamp: str

# =============================================================================
# SKIN DISEASE CLASSIFICATION - Define classes
# =============================================================================
SKIN_CLASSES = [
    'Acne', 'Actinic_Keratosis', 'Drug_Eruption', 'Eczema', 'Normal', 
    'Psoriasis', 'Rosacea', 'Seborrh_Keratoses', 'Sun_Sunlight_Damage', 
    'Tinea', 'Warts'
]

SKIN_CONDITION_CLASSES = ['Dry', 'Normal', 'Oily']

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLASSIFICATION_MODEL_PATH = os.path.join(BASE_DIR, "models/resnet50_skin_disease_complete.pth")
SEGMENTATION_MODEL_PATH = os.path.join(BASE_DIR, "models/medsam2_dermatology_best_aug2.pth")
SKIN_CONDITION_MODEL_PATH = os.path.join(BASE_DIR, "models/efficient-net-skin-conditions-classifier.pth")

# =============================================================================
# MODEL LOADING FUNCTIONS
# =============================================================================
def load_classification_model():
    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.5),
        nn.Linear(512, len(SKIN_CLASSES))
    )
    
    if not os.path.exists(CLASSIFICATION_MODEL_PATH):
        print(f"Classification model not found at: {CLASSIFICATION_MODEL_PATH}")
        return None
    
    try:
        checkpoint = torch.load(CLASSIFICATION_MODEL_PATH, map_location=device, weights_only=False)
        
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                print(f"Loaded from checkpoint - Epoch: {checkpoint.get('epoch', 'N/A')}")
                print(f"Test Accuracy: {checkpoint.get('test_acc', 'N/A')}")
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        print(f"Classification model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading classification model: {e}")
        return None

def load_skin_condition_model():
    if not os.path.exists(SKIN_CONDITION_MODEL_PATH):
        print(f"Skin condition model not found at: {SKIN_CONDITION_MODEL_PATH}")
        return None
    
    try:
        checkpoint = torch.load(SKIN_CONDITION_MODEL_PATH, map_location=device, weights_only=False)
        
        model = models.efficientnet_b0(weights=None)
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, len(SKIN_CONDITION_CLASSES))
        
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                print(f"Skin condition model - Epoch: {checkpoint.get('epoch', 'N/A')}")
                print(f"Skin condition model - Accuracy: {checkpoint.get('test_acc', checkpoint.get('val_acc', 'N/A'))}")
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()
        print(f"Skin condition model loaded successfully!")
        return model
        
    except Exception as e:
        print(f"Error loading skin condition model: {e}")
        
        for variant_name, variant_model in [
            ('EfficientNet-B1', models.efficientnet_b1),
            ('EfficientNet-B2', models.efficientnet_b2),
            ('EfficientNet-B3', models.efficientnet_b3),
        ]:
            try:
                print(f"Trying {variant_name}...")
                model = variant_model(weights=None)
                num_features = model.classifier[1].in_features
                model.classifier[1] = nn.Linear(num_features, len(SKIN_CONDITION_CLASSES))
                
                checkpoint = torch.load(SKIN_CONDITION_MODEL_PATH, map_location=device, weights_only=False)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
                
                model.load_state_dict(state_dict, strict=False)
                model.to(device)
                model.eval()
                print(f"Skin condition model loaded as {variant_name}!")
                return model
            except:
                continue
        
        return None

def load_segmentation_model():
    if not os.path.exists(SEGMENTATION_MODEL_PATH):
        print(f"Segmentation model not found at: {SEGMENTATION_MODEL_PATH}")
        return None
    
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        
        checkpoint = torch.load(SEGMENTATION_MODEL_PATH, map_location=device, weights_only=False)
        print(f"Segmentation checkpoint loaded with keys: {checkpoint.keys() if isinstance(checkpoint, dict) else 'model object'}")
        
        if isinstance(checkpoint, dict) and 'config' in checkpoint:
            config = checkpoint['config']
            print(f"Model config found: {config}")
            
            sam2_config = config.get('sam2_config', 'sam2.1_hiera_t')
            print(f"Using SAM2 config: {sam2_config}")
            
            config_map = {
                'sam2.1_hiera_t': 'configs/sam2.1/sam2.1_hiera_t.yaml',
                'sam2.1_hiera_t512': 'configs/sam2.1/sam2.1_hiera_t.yaml',
                'sam2.1_hiera_s': 'configs/sam2.1/sam2.1_hiera_s.yaml',
                'sam2.1_hiera_b+': 'configs/sam2.1/sam2.1_hiera_b+.yaml',
                'sam2.1_hiera_l': 'configs/sam2.1/sam2.1_hiera_l.yaml',
                'sam2_hiera_t': 'configs/sam2/sam2_hiera_t.yaml',
                'sam2_hiera_s': 'configs/sam2/sam2_hiera_s.yaml',
                'sam2_hiera_b+': 'configs/sam2/sam2_hiera_b+.yaml',
                'sam2_hiera_l': 'configs/sam2/sam2_hiera_l.yaml',
            }
            
            config_file = config_map.get(sam2_config, 'configs/sam2.1/sam2.1_hiera_t.yaml')
            
            import sam2
            sam2_path = os.path.dirname(sam2.__file__)
            config_path = os.path.join(sam2_path, '..', config_file)
            
            if not os.path.exists(config_path):
                config_path = os.path.join(sam2_path, config_file)
            
            if not os.path.exists(config_path):
                print(f"Config file not found at {config_path}, using model name directly")
                config_path = sam2_config
            
            print(f"Loading SAM2 with config: {config_path}")
            
            sam2_model = build_sam2(
                config_file=config_path,
                ckpt_path=None,
                device=device,
                mode='eval',
                apply_postprocessing=False
            )
            
            if 'model_state_dict' in checkpoint:
                missing_keys, unexpected_keys = sam2_model.load_state_dict(
                    checkpoint['model_state_dict'], 
                    strict=False
                )
                if missing_keys:
                    print(f"Missing keys: {len(missing_keys)} keys")
                if unexpected_keys:
                    print(f"Unexpected keys: {len(unexpected_keys)} keys")
                print(f"Loaded model weights - Best Dice: {checkpoint.get('best_val_dice', 'N/A')}")
            
            predictor = SAM2ImagePredictor(sam2_model)
            print("SAM2 segmentation model loaded successfully!")
            return predictor
            
    except ImportError as ie:
        print(f"Import error: {ie}")
        print("Make sure SAM2 is properly installed")
        return None
    except Exception as e:
        print(f"Error loading segmentation model: {e}")
        import traceback
        traceback.print_exc()
        return None

# Initialize models
classification_model = load_classification_model()
segmentation_model = load_segmentation_model()
skin_condition_model = load_skin_condition_model()

# Image preprocessing
classify_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

condition_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# =============================================================================
# STARTUP EVENT - Initialize RAG Chain
# =============================================================================
@app.on_event("startup")
async def startup_event():
    """Initialize RAG chain when server starts"""
    global rag_chain
    
    print("\n" + "=" * 80)
    print("🚀 KHỞI ĐỘNG AI DERMATOLOGY & COSMETIC CONSULTANT API SERVER")
    print("=" * 80)
    
    try:
        # 1. Setup API Key
        setup_api_key()
        
        # 2. Load/Create Vector Store
        db, embeddings = load_or_create_vectorstore()
        
        if db is None:
            print("\n⚠️ CẢNH BÁO: Không thể khởi tạo Vector Store!")
            print("   Các endpoint RAG chatbot sẽ không hoạt động.")
        else:
            # 3. Setup RAG Chain
            rag_chain = setup_rag_chain(db)
            print("\n✅ RAG Chatbot đã sẵn sàng!")
        
        print("\n✅ Server đã sẵn sàng phục vụ!")
        print("📚 API Documentation: http://localhost:8000/docs")
        print("🔗 Alternative Docs: http://localhost:8000/redoc")
        print("=" * 80 + "\n")
        
    except Exception as e:
        print(f"\n❌ LỖI khi khởi tạo RAG chain: {e}")
        print("   Các endpoint classification/segmentation vẫn hoạt động bình thường.\n")

# =============================================================================
# API ENDPOINTS - Health Check & Root
# =============================================================================
@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with system status"""
    return HealthResponse(
        status="online",
        message="AI Dermatology & Cosmetic Consultant API đang hoạt động",
        vectorstore_status="ready" if rag_chain is not None else "not_initialized",
        classification_model_status="loaded" if classification_model else "not_loaded",
        segmentation_model_status="loaded" if segmentation_model else "not_loaded",
        skin_condition_model_status="loaded" if skin_condition_model else "not_loaded",
        timestamp=datetime.now().isoformat()
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check endpoint"""
    return HealthResponse(
        status="healthy",
        message="All systems operational",
        vectorstore_status="ready" if rag_chain is not None else "not_initialized",
        classification_model_status="loaded" if classification_model else "not_loaded",
        segmentation_model_status="loaded" if segmentation_model else "not_loaded",
        skin_condition_model_status="loaded" if skin_condition_model else "not_loaded",
        timestamp=datetime.now().isoformat()
    )

# =============================================================================
# API ENDPOINTS - RAG Chatbot (from api.py)
# =============================================================================
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Chat with RAG-powered cosmetic consultant
    
    - **question**: User's question about cosmetics/skincare
    - **session_id**: Session ID (optional) to maintain conversation context
    """
    if rag_chain is None:
        raise HTTPException(
            status_code=503,
            detail="RAG chain chưa được khởi tạo. Vui lòng kiểm tra logs server."
        )
    
    try:
        start_time = time.time()
        
        session_id = request.session_id or f"session_{int(time.time() * 1000)}"
        
        if session_id not in conversation_sessions:
            conversation_sessions[session_id] = []
        
        conversation_context = conversation_sessions[session_id]
        
        if conversation_context:
            recent_context = conversation_context[-3:]
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
    Analyze skin image and recommend cosmetic products
    
    - **image**: Skin image file (jpg, png, webp, etc.)
    - **additional_text**: Additional text query (optional)
    - **session_id**: Session ID (optional)
    """
    if rag_chain is None:
        raise HTTPException(
            status_code=503,
            detail="RAG chain chưa được khởi tạo."
        )
    
    try:
        start_time = time.time()
        
        session_id = session_id or f"session_{int(time.time() * 1000)}"
        
        image_bytes = await image.read()
        
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        temp_image_path = temp_dir / f"{int(time.time() * 1000)}_{image.filename}"
        
        with open(temp_image_path, "wb") as f:
            f.write(image_bytes)
        
        try:
            skin_analysis = analyze_skin_image(str(temp_image_path))
            
            if not skin_analysis:
                raise HTTPException(status_code=400, detail="Không thể phân tích ảnh")
            
            analysis_upper = skin_analysis.upper()
            is_very_severe = 'RẤT NẶNG' in analysis_upper or 'RẤT NGHIÊM TRỌNG' in analysis_upper
            
            if additional_text:
                if is_very_severe:
                    rag_query = f"""Tình trạng da (RẤT NGHIÊM TRỌNG - CẦN GẶP BÁC SĨ):
{skin_analysis}

Yêu cầu: {additional_text}

Gợi ý 1-2 sản phẩm HỖ TRỢ NHẸ NHÀNG (không thay thế điều trị y khoa). 
NHẤN MẠNH: Cần gặp bác sĩ da liễu."""
                else:
                    rag_query = f"""Tình trạng da:
{skin_analysis}

Yêu cầu: {additional_text}

Tư vấn 2-3 sản phẩm CỤ THỂ phù hợp."""
            else:
                if is_very_severe:
                    rag_query = f"""Tình trạng da (RẤT NGHIÊM TRỌNG):
{skin_analysis}

Gợi ý 1-2 sản phẩm HỖ TRỢ. NHẤN MẠNH: Cần gặp bác sĩ."""
                else:
                    rag_query = f"""Tình trạng da:
{skin_analysis}

Tư vấn 2-3 sản phẩm phù hợp."""
            
            product_recommendation = rag_chain.invoke(rag_query)
            
            elapsed_time = time.time() - start_time
            
            severity_warning = None
            if is_very_severe:
                severity_warning = "⚠️ CẢNH BÁO: Tình trạng da RẤT NGHIÊM TRỌNG! Vui lòng đặt lịch gặp bác sĩ da liễu NGAY!"
            
            return ImageAnalysisResponse(
                skin_analysis=skin_analysis,
                product_recommendation=product_recommendation,
                severity_warning=severity_warning,
                response_time=round(elapsed_time, 2),
                session_id=session_id,
                timestamp=datetime.now().isoformat()
            )
            
        finally:
            if temp_image_path.exists():
                temp_image_path.unlink()
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi: {str(e)}")

@app.post("/analyze-image-base64", response_model=ImageAnalysisResponse)
async def analyze_image_base64_endpoint(request: ImageAnalysisRequest):
    """
    Analyze skin image from base64 string
    
    - **image_base64**: Base64 encoded image
    - **additional_text**: Additional text query (optional)
    - **session_id**: Session ID (optional)
    """
    if rag_chain is None:
        raise HTTPException(status_code=503, detail="RAG chain chưa được khởi tạo.")
    
    try:
        start_time = time.time()
        
        image_bytes = base64.b64decode(request.image_base64)
        
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        temp_image_path = temp_dir / f"{int(time.time() * 1000)}.jpg"
        
        with open(temp_image_path, "wb") as f:
            f.write(image_bytes)
        
        try:
            skin_analysis = analyze_skin_image(str(temp_image_path))
            
            if not skin_analysis:
                raise HTTPException(status_code=400, detail="Không thể phân tích ảnh")
            
            is_very_severe = 'RẤT NẶNG' in skin_analysis.upper()
            
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
    """Delete conversation context for a session"""
    if session_id in conversation_sessions:
        del conversation_sessions[session_id]
        return {"message": f"Đã xóa session {session_id}", "status": "success"}
    else:
        raise HTTPException(status_code=404, detail="Session không tồn tại")

# =============================================================================
# API ENDPOINTS - Skin Disease Classification
# =============================================================================
@app.post("/api/classification-disease")
async def classify_skin_disease(file: UploadFile = File(...)) -> Dict:
    """
    Classify skin disease from an uploaded image
    
    Returns: JSON with predicted class and confidence scores
    """
    if classification_model is None:
        raise HTTPException(status_code=503, detail="Classification model not loaded")
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        input_tensor = classify_transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = classification_model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            all_probs = probabilities[0].cpu().numpy()
            
        result = {
            "predicted_class": SKIN_CLASSES[predicted.item()],
            "confidence": float(confidence.item()),
            "all_predictions": {
                SKIN_CLASSES[i]: float(all_probs[i]) 
                for i in range(len(SKIN_CLASSES))
            }
        }
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/api/classification-condition")
async def classify_skin_condition(file: UploadFile = File(...)) -> Dict:
    """
    Classify skin condition (Dry, Normal, Oily)
    
    Returns: JSON with predicted condition and confidence scores
    """
    if skin_condition_model is None:
        raise HTTPException(status_code=503, detail="Skin condition model not loaded")
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        input_tensor = condition_transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = skin_condition_model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            all_probs = probabilities[0].cpu().numpy()
            
        result = {
            "predicted_condition": SKIN_CONDITION_CLASSES[predicted.item()],
            "confidence": float(confidence.item()),
            "all_predictions": {
                SKIN_CONDITION_CLASSES[i]: float(all_probs[i]) 
                for i in range(len(SKIN_CONDITION_CLASSES))
            }
        }
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/api/segmentation-disease")
async def segment_skin_lesion(file: UploadFile = File(...)) -> Dict:
    """
    Generate segmentation mask for skin lesion
    
    Returns: JSON with base64 encoded mask image
    """
    if segmentation_model is None:
        raise HTTPException(
            status_code=503, 
            detail="Segmentation model not loaded"
        )
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        original_size = image.size
        image_np = np.array(image)
        
        segmentation_model.set_image(image_np)
        
        with torch.no_grad():
            masks, scores, logits = segmentation_model.predict(
                point_coords=None,
                point_labels=None,
                box=None,
                multimask_output=False,
            )
        
        if len(masks) > 0:
            mask = masks[0]
        else:
            mask = np.zeros((image_np.shape[0], image_np.shape[1]), dtype=np.uint8)
        
        if mask.dtype == bool:
            mask = mask.astype(np.uint8) * 255
        elif mask.max() <= 1:
            mask = (mask * 255).astype(np.uint8)
        
        mask_image = Image.fromarray(mask)
        
        if mask_image.size != original_size:
            mask_image = mask_image.resize(original_size, Image.NEAREST)
        
        buffer = io.BytesIO()
        mask_image.save(buffer, format="PNG")
        mask_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        return {
            "mask": mask_base64,
            "format": "base64_png",
            "original_size": original_size,
            "confidence": float(scores[0]) if len(scores) > 0 else 0.0
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

# =============================================================================
# RUN SERVER
# =============================================================================
if __name__ == "__main__":
    import uvicorn
    
    print("\n🚀 Khởi động Unified AI Dermatology & Cosmetic API Server...")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("🔗 Alternative Docs: http://localhost:8000/redoc\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)