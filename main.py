"""
AI Dermatology & Cosmetic Consultant API
Stateless API for NestJS Backend Integration
"""

# =============================================================================
# IMPORTS
# =============================================================================
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
from typing import Dict, Optional, List
import os
import time
from datetime import datetime

from RAG_cosmetic import (
    setup_api_key,
    load_or_create_vectorstore,
    setup_rag_chain,
    analyze_skin_image,
    check_severity,
    build_image_analysis_query
)

# =============================================================================
# FASTAPI APP
# =============================================================================
app = FastAPI(
    title="AI Dermatology & Cosmetic Consultant API",
    description="Stateless API for skin disease classification, segmentation, and cosmetic consultation",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# CONFIGURATION
# =============================================================================
SKIN_CLASSES = [
    'Acne', 'Actinic_Keratosis', 'Drug_Eruption', 'Eczema', 'Normal', 
    'Psoriasis', 'Rosacea', 'Seborrh_Keratoses', 'Sun_Sunlight_Damage', 
    'Tinea', 'Warts'
]

SKIN_CONDITION_CLASSES = ['Dry', 'Normal', 'Oily']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATHS = {
    'classification': os.path.join(BASE_DIR, "models/resnet50_skin_disease_complete.pth"),
    'segmentation': os.path.join(BASE_DIR, "models/medsam2_dermatology_best_aug2.pth"),
    'skin_condition': os.path.join(BASE_DIR, "models/efficient-net-skin-conditions-classifier.pth")
}

IMAGE_TRANSFORMS = {
    'classification': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'condition': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}

# =============================================================================
# GLOBAL STATE
# =============================================================================
class AppState:
    rag_chain = None
    classification_model = None
    segmentation_model = None
    skin_condition_model = None

state = AppState()

# =============================================================================
# PYDANTIC MODELS
# =============================================================================
class ChatRequest(BaseModel):
    question: str
    conversation_history: Optional[List[Dict[str, str]]] = None  # [{"role": "user|ai", "content": "..."}]

class ChatResponse(BaseModel):
    answer: str
    response_time: float
    timestamp: str

class ImageAnalysisRequest(BaseModel):
    image_base64: str
    additional_text: Optional[str] = None

class ImageAnalysisResponse(BaseModel):
    skin_analysis: str
    product_recommendation: str
    severity_warning: Optional[str] = None
    response_time: float
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
# MODEL LOADING
# =============================================================================
def load_classification_model():
    """Load ResNet50 classification model"""
    model_path = MODEL_PATHS['classification']
    if not os.path.exists(model_path):
        print(f"⚠️  Classification model not found")
        return None
    
    try:
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
        
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get('model_state_dict') or checkpoint.get('state_dict') or checkpoint
        
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        print(f"✅ Classification model loaded")
        return model
    except Exception as e:
        print(f"❌ Error loading classification model: {e}")
        return None

def load_skin_condition_model():
    """Load EfficientNet skin condition model"""
    model_path = MODEL_PATHS['skin_condition']
    if not os.path.exists(model_path):
        print(f"⚠️  Skin condition model not found")
        return None
    
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get('model_state_dict') or checkpoint.get('state_dict') or checkpoint
        
        for variant_name, variant_fn in [
            ('EfficientNet-B0', models.efficientnet_b0),
            ('EfficientNet-B1', models.efficientnet_b1),
            ('EfficientNet-B2', models.efficientnet_b2),
        ]:
            try:
                model = variant_fn(weights=None)
                num_features = model.classifier[1].in_features
                model.classifier[1] = nn.Linear(num_features, len(SKIN_CONDITION_CLASSES))
                model.load_state_dict(state_dict, strict=False)
                model.to(device)
                model.eval()
                print(f"✅ Skin condition model loaded ({variant_name})")
                return model
            except:
                continue
        return None
    except Exception as e:
        print(f"❌ Error loading skin condition model: {e}")
        return None

def load_segmentation_model():
    """Load SAM2 segmentation model"""
    model_path = MODEL_PATHS['segmentation']
    if not os.path.exists(model_path):
        print(f"⚠️  Segmentation model not found")
        return None
    
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        import sam2
        
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if not isinstance(checkpoint, dict) or 'config' not in checkpoint:
            return None
        
        config = checkpoint['config']
        sam2_config = config.get('sam2_config', 'sam2.1_hiera_t')
        
        config_map = {
            'sam2.1_hiera_t': 'configs/sam2.1/sam2.1_hiera_t.yaml',
            'sam2.1_hiera_s': 'configs/sam2.1/sam2.1_hiera_s.yaml',
        }
        
        config_file = config_map.get(sam2_config, 'configs/sam2.1/sam2.1_hiera_t.yaml')
        sam2_path = os.path.dirname(sam2.__file__)
        config_path = os.path.join(sam2_path, '..', config_file)
        
        if not os.path.exists(config_path):
            config_path = os.path.join(sam2_path, config_file)
        
        sam2_model = build_sam2(
            config_file=config_path,
            ckpt_path=None,
            device=device,
            mode='eval',
            apply_postprocessing=False
        )
        
        if 'model_state_dict' in checkpoint:
            sam2_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
        predictor = SAM2ImagePredictor(sam2_model)
        print("✅ SAM2 segmentation model loaded")
        return predictor
        
    except Exception as e:
        print(f"❌ Error loading segmentation model: {e}")
        return None

# =============================================================================
# STARTUP
# =============================================================================
@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    print("\n" + "=" * 80)
    print("🚀 STARTING AI DERMATOLOGY & COSMETIC API SERVER")
    print("=" * 80)
    
    # Load models
    state.classification_model = load_classification_model()
    state.segmentation_model = load_segmentation_model()
    state.skin_condition_model = load_skin_condition_model()
    
    # Initialize RAG
    try:
        setup_api_key()
        db, embeddings = load_or_create_vectorstore()
        
        if db is None:
            print("\n⚠️  Vector Store not initialized")
        else:
            state.rag_chain = setup_rag_chain(db)
            print("\n✅ RAG Chatbot ready")
        
        print("\n✅ Server ready!")
        print("📚 API Documentation: http://localhost:8000/docs")
        print("=" * 80 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error initializing RAG: {e}\n")

# =============================================================================
# HEALTH CHECK
# =============================================================================
@app.get("/", response_model=HealthResponse)
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if state.rag_chain else "degraded",
        message="AI Dermatology & Cosmetic API",
        vectorstore_status="ready" if state.rag_chain else "not_initialized",
        classification_model_status="loaded" if state.classification_model else "not_loaded",
        segmentation_model_status="loaded" if state.segmentation_model else "not_loaded",
        skin_condition_model_status="loaded" if state.skin_condition_model else "not_loaded",
        timestamp=datetime.now().isoformat()
    )

# =============================================================================
# RAG CHATBOT ENDPOINTS (STATELESS)
# =============================================================================
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Stateless chat endpoint - NestJS handles session management
    
    - **question**: User's question
    - **conversation_history**: Optional conversation context from NestJS DB
    """
    if state.rag_chain is None:
        raise HTTPException(status_code=503, detail="RAG chain not initialized")
    
    try:
        start_time = time.time()
        
        # Build query with history if provided
        query = request.question
        if request.conversation_history:
            # Convert NestJS format to context
            context_pairs = []
            for i in range(0, len(request.conversation_history) - 1, 2):
                if i + 1 < len(request.conversation_history):
                    user_msg = request.conversation_history[i]
                    ai_msg = request.conversation_history[i + 1]
                    if user_msg.get('role') == 'user' and ai_msg.get('role') == 'ai':
                        context_pairs.append((
                            user_msg.get('content', ''),
                            ai_msg.get('content', '')
                        ))
            
            if context_pairs:
                recent = context_pairs[-3:]  # Last 3 exchanges
                context_str = "\n".join([
                    f"User: {ctx[0]}\nAI: {ctx[1][:200]}..." 
                    for ctx in recent
                ])
                
                query = f"""CONVERSATION HISTORY:
{context_str}

CURRENT QUESTION: {request.question}

Answer based on history and current question."""
        
        # Get response
        response = state.rag_chain.invoke(query)
        
        return ChatResponse(
            answer=response,
            response_time=round(time.time() - start_time, 2),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/analyze-image", response_model=ImageAnalysisResponse)
async def analyze_image_endpoint(
    image: UploadFile = File(...),
    additional_text: Optional[str] = Form(None)
):
    """
    Analyze skin image - STATELESS (no file storage)
    
    - **image**: Skin image file
    - **additional_text**: Additional query text
    """
    if state.rag_chain is None:
        raise HTTPException(status_code=503, detail="RAG chain not initialized")
    
    try:
        start_time = time.time()
        
        # Read image directly into memory (no temp file)
        image_bytes = await image.read()
        
        # Analyze using bytes directly
        skin_analysis = analyze_skin_image(image_bytes)
        if not skin_analysis:
            raise HTTPException(status_code=400, detail="Cannot analyze image")
        
        # Check severity and build query
        is_severe = check_severity(skin_analysis)
        rag_query = build_image_analysis_query(skin_analysis, additional_text, is_severe)
        
        # Get recommendation
        product_recommendation = state.rag_chain.invoke(rag_query)
        
        return ImageAnalysisResponse(
            skin_analysis=skin_analysis,
            product_recommendation=product_recommendation,
            severity_warning="⚠️ SEVERE: Please consult a dermatologist immediately!" if is_severe else None,
            response_time=round(time.time() - start_time, 2),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/analyze-image-base64", response_model=ImageAnalysisResponse)
async def analyze_image_base64_endpoint(request: ImageAnalysisRequest):
    """
    Analyze skin image from base64 - STATELESS
    
    - **image_base64**: Base64 encoded image
    - **additional_text**: Additional query text
    """
    if state.rag_chain is None:
        raise HTTPException(status_code=503, detail="RAG chain not initialized")
    
    try:
        start_time = time.time()
        
        # Analyze using base64 directly (no temp file)
        skin_analysis = analyze_skin_image(request.image_base64)
        if not skin_analysis:
            raise HTTPException(status_code=400, detail="Cannot analyze image")
        
        is_severe = check_severity(skin_analysis)
        rag_query = build_image_analysis_query(skin_analysis, request.additional_text, is_severe)
        product_recommendation = state.rag_chain.invoke(rag_query)
        
        return ImageAnalysisResponse(
            skin_analysis=skin_analysis,
            product_recommendation=product_recommendation,
            severity_warning="⚠️ SEVERE: Please consult a dermatologist!" if is_severe else None,
            response_time=round(time.time() - start_time, 2),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# =============================================================================
# CLASSIFICATION ENDPOINTS
# =============================================================================
@app.post("/api/classification-disease")
async def classify_skin_disease(file: UploadFile = File(...)) -> Dict:
    """Classify skin disease"""
    if state.classification_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        input_tensor = IMAGE_TRANSFORMS['classification'](image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = state.classification_model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            all_probs = probabilities[0].cpu().numpy()
        
        return {
            "predicted_class": SKIN_CLASSES[predicted.item()],
            "confidence": float(confidence.item()),
            "all_predictions": {SKIN_CLASSES[i]: float(all_probs[i]) for i in range(len(SKIN_CLASSES))}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/classification-condition")
async def classify_skin_condition(file: UploadFile = File(...)) -> Dict:
    """Classify skin condition"""
    if state.skin_condition_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        input_tensor = IMAGE_TRANSFORMS['condition'](image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = state.skin_condition_model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            all_probs = probabilities[0].cpu().numpy()
        
        return {
            "predicted_condition": SKIN_CONDITION_CLASSES[predicted.item()],
            "confidence": float(confidence.item()),
            "all_predictions": {SKIN_CONDITION_CLASSES[i]: float(all_probs[i]) for i in range(len(SKIN_CONDITION_CLASSES))}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/segmentation-disease")
async def segment_skin_lesion(file: UploadFile = File(...)) -> Dict:
    """Segment skin lesion"""
    if state.segmentation_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        original_size = image.size
        image_np = np.array(image)
        
        state.segmentation_model.set_image(image_np)
        
        with torch.no_grad():
            masks, scores, _ = state.segmentation_model.predict(
                point_coords=None, point_labels=None, box=None, multimask_output=False
            )
        
        mask = masks[0] if len(masks) > 0 else np.zeros(image_np.shape[:2], dtype=np.uint8)
        
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
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# RUN SERVER
# =============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)