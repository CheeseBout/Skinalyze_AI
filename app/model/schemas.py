from pydantic import BaseModel
from typing import List, Dict, Optional

class ChatRequest(BaseModel):
    question: str
    conversation_history: Optional[List[Dict[str, str]]] = None 

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

class VLMAnalysisResponse(BaseModel):
    skin_analysis: str
    response_time: float
    timestamp: str