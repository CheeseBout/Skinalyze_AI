from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import base64
import numpy as np
from typing import Dict
import os
from fastapi.responses import StreamingResponse
import cv2

app = FastAPI(title="AI Dermatology API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define classes for classification
SKIN_CLASSES = [
    'Acne', 'Actinic_Keratosis', 'Drug_Eruption', 'Eczema', 'Normal', 
    'Psoriasis', 'Rosacea', 'Seborrh_Keratoses', 'Sun_Sunlight_Damage', 
    'Tinea', 'Warts'
]

# Define skin condition classes
SKIN_CONDITION_CLASSES = ['Dry', 'Normal', 'Oily']

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLASSIFICATION_MODEL_PATH = os.path.join(BASE_DIR, "models/resnet50_skin_disease_complete.pth")
SEGMENTATION_MODEL_PATH = os.path.join(BASE_DIR, "models/medsam2_dermatology_best_aug2.pth")
SKIN_CONDITION_MODEL_PATH = os.path.join(BASE_DIR, "models/efficientnet-skin-conditions-classifier.pth")

# Load ResNet50 classification model
def load_classification_model():
    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    
    # Replace fc layer with Sequential (matching the saved model architecture with correct indices)
    model.fc = nn.Sequential(
        nn.Dropout(0.5),           # Index 0
        nn.Linear(num_features, 512),  # Index 1
        nn.ReLU(),                 # Index 2
        nn.BatchNorm1d(512),       # Index 3
        nn.Dropout(0.5),           # Index 4
        nn.Linear(512, len(SKIN_CLASSES))  # Index 5
    )
    
    if not os.path.exists(CLASSIFICATION_MODEL_PATH):
        print(f"Classification model not found at: {CLASSIFICATION_MODEL_PATH}")
        return None
    
    try:
        # Load checkpoint
        checkpoint = torch.load(CLASSIFICATION_MODEL_PATH, map_location=device, weights_only=False)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                print(f"Loaded from checkpoint - Epoch: {checkpoint.get('epoch', 'N/A')}")
                print(f"Test Accuracy: {checkpoint.get('test_acc', 'N/A')}")
                if 'class_names' in checkpoint:
                    print(f"Classes in checkpoint: {checkpoint['class_names']}")
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

# Load EfficientNet skin condition model
def load_skin_condition_model():
    if not os.path.exists(SKIN_CONDITION_MODEL_PATH):
        print(f"Skin condition model not found at: {SKIN_CONDITION_MODEL_PATH}")
        return None
    
    try:
        # Load checkpoint first to inspect structure
        checkpoint = torch.load(SKIN_CONDITION_MODEL_PATH, map_location=device, weights_only=False)
        
        # Check if it's EfficientNet-B0 (most common)
        model = models.efficientnet_b0(weights=None)
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, len(SKIN_CONDITION_CLASSES))
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                print(f"Skin condition model - Epoch: {checkpoint.get('epoch', 'N/A')}")
                print(f"Skin condition model - Accuracy: {checkpoint.get('test_acc', checkpoint.get('val_acc', 'N/A'))}")
                if 'class_names' in checkpoint:
                    print(f"Classes in checkpoint: {checkpoint['class_names']}")
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
        import traceback
        traceback.print_exc()
        
        # Try different EfficientNet variants
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

# Load SAM2 segmentation model
def load_segmentation_model():
    if not os.path.exists(SEGMENTATION_MODEL_PATH):
        print(f"Segmentation model not found at: {SEGMENTATION_MODEL_PATH}")
        return None
    
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        
        # Load checkpoint
        checkpoint = torch.load(SEGMENTATION_MODEL_PATH, map_location=device, weights_only=False)
        print(f"Segmentation checkpoint loaded with keys: {checkpoint.keys() if isinstance(checkpoint, dict) else 'model object'}")
        
        # Get config from checkpoint
        if isinstance(checkpoint, dict) and 'config' in checkpoint:
            config = checkpoint['config']
            print(f"Model config found: {config}")
            
            # Extract model config name from checkpoint
            sam2_config = config.get('sam2_config', 'sam2.1_hiera_t')
            print(f"Using SAM2 config: {sam2_config}")
            
            # Map config names to actual YAML files
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
            
            # Get the config file path
            config_file = config_map.get(sam2_config, 'configs/sam2.1/sam2.1_hiera_t.yaml')
            
            # Try to find the config file in SAM2 installation
            import sam2
            sam2_path = os.path.dirname(sam2.__file__)
            config_path = os.path.join(sam2_path, '..', config_file)
            
            if not os.path.exists(config_path):
                # Try alternative path
                config_path = os.path.join(sam2_path, config_file)
            
            if not os.path.exists(config_path):
                print(f"Config file not found at {config_path}, using model name directly")
                config_path = sam2_config
            
            print(f"Loading SAM2 with config: {config_path}")
            
            # Build SAM2 model without checkpoint (we'll load manually)
            sam2_model = build_sam2(
                config_file=config_path,
                ckpt_path=None,
                device=device,
                mode='eval',
                apply_postprocessing=False
            )
            
            # Load the state dict
            if 'model_state_dict' in checkpoint:
                # Load with strict=False to handle any architecture mismatches
                missing_keys, unexpected_keys = sam2_model.load_state_dict(
                    checkpoint['model_state_dict'], 
                    strict=False
                )
                if missing_keys:
                    print(f"Missing keys: {len(missing_keys)} keys")
                if unexpected_keys:
                    print(f"Unexpected keys: {len(unexpected_keys)} keys")
                print(f"Loaded model weights - Best Dice: {checkpoint.get('best_val_dice', 'N/A')}")
            
            # Create predictor
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

# Load YOLOv8 face detection model
def load_face_detection_model():
    if not os.path.exists(FACE_DETECTION_MODEL_PATH):
        print(f"Face detection model not found at: {FACE_DETECTION_MODEL_PATH}")
        return None
    
    try:
        # Try using ultralytics YOLO
        try:
            from ultralytics import YOLO
            model = YOLO(FACE_DETECTION_MODEL_PATH)
            model.to(device)
            print("YOLOv8 face detection model loaded successfully!")
            return model
        except ImportError:
            print("ultralytics not installed. Installing...")
            import subprocess
            subprocess.check_call(['pip', 'install', 'ultralytics'])
            from ultralytics import YOLO
            model = YOLO(FACE_DETECTION_MODEL_PATH)
            model.to(device)
            print("YOLOv8 face detection model loaded successfully!")
            return model
            
    except Exception as e:
        print(f"Error loading face detection model: {e}")
        import traceback
        traceback.print_exc()
        return None

# Initialize models
classification_model = load_classification_model()
segmentation_model = load_segmentation_model()
skin_condition_model = load_skin_condition_model()
face_detection_model = load_face_detection_model()

# Image preprocessing for classification
classify_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Image preprocessing for skin condition (EfficientNet typically uses 224x224)
condition_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.get("/")
async def root():
    return {
        "message": "AI Dermatology API",
        "status": {
            "classification_model": "loaded" if classification_model else "not loaded",
            "segmentation_model": "loaded" if segmentation_model else "not loaded",
            "skin_condition_model": "loaded" if skin_condition_model else "not loaded",
            "face_detection_model": "loaded" if face_detection_model else "not loaded"
        },
        "endpoints": {
            "classify_disease": "/api/classification-disease",
            "classify_condition": "/api/classification-condition",
            "segment": "/api/segmentation-disease",
            "detect_face": "/api/face-detection"
        },
        "disease_classes": SKIN_CLASSES,
        "condition_classes": SKIN_CONDITION_CLASSES
    }

@app.post("/api/classification-disease")
async def classify_skin_disease(file: UploadFile = File(...)) -> Dict:
    """
    Classify skin disease from an uploaded image.
    
    Args:
        file: Image file (JPEG, PNG, etc.)
    
    Returns:
        JSON with predicted class and confidence scores
    """
    if classification_model is None:
        raise HTTPException(status_code=503, detail="Classification model not loaded. Please check model file exists.")
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and preprocess image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        input_tensor = classify_transform(image).unsqueeze(0).to(device)
        
        # Perform inference
        with torch.no_grad():
            outputs = classification_model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            # Get all class probabilities
            all_probs = probabilities[0].cpu().numpy()
            
        # Prepare response
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
    Classify skin condition (Dry, Normal, Oily) from an uploaded image.
    
    Args:
        file: Image file (JPEG, PNG, etc.)
    
    Returns:
        JSON with predicted condition and confidence scores
    """
    if skin_condition_model is None:
        raise HTTPException(status_code=503, detail="Skin condition model not loaded. Please check model file exists.")
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and preprocess image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        input_tensor = condition_transform(image).unsqueeze(0).to(device)
        
        # Perform inference
        with torch.no_grad():
            outputs = skin_condition_model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            # Get all class probabilities
            all_probs = probabilities[0].cpu().numpy()
            
        # Prepare response
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
    Generate segmentation mask for skin lesion.
    
    Args:
        file: Image file (JPEG, PNG, etc.)
    
    Returns:
        JSON with base64 encoded mask image
    """
    if segmentation_model is None:
        raise HTTPException(
            status_code=503, 
            detail="Segmentation model not loaded. Please check if SAM2 is properly installed and model file exists."
        )
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        original_size = image.size
        image_np = np.array(image)
        
        # Set image in predictor
        segmentation_model.set_image(image_np)
        
        # Predict mask - automatic mode (no prompts)
        # This will segment the most prominent object in the image
        with torch.no_grad():
            masks, scores, logits = segmentation_model.predict(
                point_coords=None,
                point_labels=None,
                box=None,
                multimask_output=False,
            )
        
        # Get the mask with highest score
        if len(masks) > 0:
            mask = masks[0]  # First mask (best prediction)
        else:
            # If no mask predicted, create empty mask
            mask = np.zeros((image_np.shape[0], image_np.shape[1]), dtype=np.uint8)
        
        # Convert to uint8 if needed
        if mask.dtype == bool:
            mask = mask.astype(np.uint8) * 255
        elif mask.max() <= 1:
            mask = (mask * 255).astype(np.uint8)
        
        # Create PIL image from mask
        mask_image = Image.fromarray(mask)
        
        # Resize to original size if needed
        if mask_image.size != original_size:
            mask_image = mask_image.resize(original_size, Image.NEAREST)
        
        # Convert mask to base64
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)