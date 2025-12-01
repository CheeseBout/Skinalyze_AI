import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import mediapipe as mp
import os
import io
import base64
from PIL import Image
from app.core.config import settings

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AIEngine:
    def __init__(self):
        self.classification_model = None
        self.segmentation_model = None
        self.skin_condition_model = None
        self.face_detector = None
        self.transforms = {
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

    def load_all_models(self):
        print("⏳ Loading AI Models...")
        self.classification_model = self._load_classification_model()
        self.segmentation_model = self._load_segmentation_model()
        self.skin_condition_model = self._load_skin_condition_model()
        self.face_detector = self._load_face_detection_model()
        print("✅ AI Models Loaded.")

    def _load_classification_model(self):
        path = settings.MODEL_CLASSIFICATION
        if not path.exists():
            print(f"⚠️ Classification model missing: {path}")
            return None
        try:
            checkpoint = torch.load(path, map_location=device, weights_only=False)
            if not isinstance(checkpoint, dict):
                model = checkpoint
                model.to(device).eval()
                return model
            
            # Reconstruction logic from original code
            state_dict = checkpoint.get('model_state_dict') or checkpoint.get('state_dict') or checkpoint
            model = models.efficientnet_b0(weights=None)
            
            # Dynamic layer sizing based on weights
            linear1_out = 512
            linear2_out = 256
            num_classes = len(settings.SKIN_CLASSES)

            if 'classifier.1.weight' in state_dict:
                linear1_out = state_dict['classifier.1.weight'].shape[0]
            if 'classifier.5.weight' in state_dict:
                linear2_out = state_dict['classifier.5.weight'].shape[0]
            if 'classifier.8.weight' in state_dict:
                num_classes = state_dict['classifier.8.weight'].shape[0]

            model.classifier = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(1280, linear1_out),
                nn.BatchNorm1d(linear1_out),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(linear1_out, linear2_out),
                nn.BatchNorm1d(linear2_out),
                nn.ReLU(),
                nn.Linear(linear2_out, num_classes)
            )
            model.load_state_dict(state_dict, strict=False)
            model.to(device).eval()
            return model
        except Exception as e:
            print(f"❌ Error loading Classification Model: {e}")
            return None

    def _load_skin_condition_model(self):
        path = settings.MODEL_SKIN_CONDITION
        if not path.exists(): return None
        try:
            checkpoint = torch.load(path, map_location=device, weights_only=False)
            state_dict = checkpoint.get('model_state_dict') or checkpoint.get('state_dict') or checkpoint
            
            # Try variants as per original code
            for variant_fn in [models.efficientnet_b0, models.efficientnet_b1, models.efficientnet_b2]:
                try:
                    model = variant_fn(weights=None)
                    num_ftrs = model.classifier[1].in_features
                    model.classifier[1] = nn.Linear(num_ftrs, len(settings.SKIN_CONDITION_CLASSES))
                    model.load_state_dict(state_dict, strict=False)
                    model.to(device).eval()
                    return model
                except: continue
            return None
        except Exception as e:
            print(f"❌ Error loading Condition Model: {e}")
            return None

    def _load_segmentation_model(self):
        path = settings.MODEL_SEGMENTATION
        # Nếu chưa có file model, trả về None (hoặc thêm logic download ở đây)
        if not path.exists(): 
            print(f"⚠️ Segmentation model not found at {path}")
            return None
            
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            import requests # Cần import thêm requests
            
            checkpoint = torch.load(path, map_location=device, weights_only=False)
            if not isinstance(checkpoint, dict) or 'config' not in checkpoint: return None
            
            # 1. Xác định tên config
            config_name = checkpoint['config'].get('sam2_config', 'sam2.1_hiera_t')
            # Mapping tên file
            config_filename = "sam2.1_hiera_t.yaml" if "t" in config_name else "sam2.1_hiera_s.yaml"
            
            # 2. Tạo đường dẫn file config nằm ngay trong thư mục models của bạn cho an toàn
            local_config_path = settings.MODELS_DIR / config_filename
            
            # 3. Nếu chưa có file yaml, tải về từ GitHub chính chủ
            if not local_config_path.exists():
                print(f"⬇️ Downloading config {config_filename}...")
                url = f"https://raw.githubusercontent.com/facebookresearch/segment-anything-2/main/sam2/configs/sam2.1/{config_filename}"
                response = requests.get(url)
                if response.status_code == 200:
                    with open(local_config_path, 'wb') as f:
                        f.write(response.content)
                else:
                    print("❌ Cannot download SAM2 config")
                    return None

            # 4. Load model với đường dẫn config tuyệt đối
            sam2_model = build_sam2(
                config_file=str(local_config_path), # Dùng file vừa tải
                ckpt_path=None, 
                device=device, 
                mode='eval', 
                apply_postprocessing=False
            )
            
            if 'model_state_dict' in checkpoint:
                sam2_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            
            return SAM2ImagePredictor(sam2_model)
        except Exception as e:
            print(f"❌ Error loading SAM2: {e}")
            return None

    # --- Inference Methods ---

    def predict_disease(self, image: Image.Image, check_face: bool = False):
        if check_face and self.face_detector:
            if not self.detect_face(image):
                raise ValueError("No face detected")
        
        input_tensor = self.transforms['classification'](image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = self.classification_model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)
            
            idx = pred.item()
            if idx >= len(settings.SKIN_CLASSES): return "Unknown", 0.0, {}
            
            all_probs = probs[0].cpu().numpy()
            return settings.SKIN_CLASSES[idx], float(conf.item()), {settings.SKIN_CLASSES[i]: float(all_probs[i]) for i in range(len(settings.SKIN_CLASSES))}

    def predict_condition(self, image: Image.Image):
        if self.face_detector and not self.detect_face(image):
            raise ValueError("No face detected")
            
        input_tensor = self.transforms['condition'](image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = self.skin_condition_model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)
            all_probs = probs[0].cpu().numpy()
            return settings.SKIN_CONDITION_CLASSES[pred.item()], float(conf.item()), {settings.SKIN_CONDITION_CLASSES[i]: float(all_probs[i]) for i in range(len(settings.SKIN_CONDITION_CLASSES))}

    def segment_lesion(self, image: Image.Image):
        if not self.segmentation_model: return None
        img_np = np.array(image)
        self.segmentation_model.set_image(img_np)
        
        with torch.no_grad():
            masks, scores, _ = self.segmentation_model.predict(point_coords=None, point_labels=None, box=None, multimask_output=False)
        
        mask = masks[0] if len(masks) > 0 else np.zeros(img_np.shape[:2], dtype=np.uint8)
        mask = (mask * 255).astype(np.uint8) if mask.dtype != np.uint8 else mask
        
        # Create Black BG image
        black_bg = Image.new("RGB", image.size, (0,0,0))
        mask_pil = Image.fromarray(mask).convert("L").resize(image.size, Image.NEAREST)
        lesion_img = Image.composite(image, black_bg, mask_pil)
        
        # Base64 Encode
        buf_mask = io.BytesIO()
        mask_pil.save(buf_mask, format="PNG")
        
        buf_lesion = io.BytesIO()
        lesion_img.save(buf_lesion, format="JPEG")
        
        return {
            "mask": base64.b64encode(buf_mask.getvalue()).decode('utf-8'),
            "lesion_on_black": base64.b64encode(buf_lesion.getvalue()).decode('utf-8'),
            "confidence": float(scores[0]) if len(scores) > 0 else 0.0
        }

    def detect_face(self, image: Image.Image) -> bool:
        if not self.face_detector: return True # Fail open if model missing
        res = self.face_detector.process(np.array(image))
        return bool(res.detections)

ai_engine = AIEngine()