import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import mediapipe as mp
import os
import io
import base64
import requests
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
        self.skin_condition_model = self._load_skin_condition_model()
        # Quan trọng: Gọi load face detection sau khi đã chắc chắn hàm tồn tại
        self.face_detector = self._load_face_detection_model()
        self.segmentation_model = self._load_segmentation_model()
        print("✅ AI Models Loaded.")

    def _load_face_detection_model(self):
        """Hàm này bị thiếu trong code cũ gây ra lỗi AttributeError"""
        try:
            print("⏳ Loading Face Detection...")
            mp_face = mp.solutions.face_detection
            # model_selection=0 cho khoảng cách gần (selfie), =1 cho khoảng cách xa
            return mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)
        except Exception as e:
            print(f"⚠️ Face Detection failed to load: {e}")
            return None

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
            
            state_dict = checkpoint.get('model_state_dict') or checkpoint.get('state_dict') or checkpoint
            model = models.efficientnet_b0(weights=None)
            
            # Dynamic layer sizing
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
        if not path.exists(): 
            print(f"⚠️ Segmentation model not found at {path}")
            return None
            
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            import sam2 # Import module để tìm đường dẫn cài đặt
            
            checkpoint = torch.load(path, map_location=device, weights_only=False)
            if not isinstance(checkpoint, dict) or 'config' not in checkpoint: return None
            
            # 1. Xác định tên config
            config_name = checkpoint['config'].get('sam2_config', 'sam2.1_hiera_t')
            config_filename = "sam2.1_hiera_t.yaml" if "t" in config_name else "sam2.1_hiera_s.yaml"
            
            # 2. MẸO QUAN TRỌNG: Tìm đường dẫn cài đặt của thư viện sam2 trong Docker
            # Thường là /usr/local/lib/python3.10/site-packages/sam2
            sam2_base_dir = os.path.dirname(sam2.__file__)
            
            # Tạo đường dẫn đích bên trong thư mục configs của thư viện
            # Cấu trúc đích: .../sam2/configs/sam2.1/sam2.1_hiera_t.yaml
            target_config_dir = os.path.join(sam2_base_dir, "configs", "sam2.1")
            os.makedirs(target_config_dir, exist_ok=True) # Tạo folder nếu chưa có
            
            target_config_path = os.path.join(target_config_dir, config_filename)
            
            # 3. Download config thẳng vào thư mục của thư viện
            if not os.path.exists(target_config_path):
                print(f"⬇️ Downloading config to library path: {target_config_path}...")
                url = f"https://raw.githubusercontent.com/facebookresearch/segment-anything-2/main/sam2/configs/sam2.1/{config_filename}"
                try:
                    response = requests.get(url, timeout=10)
                    if response.status_code == 200:
                        with open(target_config_path, 'wb') as f:
                            f.write(response.content)
                        print("✅ Config downloaded successfully.")
                    else:
                        print(f"❌ Cannot download config. Status: {response.status_code}")
                        return None
                except Exception as dl_err:
                     print(f"❌ Download error: {dl_err}")
                     return None

            # 4. Load model
            # Khi file nằm đúng trong folder configs của thư viện, ta chỉ cần gọi tên file tương đối
            relative_config_path = f"configs/sam2.1/{config_filename}"
            
            print(f"🔍 Loading SAM2 with relative config: {relative_config_path}")
            sam2_model = build_sam2(
                config_file=relative_config_path, # Hydra sẽ tìm thấy file này trong package
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
        
        black_bg = Image.new("RGB", image.size, (0,0,0))
        mask_pil = Image.fromarray(mask).convert("L").resize(image.size, Image.NEAREST)
        lesion_img = Image.composite(image, black_bg, mask_pil)
        
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
        if not self.face_detector: return True 
        try:
            res = self.face_detector.process(np.array(image))
            return bool(res.detections)
        except:
            return True # Fail safe

ai_engine = AIEngine()