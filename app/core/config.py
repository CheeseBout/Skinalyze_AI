import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # Base Project Path (Root of the project)
    BASE_DIR = Path(__file__).resolve().parent.parent.parent

    # Data Directories
    DATA_DIR = BASE_DIR / "data"
    MODELS_DIR = BASE_DIR / "models"
    DB_DIR = BASE_DIR / "db_chroma"
    HISTORY_DIR = BASE_DIR / "chat_history"

    # File Paths
    CHUNKS_FILE = DATA_DIR / "product_chunks.txt"
    CSV_FILE = DATA_DIR / "cosmetics.csv"

    # Model Paths
    MODEL_CLASSIFICATION = MODELS_DIR / "efficientnet_b0_complete.pt"
    MODEL_SEGMENTATION = MODELS_DIR / "medsam2_dermatology_best_aug2.pth"
    MODEL_SKIN_CONDITION = MODELS_DIR / "efficient-net-skin-conditions-classifier.pth"

    # AI Settings
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    GEMINI_TEXT_MODEL = "gemini-2.0-flash"
    GEMINI_VISION_MODEL = "gemini-2.0-flash" # Updated based on availability logic
    
    # RAG Settings
    USD_TO_VND = 26349
    RETRIEVER_K = 2
    
    # Security
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

    # Classes
    SKIN_CLASSES = [
        'Acne', 'Actinic_Keratosis', 'Eczema', 'Normal', 'Psoriasis', 
        'Rosacea', 'Seborrh_Keratoses', 'Sun_Sunlight_Damage', 'Tinea', 'Warts'
    ]
    SKIN_CONDITION_CLASSES = ['Dry', 'Normal', 'Oily']

settings = Settings()