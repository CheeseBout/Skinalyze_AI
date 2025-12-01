from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.api.routes import router
from app.services.ai_engine import ai_engine
from app.services.rag_engine import rag_engine

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n🚀 STARTING SERVER...")
    
    # Load Models
    ai_engine.load_all_models()
    
    # Load RAG
    try:
        rag_engine.initialize()
    except Exception as e:
        print(f"⚠️ RAG Initialization failed: {e}")
        
    yield
    print("🛑 Shutting down...")

app = FastAPI(
    title="AI Dermatology & Cosmetic Consultant API",
    version="4.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)