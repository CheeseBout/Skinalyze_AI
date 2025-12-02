import os
import re
import google.generativeai as genai
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import base64
import io
from PIL import Image
from app.core.config import settings

class RAGEngine:
    def __init__(self):
        self.db = None
        self.rag_chain = None
        self.embeddings = None
        self.skin_map = {
            "acne": ["Hỗn hợp", "Dầu", "Nhạy cảm"],
            "mụn": ["Hỗn hợp", "Dầu", "Nhạy cảm"],
            "eczema": ["Hỗn hợp", "Khô", "Thường", "Dầu", "Nhạy cảm"],
            "chàm": ["Hỗn hợp", "Khô", "Thường", "Dầu", "Nhạy cảm"],
            "psoriasis": ["Khô"], "vảy nến": ["Khô"],
            "rosacea": ["Hỗn hợp", "Dầu", "Nhạy cảm"],
            "sun damage": ["Hỗn hợp", "Khô", "Thường", "Nhạy cảm"],
            "tinea": ["Hỗn hợp", "Dầu"], "nấm": ["Hỗn hợp", "Dầu"]
        }

    def initialize(self):
        print("⏳ Initializing RAG Engine...")
        if not settings.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY missing in .env")
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={'device': 'cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self._load_vector_store()
        if self.db:
            self._setup_chain()
        print("✅ RAG Engine Ready.")

    def _load_vector_store(self):
        if settings.DB_DIR.exists():
            self.db = Chroma(persist_directory=str(settings.DB_DIR), embedding_function=self.embeddings)
        elif settings.CHUNKS_FILE.exists():
            print("🆕 Creating new Vector Store...")
            loader = TextLoader(str(settings.CHUNKS_FILE), encoding='utf-8')
            docs = RecursiveCharacterTextSplitter(separators=["---"], chunk_size=400, chunk_overlap=50).split_documents(loader.load())
            
            for doc in docs:
                doc.metadata['product_name'] = self._extract_product_name(doc.page_content)
                
            self.db = Chroma.from_documents(docs, self.embeddings, persist_directory=str(settings.DB_DIR))
        else:
            print("⚠️ No data found to initialize Vector Store.")

    def _setup_chain(self):
        llm = ChatGoogleGenerativeAI(
            model=settings.GEMINI_TEXT_MODEL,
            temperature=0.1,
            max_output_tokens=1200,
            convert_system_message_to_human=True,
            max_retries=2
        )
        
        retriever = self.db.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 8, "fetch_k": 16, "lambda_mult": 0.7}
        )

        template = """Bạn là chuyên gia tư vấn mỹ phẩm. 
        CONTEXT: {context}
        HISTORY: {question}
        
        NHIỆM VỤ:
        - Nếu hỏi bệnh da: Xác định bệnh -> Gợi ý 2 sản phẩm.
        - Nếu hỏi loại da: Gợi ý 2 sản phẩm phù hợp.
        - Hiển thị giá VND (không USD).
        """
        
        prompt = ChatPromptTemplate.from_template(template)
        
        self.rag_chain = (
            {"context": retriever | self._format_docs, "question": RunnablePassthrough()}
            | prompt | llm | StrOutputParser()
        )

    def analyze_image(self, image_data, note: str = None) -> str:
        try:
            img = None
            if isinstance(image_data, bytes):
                img = Image.open(io.BytesIO(image_data))
            elif isinstance(image_data, str): # base64
                if "base64," in image_data: image_data = image_data.split(",")[1]
                img = Image.open(io.BytesIO(base64.b64decode(image_data)))
            
            model = genai.GenerativeModel(settings.GEMINI_VISION_MODEL)
            prompt = "Phân tích da: 1. Loại da 2. Vấn đề & Mức độ (NHẸ/TRUNG BÌNH/NẶNG/RẤT NẶNG). 3. Gợi ý ngắn."
            if note: prompt += f" Note: {note}"
            
            response = model.generate_content([prompt, img])
            return response.text
        except Exception as e:
            return f"Error analyzing image: {str(e)}"

    def check_severity(self, text: str) -> bool:
        return any(k in text.upper() for k in ['RẤT NẶNG', 'RẤT NGHIÊM TRỌNG', 'SEVERE'])

    def detect_condition(self, query: str):
        q = query.lower()
        for k, v in self.skin_map.items():
            if k in q: return k, v
        return None, None

    # --- Helpers ---
    def _extract_product_name(self, text):
        match = re.search(r'Product Name:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
        return match.group(1).strip() if match else "Unknown"

    def _format_docs(self, docs):
        if not docs: return "KHÔNG TÌM THẤY SẢN PHẨM"
        grouped = {}
        for d in docs:
            name = d.metadata.get('product_name', 'Unknown')
            if name not in grouped: grouped[name] = []
            grouped[name].append(d.page_content)
        
        # Take top 2
        result = []
        for name in list(grouped.keys())[:2]:
            text = "\n".join(grouped[name])
            # Convert Price
            text = re.sub(r'\$([0-9]+(?:\.[0-9]+)?)', 
                          lambda m: f"{int(float(m.group(1)) * settings.USD_TO_VND):,} VND", text)
            result.append(f"=== {name} ===\n{text}")
        return "\n\n".join(result)

    def extract_product_names(self, rag_response: str) -> list:
        """
        Trích xuất tên sản phẩm từ RAG response
        Returns: List of product names only
        """
        import re
        product_names = []
        
        # Pattern 1: Tìm các dòng có "=== Product Name ==="
        pattern1 = r'===\s*(.+?)\s*==='
        matches1 = re.findall(pattern1, rag_response)
        product_names.extend(matches1)
        
        # Pattern 2: Tìm "Product Name: ..."
        pattern2 = r'Product Name:\s*(.+?)(?:\n|$)'
        matches2 = re.findall(pattern2, rag_response, re.IGNORECASE)
        product_names.extend(matches2)
        
        # Pattern 3: Tìm các dòng bắt đầu bằng số (1. Product, 2. Product)
        pattern3 = r'\d+\.\s*([^\n]+?)(?:\s*-|\s*\(|$)'
        matches3 = re.findall(pattern3, rag_response)
        product_names.extend(matches3)
        
        # Loại bỏ trùng lặp và clean up
        seen = set()
        cleaned = []
        for name in product_names:
            name = name.strip()
            # Loại bỏ các ký tự đặc biệt ở cuối
            name = re.sub(r'[\:\-\(].*$', '', name).strip()
            if name and name not in seen and len(name) > 3:
                seen.add(name)
                cleaned.append(name)
        
        # Giới hạn tối đa 5 sản phẩm
        return cleaned[:5]

rag_engine = RAGEngine()