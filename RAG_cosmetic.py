"""
RAG Cosmetic Chatbot Core - Stateless for NestJS Backend Integration
No local file storage, no session management - all handled by NestJS
"""

import os
from pathlib import Path
import torch
from PIL import Image
import google.generativeai as genai
import base64
import io

from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# =============================================================================
# CONFIGURATION
# =============================================================================
PATH = Path(__file__).parent.resolve()
CHUNKS_FILE = PATH / "data" / "product_chunks.txt"
PERSIST_DIRECTORY = PATH / "db_chroma"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Global cache for embeddings
_CACHED_EMBEDDINGS = None

# =============================================================================
# API KEY SETUP
# =============================================================================
def setup_api_key():
    """Setup Google API Key"""
    if "GOOGLE_API_KEY" not in os.environ:
        print("\n🔑 Cần Google API Key để sử dụng Gemini")
        print("💡 Lấy key miễn phí tại: https://makersuite.google.com/app/apikey\n")
        api_key = "AIzaSyDLKLqpBHxf3xiutoYk5MjMzTywvju0Dx0"
        os.environ["GOOGLE_API_KEY"] = api_key
        print("✅ Đã thiết lập API Key!\n")
    else:
        print("✅ API Key đã được cấu hình sẵn!\n")
    
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# =============================================================================
# VECTOR STORE
# =============================================================================
def load_or_create_vectorstore():
    """Load or create vector store"""
    global _CACHED_EMBEDDINGS
    
    print("=" * 80)
    print("📚 KHỞI TẠO VECTOR STORE")
    print("=" * 80)
    
    try:
        # Load embedding model
        if _CACHED_EMBEDDINGS is not None:
            print(f"\n⚡ Sử dụng cached embedding model")
            embeddings = _CACHED_EMBEDDINGS
        else:
            print(f"\n⏳ Đang tải embedding model: {MODEL_NAME}...")
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"   🖥️ Sử dụng thiết bị: {device}")
            
            embeddings = HuggingFaceEmbeddings(
                model_name=MODEL_NAME,
                model_kwargs={'device': device},
                encode_kwargs={'normalize_embeddings': True}
            )
            _CACHED_EMBEDDINGS = embeddings
            print("✅ Đã tải embedding model!\n")

        # Load or create database
        if os.path.exists(PERSIST_DIRECTORY):
            print(f"📂 Loading Vector Store from: {PERSIST_DIRECTORY}")
            db = Chroma(
                persist_directory=str(PERSIST_DIRECTORY),
                embedding_function=embeddings
            )
            count = db._collection.count() if db._collection else 0
            print(f"✅ Loaded {count} documents\n")
        else:
            print(f"🆕 Creating new Vector Store from {CHUNKS_FILE.name}...\n")
            
            if not CHUNKS_FILE.exists():
                raise FileNotFoundError(f"File không tồn tại: {CHUNKS_FILE}")
            
            loader = TextLoader(str(CHUNKS_FILE), encoding='utf-8')
            documents = loader.load()
            print(f"   ✓ Loaded {len(documents)} document base")
            
            text_splitter = RecursiveCharacterTextSplitter(
                separators=["---"],
                chunk_size=400,
                chunk_overlap=50,
                length_function=len
            )
            docs = text_splitter.split_documents(documents)
            print(f"   ✓ Split into {len(docs)} chunks")
            
            print("💾 Creating embeddings and saving to database...")
            db = Chroma.from_documents(
                documents=docs,
                embedding=embeddings,
                persist_directory=str(PERSIST_DIRECTORY)
            )
            print(f"✅ Created Vector Store with {len(docs)} vectors\n")

        return db, embeddings
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return None, None

# =============================================================================
# RAG CHAIN SETUP
# =============================================================================
def setup_rag_chain(db):
    """Setup RAG chain"""
    print("\n" + "=" * 80)
    print("⛓️ KHỞI TẠO RAG CHAIN")
    print("=" * 80)
    
    # LLM
    print("\n🤖 [1/3] Connecting to Google Gemini...")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.1,
        max_output_tokens=512,
        convert_system_message_to_human=True,
        request_options={"timeout": 60},
        max_retries=2
    )
    print("   ✓ Connected to Gemini 2.0 Flash")
    
    # Retriever
    print("🔍 [2/3] Creating Retriever...")
    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 2,
            "fetch_k": 5,
            "lambda_mult": 0.7
        }
    )
    print("   ✓ Retriever ready (top 2 chunks, MMR)")
    
    # Prompt
    print("📝 [3/3] Creating Prompt Template...")
    template = """Bạn là chuyên gia tư vấn mỹ phẩm chuyên nghiệp, thân thiện và hiểu tâm lý khách hàng.

PHÂN LOẠI CÂU HỎI VÀ CÁCH TRẢ LỜI:

🔹 **CHÀO HỎI/GIAO TIẾP CƠ BẢN**
Câu hỏi: "xin chào", "hi", "hello", "chào bạn", "hey"
→ "Chào bạn! 👋 Mình là trợ lý tư vấn mỹ phẩm. Bạn muốn tìm sản phẩm gì hôm nay? 😊"

🔹 **HỎI VỀ CHỨC NĂNG/GIỚI THIỆU**
Câu hỏi: "bạn là ai", "bạn làm gì", "có thể giúp gì", "bạn biết gì"
→ "Mình là chuyên gia tư vấn mỹ phẩm! 💄 Mình có thể giúp bạn:
• Tìm sản phẩm theo loại da (khô, dầu, nhạy cảm, hỗn hợp, mụn...)
• Tư vấn kem dưỡng, serum, toner, mặt nạ, sữa rửa mặt, kem chống nắng
• Giải thích thành phần và công dụng sản phẩm
• Gợi ý routine chăm sóc da
Bạn đang gặp vấn đề gì về da hoặc cần tìm sản phẩm nào? 😊"

🔹 **HỎI VỀ VẤN ĐỀ DA**
Câu hỏi: "da tôi bị...", "tôi bị mụn", "da khô", "da dầu", "da nhạy cảm"
→ Phân tích vấn đề và GỢI Ý 1-2 sản phẩm CỤ THỂ từ database phù hợp nhất

🔹 **HỎI THEO LOẠI SẢN PHẨM**
Câu hỏi: "có kem dưỡng nào...", "serum gì tốt", "toner cho da...", "mặt nạ..."
→ Gợi ý 1-2 sản phẩm PHÙ HỢP từ database, nêu rõ CÔNG DỤNG và LOẠI DA phù hợp

🔹 **HỎI VỀ THƯƠNG HIỆU**
Câu hỏi: "bạn có [tên thương hiệu] không", "sản phẩm của [brand]"
→ Kiểm tra database, nếu có thì liệt kê, nếu không: "Mình chưa có thông tin về [brand] trong database. Bạn muốn tư vấn sản phẩm theo loại da hay vấn đề cụ thể không? 😊"

🔹 **HỎI SO SÁNH**
Câu hỏi: "A hay B tốt hơn", "khác nhau thế nào", "nên chọn cái nào"
→ So sánh 2 sản phẩm dựa trên THÀNH PHẦN, CÔNG DỤNG, LOẠI DA phù hợp

🔹 **HỎI GIÁ/MUA Ở ĐÂU**
Câu hỏi: "giá bao nhiêu", "mua ở đâu", "có ship không"
→ "Xin lỗi, mình chỉ tư vấn về sản phẩm thôi nhé. Bạn có thể mua tại các store chính hãng hoặc website của thương hiệu. Mình có thể tư vấn thêm về sản phẩm khác không? 😊"

🔹 **HỎI ROUTINE/CÁCH DÙNG**
Câu hỏi: "routine cho da...", "thứ tự dùng", "dùng như thế nào", "dùng khi nào"
→ Gợi ý routine cơ bản: Sáng (sữa rửa mặt → toner → serum → kem dưỡng → chống nắng), Tối (tương tự nhưng thay chống nắng = mặt nạ 2-3 lần/tuần)

🔹 **CẢM ƠN/TẠM BIỆT**
Câu hỏi: "cảm ơn", "thank you", "ok rồi", "tạm biệt", "bye"
→ "Không có gì! 😊 Chúc bạn có làn da đẹp! Hẹn gặp lại bạn! 💕"

🔹 **CÂU HỎI NGOÀI LỀ**
Câu hỏi: thời tiết, tin tức, thể thao, chính trị, toán học...
→ "Xin lỗi, mình chỉ chuyên về mỹ phẩm và skincare thôi 💄 Bạn có muốn hỏi về chăm sóc da không?"

---

**CHÚ Ý KHI TRẢ LỜI:**
- Luôn THÂN THIỆN, dùng "mình/bạn" thay vì "tôi/bạn" để gần gũi hơn
- Nếu tư vấn sản phẩm: TỐI ĐA 2 sản phẩm, nêu rõ TÊN - THƯƠNG HIỆU - CÔNG DỤNG - LOẠI DA
- Dùng emoji phù hợp: 😊💄✨💕👋
- Nếu KHÔNG chắc chắn: "Bạn có thể mô tả cụ thể hơn về [vấn đề] để mình tư vấn chính xác hơn không?"

THÔNG TIN SẢN PHẨM:
{context}

CÂU HỎI: {question}

TRẢ LỜI (ngắn gọn, 2-4 câu):"""
    
    prompt = ChatPromptTemplate.from_template(template)
    print("   ✓ Prompt Template created")
    
    # Build chain
    def format_docs(docs):
        formatted = []
        for i, doc in enumerate(docs, 1):
            content = doc.page_content.strip()
            if content:
                if len(content) > 500:
                    content = content[:500] + "..."
                formatted.append(f"[{i}] {content}")
        return "\n".join(formatted)
    
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    print("\n✅ RAG Chain ready!\n")
    return rag_chain

# =============================================================================
# VISION ANALYSIS - STATELESS (accepts PIL Image or base64)
# =============================================================================
def analyze_skin_image(image_input):
    """
    Analyze skin image - STATELESS version
    
    Args:
        image_input: Can be:
            - PIL Image object
            - base64 encoded string
            - file path string (for backward compatibility)
    
    Returns:
        str: Analysis result or None
    """
    try:
        print("\n📸 Analyzing skin image...")
        
        # Handle different input types
        if isinstance(image_input, str):
            # Check if it's base64 or file path
            if image_input.startswith('data:image'):
                # Remove data URL prefix
                image_input = image_input.split(',')[1]
            
            # Try to decode as base64
            try:
                image_bytes = base64.b64decode(image_input)
                img = Image.open(io.BytesIO(image_bytes))
            except:
                # Assume it's a file path
                img = Image.open(image_input)
        elif isinstance(image_input, Image.Image):
            img = image_input
        elif isinstance(image_input, bytes):
            img = Image.open(io.BytesIO(image_input))
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")
        
        # Initialize Gemini Vision model
        vision_model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # Prompt
        vision_prompt = """Bạn là chuyên gia da liễu. Phân tích ảnh da và TÓM TẮT NGẮN GỌN:

1. LOẠI DA: (khô/dầu/hỗn hợp/nhạy cảm/thường)

2. VẤN ĐỀ CHÍNH & MỨC ĐỘ NGHIÊM TRỌNG:
- Nếu có mụn: loại mụn (viêm/đầu đen/đầu trắng/bọc), mức độ (NHẸ/TRUNG BÌNH/NẶNG/RẤT NẶNG)
- Nếu có thâm/sẹo: mức độ (NHẸ/TRUNG BÌNH/NẶNG/RẤT NẶNG), màu sắc, phân bố
- Nếu có lão hóa: mức độ (NHẸ/TRUNG BÌNH/NẶNG)
- Nếu có vấn đề khác: nêu rõ

3. MỨC ĐỘ CHUNG: Chọn 1 trong 4:
   - NHẸ: Vấn đề nhỏ, ít nốt, có thể tự chăm sóc
   - TRUNG BÌNH: Vấn đề rõ ràng, nhiều nốt, cần sản phẩm chuyên dụng
   - NẶNG: Vấn đề lan rộng, viêm nhiều, cần điều trị tích cực
   - RẤT NẶNG: Viêm trầm trọng, sẹo nhiều, cần gặp bác sĩ da liễu

4. GỢI Ý: (1 câu ngắn)

QUAN TRỌNG: Phải ghi rõ MỨC ĐỘ (NHẸ/TRUNG BÌNH/NẶNG/RẤT NẶNG).

Trả lời NGẮN GỌN, bằng tiếng Việt."""
        
        # Call vision model
        response = vision_model.generate_content([vision_prompt, img])
        analysis = response.text
        
        print("✅ Analysis complete!")
        return analysis
        
    except Exception as e:
        print(f"❌ Error analyzing image: {str(e)}")
        return None

# =============================================================================
# HELPER FUNCTIONS FOR NESTJS INTEGRATION
# =============================================================================
def analyze_with_context(question: str, conversation_history: list = None) -> str:
    """
    Analyze question with conversation context (for NestJS)
    
    Args:
        question: User's question
        conversation_history: List of (user_msg, bot_response) tuples
    
    Returns:
        str: AI response
    """
    if conversation_history:
        recent_context = conversation_history[-3:]  # Last 3 exchanges
        context_str = "\n".join([
            f"User đã hỏi: {ctx[0]}\nBot đã trả lời: {ctx[1][:200]}..." 
            for ctx in recent_context
        ])
        
        query = f"""LỊCH SỬ HỘI THOẠI GẦN ĐÂY:
{context_str}

CÂU HỎI HIỆN TẠI: {question}

Hãy trả lời dựa trên LỊCH SỬ và câu hỏi hiện tại."""
    else:
        query = question
    
    return query

def build_image_analysis_query(skin_analysis: str, additional_text: str = None, is_severe: bool = False) -> str:
    """
    Build RAG query for image analysis (for NestJS)
    
    Args:
        skin_analysis: VLM analysis result
        additional_text: Optional user text
        is_severe: Whether condition is severe
    
    Returns:
        str: RAG query
    """
    if additional_text:
        return f"""Tình trạng da {'(RẤT NGHIÊM TRỌNG - CẦN GẶP BÁC SĨ)' if is_severe else ''}:
{skin_analysis}

Yêu cầu: {additional_text}

{'Gợi ý 1-2 sản phẩm HỖ TRỢ NHẸ NHÀNG (không thay thế điều trị y khoa). NHẤN MẠNH: Cần gặp bác sĩ da liễu.' if is_severe else 'Tư vấn 2-3 sản phẩm CỤ THỂ phù hợp.'}"""
    else:
        return f"""Tình trạng da:
{skin_analysis}

{'Gợi ý 1-2 sản phẩm HỖ TRỢ. NHẤN MẠNH: Cần gặp bác sĩ.' if is_severe else 'Tư vấn 2-3 sản phẩm phù hợp.'}"""

def check_severity(analysis: str) -> bool:
    """Check if skin condition is severe"""
    return any(keyword in analysis.upper() for keyword in ['RẤT NẶNG', 'RẤT NGHIÊM TRỌNG'])