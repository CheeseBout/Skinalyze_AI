
import os
import pandas as pd
from pathlib import Path
import chromadb
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import torch
import time
from getpass import getpass
from PIL import Image
import google.generativeai as genai
from datetime import datetime
import json

# =============================================================================
# CẤU HÌNH - THAY ĐỔI CÁC ĐƯỜNG DẪN NÀY
# =============================================================================
CHUNKS_FILE = Path(r"D:\rag-cosmetic-chatbot\data\product_chunks.txt")
PERSIST_DIRECTORY = Path(r"D:\rag-cosmetic-chatbot\db_chroma")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHAT_HISTORY_DIR = Path(r"D:\rag-cosmetic-chatbot\chat-history")  # Thư mục lưu lịch sử chat

# Global cache cho embeddings để tránh load lại
_CACHED_EMBEDDINGS = None

# =============================================================================
# THIẾT LẬP API KEY
# =============================================================================
def setup_api_key():
    """Thiết lập Google API Key"""
    if "GOOGLE_API_KEY" not in os.environ:
        print("\n🔑 Cần Google API Key để sử dụng Gemini")
        print("💡 Lấy key miễn phí tại: https://makersuite.google.com/app/apikey\n")
        api_key = "AIzaSyDLKLqpBHxf3xiutoYk5MjMzTywvju0Dx0"
        os.environ["GOOGLE_API_KEY"] = api_key
        print("✅ Đã thiết lập API Key!\n")
    else:
        print("✅ API Key đã được cấu hình sẵn!\n")
    
    # Configure genai for vision
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# =============================================================================
# LOAD HOẶC TẠO VECTOR STORE
# =============================================================================
def load_or_create_vectorstore():
    """Load vector store có sẵn hoặc tạo mới nếu chưa có, với error handling."""
    global _CACHED_EMBEDDINGS
    
    print("=" * 80)
    print("📚 KHỞI TẠO VECTOR STORE")
    print("=" * 80)
    
    db = None
    embeddings = None
    
    try: # <<< Try chính bao quanh toàn bộ hàm >>>
        
        # ----- Tải Embedding Model (với cache) -----
        if _CACHED_EMBEDDINGS is not None:
            print(f"\n⚡ Sử dụng cached embedding model")
            embeddings = _CACHED_EMBEDDINGS
        else:
            print(f"\n⏳ Đang tải embedding model: {MODEL_NAME}...")
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"   🖥️ Sử dụng thiết bị: {device}")
            
            try: # <<< Try cho việc tải embedding model >>>
                embeddings = HuggingFaceEmbeddings(
                    model_name=MODEL_NAME,
                    model_kwargs={'device': device},
                    encode_kwargs={'normalize_embeddings': True}
                )
                _CACHED_EMBEDDINGS = embeddings  # Cache lại
                print("✅ Đã tải embedding model!\n")
            except Exception as e_embed_load:
                print(f"\n❌ LỖI NGHIÊM TRỌNG khi tải embedding model: {e_embed_load}")
                print("   Kiểm tra lại tên model, kết nối mạng và cài đặt thư viện.")
                return None, None # Trả về None nếu không tải được model

        # ----- Load hoặc Tạo Database -----
        if os.path.exists(PERSIST_DIRECTORY):
            print(f"📂 Phát hiện Vector Store có sẵn tại: {PERSIST_DIRECTORY}")
            print("⏳ Đang load database...\n")
            
            try: # <<< Try cho việc load DB có sẵn >>>
                db = Chroma(
                    persist_directory=str(PERSIST_DIRECTORY),
                    embedding_function=embeddings
                )
                
                # Kiểm tra xem collection có dữ liệu không
                count = db._collection.count() if db._collection else 0
                
                print(f"✅ Đã load Vector Store thành công!")
                print(f"   📊 Số documents trong database: {count}\n")
                if count == 0:
                     print("   ⚠️ Cảnh báo: Database có sẵn nhưng không có document nào.")

            except Exception as e_db_load:
                print(f"\n❌ LỖI khi load Vector Store có sẵn: {e_db_load}")
                print(f"   Thử xóa thư mục '{PERSIST_DIRECTORY}' và chạy lại để tạo mới.")
                return None, embeddings # Trả về embeddings đã load được, nhưng db là None
                
        else:
            print(f"🆕 Không tìm thấy Vector Store. Đang tạo mới từ {CHUNKS_FILE.name}...\n")
            
            # --- Các bước tạo DB mới ---
            docs = None
            try: # <<< Try cho việc load và split file chunks >>>
                # 1. Load file chunks
                print("📖 [1/4] Đang load file chunks...")
                if not CHUNKS_FILE.exists():
                     raise FileNotFoundError(f"File chunk không tồn tại tại: {CHUNKS_FILE}")
                loader = TextLoader(str(CHUNKS_FILE), encoding='utf-8')
                documents = loader.load()
                print(f"   ✓ Đã load {len(documents)} document base")
                
                # 2. Split documents
                print("✂️  [2/4] Đang split thành từng chunk...")
                text_splitter = RecursiveCharacterTextSplitter(
                    separators=["---"], # Tách theo dấu ---
                    chunk_size=400,   # Giảm chunk size để LLM xử lý nhanh hơn
                    chunk_overlap=50,  # Thêm overlap để không mất context
                    length_function=len
                )
                docs = text_splitter.split_documents(documents)
                if not docs:
                     print("   ⚠️ Cảnh báo: Không split được chunk nào. Kiểm tra file và separator.")
                     return None, embeddings # Không có docs để tạo DB
                print(f"   ✓ Đã split thành {len(docs)} chunks")
                
            except FileNotFoundError as e_file:
                 print(f"\n❌ LỖI: {e_file}")
                 return None, embeddings
            except Exception as e_load_split:
                 print(f"\n❌ LỖI khi load hoặc split file chunks: {e_load_split}")
                 return None, embeddings

            # --- Tạo embeddings và lưu ---
            try: # <<< Try cho việc tạo DB mới và thêm docs >>>
                print("💾 [3/4] Đang tạo embeddings và lưu vào database...")
                print("   (Quá trình này có thể mất vài phút, vui lòng đợi...)\n")
                
                start_time = time.time()
                
                # Xử lý theo batch
                batch_size = 50 # Giảm batch size nếu gặp lỗi bộ nhớ
                total_docs = len(docs)
                
                if total_docs == 0:
                     print("   ⚠️ Không có chunk nào để thêm vào database.")
                     return None, embeddings # Không thể tạo DB rỗng theo cách này

                if total_docs <= batch_size:
                    # Nếu ít docs thì tạo một lần
                    print(f"   ⏳ Đang xử lý {total_docs} documents...")
                    db = Chroma.from_documents(
                        documents=docs,
                        embedding=embeddings,
                        persist_directory=str(PERSIST_DIRECTORY)
                    )
                else:
                    # Nếu nhiều docs thì chia batch
                    print(f"   ⏳ Đang xử lý theo batch ({batch_size} docs/batch)...")
                    
                    # Batch đầu tiên - tạo database
                    current_batch_docs = docs[:batch_size]
                    print(f"   → Batch 1/{(total_docs-1)//batch_size + 1}: docs 0-{len(current_batch_docs)}")
                    db = Chroma.from_documents(
                        documents=current_batch_docs,
                        embedding=embeddings,
                        persist_directory=str(PERSIST_DIRECTORY)
                    )
                    
                    # Các batch tiếp theo - thêm vào database
                    for i in range(batch_size, total_docs, batch_size):
                        batch_start = i
                        batch_end = min(i + batch_size, total_docs)
                        current_batch_docs = docs[batch_start:batch_end]
                        batch_num = (i // batch_size) + 1
                        total_batches = (total_docs - 1) // batch_size + 1
                        
                        print(f"   → Batch {batch_num}/{total_batches}: docs {batch_start}-{batch_end}")
                        if not current_batch_docs: # Kiểm tra batch rỗng (dư thừa nhưng an toàn)
                             continue
                        db.add_documents(current_batch_docs)
                        
                        # Giải phóng bộ nhớ GPU nếu dùng CUDA
                        if device == 'cuda':
                            torch.cuda.empty_cache()
                
                end_time = time.time()
            
                print(f"\n   ✓ Hoàn thành sau {end_time - start_time:.2f} giây")
                # Kiểm tra lại số lượng sau khi tạo
                count_after_create = db._collection.count() if db and db._collection else 0
                print(f"   📊 Đã tạo và lưu {count_after_create} vectors")
                if count_after_create != total_docs:
                     print(f"   ⚠️ Cảnh báo: Số vector lưu ({count_after_create}) không khớp số chunk ({total_docs}).")

                print("\n✅ Đã tạo Vector Store thành công!")

            except Exception as e_db_create:
                 print(f"\n❌ LỖI NGHIÊM TRỌNG khi tạo/lưu Vector Store mới: {e_db_create}")
                 print(f"   Thử kiểm tra dung lượng ổ đĩa, quyền ghi vào '{PERSIST_DIRECTORY}', hoặc giảm 'batch_size'.")
                 # Xóa thư mục có thể bị tạo dở dang
                 if os.path.exists(PERSIST_DIRECTORY):
                      try:
                           import shutil
                           shutil.rmtree(PERSIST_DIRECTORY)
                           print(f"   Đã xóa thư mục '{PERSIST_DIRECTORY}' có thể bị lỗi.")
                      except Exception as e_del:
                           print(f"   Không thể xóa thư mục lỗi '{PERSIST_DIRECTORY}': {e_del}")
                 db = None # Đặt lại db thành None vì tạo lỗi
                 return None, embeddings # Trả về embeddings nhưng db là None

    except Exception as e_global: # <<< Except cho try chính >>>
         print(f"\n❌ ĐÃ XẢY RA LỖI KHÔNG XÁC ĐỊNH: {e_global}")
         return None, None # Trả về None cho cả hai nếu có lỗi lớn

    # Nếu mọi thứ thành công
    return db, embeddings
# =============================================================================
# KHỞI TẠO RAG CHAIN
# =============================================================================
def setup_rag_chain(db):
    """Thiết lập RAG chain với Retriever, LLM và Prompt"""
    print("\n" + "=" * 80)
    print("⛓️ KHỞI TẠO RAG CHAIN")
    print("=" * 80)
    
    # 1. Khởi tạo LLM (tối ưu cho tốc độ)
    print("\n🤖 [1/3] Đang kết nối với Google Gemini...")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",  # Gemini 2.0 stable - mới nhất, cân bằng tốc độ & quota
        temperature=0.1,  # Giảm temperature để phản hồi nhanh hơn
        max_output_tokens=512,  # Giới hạn độ dài output để nhanh hơn
        convert_system_message_to_human=True,
        request_options={"timeout": 60},  # Timeout 60s
        max_retries=2  # Chỉ retry 2 lần thay vì mặc định
    )
    print("   ✓ Đã kết nối Gemini 2.0 Flash Stable (mới nhất, cân bằng)")
    
    # 2. Tạo Retriever (giảm k để nhanh hơn)
    print("🔍 [2/3] Đang tạo Retriever...")
    retriever = db.as_retriever(
        search_type="mmr",  # Sử dụng MMR để giảm độ trùng lặp
        search_kwargs={
            "k": 2,  # Giảm xuống 2 chunks để nhanh hơn
            "fetch_k": 5,  # Fetch 5 rồi lọc xuống 2
            "lambda_mult": 0.7  # Cân bằng giữa relevance và diversity
        }
    )
    print("   ✓ Retriever sẽ lấy top 2 chunks đa dạng nhất (MMR, siêu nhanh)")
    
    # 3. Tạo Prompt Template (tối ưu, ngắn gọn hơn)
    print("📝 [3/3] Đang tạo Prompt Template...")
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
→ Kiểm tra database, nếu có thì liệt kê, nếu không: "Mình chưa có thông tin về [brand] trong database. Bạn muốn tư vấn sản phẩm theo loại da hay vấn đề cụ thể không? �"

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
    print("   ✓ Đã tạo Prompt Template (compact + smart filtering)")
    
    # 4. Xây dựng RAG Chain
    def format_docs(docs):
        """Format documents thành string (tối ưu, loại bỏ thông tin dư thừa)"""
        formatted = []
        for i, doc in enumerate(docs, 1):
            # Chỉ lấy thông tin quan trọng, bỏ qua các dòng trống
            content = doc.page_content.strip()
            if content:
                # Giới hạn độ dài mỗi chunk để giảm token
                if len(content) > 500:
                    content = content[:500] + "..."
                formatted.append(f"[{i}] {content}")
        return "\n".join(formatted)  # Dùng \n thay vì \n\n để compact hơn
    
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    print("\n✅ RAG Chain đã sẵn sàng!")
    print("\n📊 Luồng hoạt động (SIÊU TỐI ƯU):")
    print("   1️⃣  User Question → Retriever")
    print("   2️⃣  Retriever → Top 2 chunks MMR (đa dạng, không trùng lặp)")
    print("   3️⃣  Format chunks → Context string (max 500 chars/chunk)")
    print("   4️⃣  Context + Question → Prompt (compact)")
    print("   5️⃣  Prompt → LLM (Gemini 2.0 Flash Exp, max_tokens=512)")
    print("   6️⃣  LLM → Final Answer ⚡⚡⚡")

    return rag_chain

# =============================================================================
# CHAT HISTORY - QUẢN LÝ LỊCH SỬ HỘI THOẠI
# =============================================================================
def save_chat_history(chat_history):
    """Lưu lịch sử chat vào file JSON"""
    try:
        # Tạo thư mục nếu chưa có
        CHAT_HISTORY_DIR.mkdir(exist_ok=True)
        
        # Tên file theo timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = CHAT_HISTORY_DIR / f"chat_{timestamp}.json"
        
        # Lưu vào file
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(chat_history, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 Đã lưu lịch sử chat: {filename.name}")
        return filename
    except Exception as e:
        print(f"\n⚠️  Lỗi khi lưu lịch sử: {str(e)}")
        return None

def load_latest_chat_history():
    """Load lịch sử chat gần nhất (nếu có)"""
    try:
        if not CHAT_HISTORY_DIR.exists():
            return None
        
        # Tìm file mới nhất
        chat_files = list(CHAT_HISTORY_DIR.glob("chat_*.json"))
        if not chat_files:
            return None
        
        latest_file = max(chat_files, key=lambda f: f.stat().st_mtime)
        
        with open(latest_file, 'r', encoding='utf-8') as f:
            history = json.load(f)
        
        return history, latest_file
    except Exception as e:
        print(f"\n⚠️  Lỗi khi load lịch sử: {str(e)}")
        return None

# =============================================================================
# VISION ANALYSIS - PHÂN TÍCH ẢNH DA ĐỂ BỔ SUNG THÔNG TIN CHO RAG
# =============================================================================
def analyze_skin_image(image_path):
    """Phân tích ảnh da bằng VLM - Tập trung vào mức độ nghiêm trọng làm đầu vào cho RAG"""
    try:
        print("\n📸 Đang phân tích tình trạng da từ ảnh...")
        
        # Load image
        img = Image.open(image_path)
        
        # Khởi tạo Gemini Vision model
        vision_model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Prompt tập trung vào mức độ nghiêm trọng
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
        
        # Gọi vision model
        response = vision_model.generate_content([vision_prompt, img])
        analysis = response.text
        
        print("✅ Đã phân tích xong!")
        
        return analysis
        
    except FileNotFoundError:
        print(f"❌ Không tìm thấy file ảnh: {image_path}")
        return None
    except Exception as e:
        print(f"❌ Lỗi khi phân tích ảnh: {str(e)}")
        return None

# =============================================================================
# INTERACTIVE CHAT
# =============================================================================
def chat(rag_chain):
    """Interactive chat trong terminal với hỗ trợ phân tích ảnh da và lưu lịch sử"""
    print("\n" + "=" * 80)
    print("💬 COSMETIC CONSULTANT CHATBOT (⚡ RAG + 📸 VLM + 💾 HISTORY)")
    print("=" * 80)
    
    # Load lịch sử chat trước đó (nếu có)
    previous_history = load_latest_chat_history()
    if previous_history:
        history, history_file = previous_history
        print(f"\n📖 Tìm thấy lịch sử chat trước: {history_file.name}")
        print(f"   Số lượng: {len(history)} tin nhắn")
        view = input("   Xem lịch sử? (y/n): ").strip().lower()
        if view == 'y':
            print("\n" + "=" * 80)
            print("� LỊCH SỬ CHAT TRƯỚC:")
            print("=" * 80)
            for msg in history[-10:]:  # Hiển thị 10 tin nhắn cuối
                role = "🧑 Bạn" if msg['role'] == 'user' else "🤖 Bot"
                content = msg['content'][:200] + "..." if len(msg['content']) > 200 else msg['content']
                print(f"{role}: {content}")
                print("-" * 40)
            print("=" * 80)
    
    print("\n�📝 Gõ câu hỏi của bạn và nhấn Enter")
    print("💡 Ví dụ text: 'Tôi cần kem dưỡng cho da khô nhạy cảm'")
    print("📸 Phân tích ảnh DA: Gửi đường dẫn ảnh da của bạn (tự động nhận diện)")
    print("   → VLM phân tích chi tiết tình trạng da")
    print("   → RAG tư vấn sản phẩm phù hợp dựa trên phân tích")
    print("   Ví dụ: C:\\Users\\Photos\\my_skin.jpg")
    print("🚪 Gõ 'exit', 'quit' hoặc 'thoát' để kết thúc và LƯU LỊCH SỬ")
    print("⚡ Công nghệ: VLM (Gemini 2.5 Flash) + RAG (ChromaDB)\n")
    print("=" * 80)
    
    # Khởi tạo lịch sử chat mới và conversation memory
    chat_history = {
        'session_start': datetime.now().isoformat(),
        'messages': []
    }
    
    # Conversation memory - lưu context trong phiên (bot sẽ nhớ!)
    conversation_context = []  # Lưu tất cả trao đổi: [(user_msg, bot_response), ...]
    
    # Các đuôi file ảnh được hỗ trợ
    IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif', '.tiff')
    
    while True:
        print()
        try:
            # Nhận input từ user
            question = input("🧑 Bạn: ").strip()
            
            # Kiểm tra điều kiện thoát
            if not question:
                print("⚠️  Vui lòng nhập câu hỏi!")
                continue
                
            if question.lower() in ['exit', 'quit', 'thoát', 'bye', 'goodbye']:
                print("\n👋 Cảm ơn bạn đã sử dụng dịch vụ!")
                # Lưu lịch sử trước khi thoát
                if chat_history['messages']:
                    chat_history['session_end'] = datetime.now().isoformat()
                    save_chat_history(chat_history)
                print("=" * 80)
                break
            
            # Tự động nhận diện đường dẫn ảnh
            # Loại bỏ dấu ngoặc kép nếu có
            question_clean = question.strip('"').strip("'")
            
            # Tách đường dẫn ảnh và text (nếu user gửi cả hai)
            image_path_candidate = None
            text_question = None
            
            # Kiểm tra nếu có đường dẫn file trong câu hỏi
            # Tìm đường dẫn trong dấu ngoặc kép trước
            if '"' in question:
                # Extract path trong dấu ngoặc kép
                import re
                matches = re.findall(r'"([^"]+)"', question)
                for match in matches:
                    if any(match.lower().endswith(ext) for ext in IMAGE_EXTENSIONS):
                        image_path_candidate = match
                        # Phần còn lại là text question
                        text_question = question.replace(f'"{match}"', '').strip()
                        break
            
            # Nếu không có dấu ngoặc, kiểm tra đường dẫn trực tiếp
            if not image_path_candidate:
                # Tách câu theo space để tìm đường dẫn
                words = question_clean.split()
                for word in words:
                    if any(word.lower().endswith(ext) for ext in IMAGE_EXTENSIONS):
                        # Kiểm tra nếu là đường dẫn hợp lệ (có \ hoặc / hoặc :)
                        if '\\' in word or '/' in word or ':' in word:
                            image_path_candidate = word
                            # Phần còn lại là text
                            text_question = question_clean.replace(word, '').strip()
                            break
            
            # Nếu toàn bộ input là đường dẫn ảnh
            if not image_path_candidate and any(question_clean.lower().endswith(ext) for ext in IMAGE_EXTENSIONS):
                image_path_candidate = question_clean
            
            # Xử lý nếu có prefix image:/ảnh:/anh:
            if not image_path_candidate and question.lower().startswith(('image:', 'ảnh:', 'anh:')):
                parts = question.split(':', 1)
                if len(parts) > 1:
                    image_path_candidate = parts[1].strip().strip('"').strip("'")
            
            # Nếu tìm thấy đường dẫn ảnh
            if image_path_candidate:
                image_path = image_path_candidate
                
            # Nếu tìm thấy đường dẫn ảnh
            if image_path_candidate:
                image_path = image_path_candidate
                
                # Xử lý đường dẫn tương đối
                if not os.path.isabs(image_path):
                    image_path = os.path.join(os.getcwd(), image_path)
                
                # Kiểm tra file tồn tại
                if not os.path.exists(image_path):
                    print(f"❌ Không tìm thấy file ảnh: {image_path}")
                    print("💡 Vui lòng kiểm tra lại đường dẫn!")
                    print("-" * 80)
                    continue
                
                # Bước 1: VLM phân tích ảnh da
                skin_analysis = analyze_skin_image(image_path)
                
                # Lưu input user (ảnh)
                chat_history['messages'].append({
                    'timestamp': datetime.now().isoformat(),
                    'role': 'user',
                    'type': 'image',
                    'content': f"[Gửi ảnh: {os.path.basename(image_path)}]",
                    'image_path': image_path,
                    'additional_text': text_question if text_question else None
                })
                
                if skin_analysis:
                    # Kiểm tra mức độ nghiêm trọng
                    analysis_upper = skin_analysis.upper()
                    is_very_severe = 'RẤT NẶNG' in analysis_upper or 'RẤT NGHIÊM TRỌNG' in analysis_upper
                    
                    # Hiển thị cảnh báo nếu rất nặng
                    if is_very_severe:
                        print("\n" + "⚠️ " * 20)
                        print("⚠️  CẢNH BÁO: TÌNH TRẠNG DA RẤT NGHIÊM TRỌNG!")
                        print("⚠️ " * 20)
                        print("\n🏥 KHUYẾN CÁO:")
                        print("   • Tình trạng da của bạn CẦN được bác sĩ da liễu thăm khám")
                        print("   • Không nên tự điều trị hoặc chỉ dùng mỹ phẩm")
                        print("   • Vui lòng đặt lịch gặp bác sĩ da liễu NGAY")
                        print("\n" + "=" * 80)
                        
                        # Vẫn tư vấn sản phẩm hỗ trợ nhưng có disclaimer
                        print("\n💡 Tuy nhiên, dưới đây là một số sản phẩm HỖ TRỢ (KHÔNG THAY THẾ điều trị y khoa):\n")
                    
                    # Bước 2: Kết hợp phân tích VLM với câu hỏi để query RAG
                    if text_question:
                        if is_very_severe:
                            rag_query = f"""Tình trạng da (RẤT NGHIÊM TRỌNG - CẦN GẶP BÁC SĨ):
{skin_analysis}

Yêu cầu: {text_question}

Gợi ý 1-2 sản phẩm HỖ TRỢ NHẸ NHÀNG (không thay thế điều trị y khoa). 
NHẤN MẠNH: Cần gặp bác sĩ da liễu."""
                        else:
                            rag_query = f"""Tình trạng da (từ phân tích ảnh):
{skin_analysis}

Yêu cầu: {text_question}

Tư vấn 2-3 sản phẩm CỤ THỂ phù hợp với MỨC ĐỘ."""
                    else:
                        # Không có câu hỏi, chỉ dựa vào phân tích
                        if is_very_severe:
                            rag_query = f"""Tình trạng da (RẤT NGHIÊM TRỌNG - CẦN GẶP BÁC SĨ):
{skin_analysis}

Gợi ý 1-2 sản phẩm HỖ TRỢ NHẸ NHÀNG (không thay thế điều trị y khoa).
NHẤN MẠNH: Cần gặp bác sĩ da liễu."""
                        else:
                            rag_query = f"""Tình trạng da (từ phân tích ảnh):
{skin_analysis}

Tư vấn 2-3 sản phẩm CỤ THỂ phù hợp với MỨC ĐỘ."""
                    
                    print("\n🔎 Tìm sản phẩm dựa trên mức độ nghiêm trọng...")
                    time.sleep(1)
                    
                    product_recommendation = rag_chain.invoke(rag_query)
                    
                    # Lưu vào conversation context
                    user_input_desc = f"[Gửi ảnh da] {text_question if text_question else 'Phân tích và tư vấn'}"
                    conversation_context.append((user_input_desc, product_recommendation))
                    
                    print("\n💄 TƯ VẤN SẢN PHẨM:")
                    print("=" * 80)
                    print(product_recommendation)
                    print("=" * 80)
                    
                    # Lưu response của bot
                    bot_response = product_recommendation
                    if is_very_severe:
                        bot_response = f"⚠️ CẢNH BÁO: RẤT NGHIÊM TRỌNG - CẦN GẶP BÁC SĨ!\n\n{product_recommendation}"
                    
                    chat_history['messages'].append({
                        'timestamp': datetime.now().isoformat(),
                        'role': 'assistant',
                        'type': 'product_recommendation',
                        'content': bot_response,
                        'skin_analysis': skin_analysis,
                        'severity': 'VERY_SEVERE' if is_very_severe else 'NORMAL'
                    })
                    
                    # Nhắc lại cảnh báo nếu rất nặng
                    if is_very_severe:
                        print("\n" + "⚠️ " * 20)
                        print("⚠️  LƯU Ý: Các sản phẩm trên CHỈ HỖ TRỢ, KHÔNG THAY THẾ điều trị y khoa!")
                        print("⚠️  VUI LÒNG ĐẶT LỊCH GẶP BÁC SĨ DA LIỄU NGAY! 🏥")
                        print("⚠️ " * 20)
                
                print("-" * 80)
                continue
            
            # Xử lý câu hỏi text thông thường
            print("\n⏳ Đang tìm kiếm và tạo câu trả lời...")
            start_time = time.time()
            
            # Lưu câu hỏi user
            chat_history['messages'].append({
                'timestamp': datetime.now().isoformat(),
                'role': 'user',
                'type': 'text',
                'content': question
            })
            
            # Thêm delay nhỏ để tránh rate limit
            time.sleep(1)  # Chờ 1 giây trước mỗi request
            
            # Tạo query với context từ conversation history
            if conversation_context:
                # Lấy 3 cặp hội thoại gần nhất để làm context
                recent_context = conversation_context[-3:]
                context_str = "\n".join([
                    f"User đã hỏi: {ctx[0]}\nBot đã trả lời: {ctx[1][:200]}..." 
                    for ctx in recent_context
                ])
                
                query_with_context = f"""LỊCH SỬ HỘI THOẠI GẦN ĐÂY:
{context_str}

CÂU HỎI HIỆN TẠI: {question}

Hãy trả lời dựa trên LỊCH SỬ và câu hỏi hiện tại (nếu user đang hỏi tiếp về cùng topic)."""
                response = rag_chain.invoke(query_with_context)
            else:
                # Lần đầu tiên, không có context
                response = rag_chain.invoke(question)
            
            elapsed_time = time.time() - start_time
            
            # Lưu vào conversation context (bot sẽ nhớ!)
            conversation_context.append((question, response))
            
            # In response
            print(f"\n🤖 Bot: {response}")
            print(f"\n⚡ Thời gian phản hồi: {elapsed_time:.2f}s")
            print("-" * 80)
            
            # Lưu response của bot
            chat_history['messages'].append({
                'timestamp': datetime.now().isoformat(),
                'role': 'assistant',
                'type': 'text',
                'content': response,
                'response_time': elapsed_time
            })
            
        except KeyboardInterrupt:
            print("\n\n👋 Đã nhận tín hiệu thoát. Cảm ơn bạn đã sử dụng!")
            print("=" * 80)
            break
            
        except Exception as e:
            print(f"\n❌ Đã có lỗi xảy ra: {str(e)}")
            print("💡 Vui lòng thử lại với câu hỏi khác!")
            print("-" * 80)

# =============================================================================
# MAIN FUNCTION
# =============================================================================
def main():
    """Main function - điểm khởi đầu chương trình"""
    try:
        print("\n🎯 COSMETIC RAG CHATBOT - INTERACTIVE MODE")
        print("=" * 80)
        
        # 1. Setup API Key
        setup_api_key()
        
        # 2. Load/Create Vector Store
        db, embeddings = load_or_create_vectorstore()
        
        # 3. Setup RAG Chain
        rag_chain = setup_rag_chain(db)
        
        # 4. Start Chat
        chat(rag_chain)
        
    except Exception as e:
        print(f"\n❌ LỖI NGHIÊM TRỌNG: {str(e)}")
        print("💡 Vui lòng kiểm tra lại cấu hình và thử lại!")
        return 1
    
    return 0

# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    exit(main())