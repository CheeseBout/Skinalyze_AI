# Sử dụng Python 3.10 (ổn định cho PyTorch và Mediapipe)
FROM python:3.10-slim

# Thiết lập thư mục làm việc
WORKDIR /app

# Cài đặt các thư viện hệ thống cần thiết cho OpenCV và Mediapipe
# git: để pip install từ github (SAM2)
# libgl1-mesa-glx & libglib2.0-0: bắt buộc cho xử lý ảnh
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt trước để tận dụng Docker cache
COPY requirements.txt .

# Cài đặt các thư viện Python
# Tăng timeout vì cài Torch và SAM2 khá lâu
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt --timeout 1000

# Tạo các thư mục cần thiết (như trong setup.bat của bạn)
# Và cấp quyền ghi cho user (Hugging Face chạy user 1000)
RUN mkdir -p data db_chroma chat_history models && \
    chmod -R 777 data db_chroma chat_history models

# Copy toàn bộ code vào container
COPY . .

# Mở cổng 7860 (Cổng mặc định của Hugging Face Spaces)
EXPOSE 7860

# Lệnh chạy server
# Lưu ý: Thay 'main:app' bằng 'tên_file_chính_của_bạn:app'
# Ví dụ: nếu file chính là main.py thì để nguyên.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]