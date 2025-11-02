***Ứng Dụng Trợ Lý Học Biển Báo Giao Thông Thông Minh Dựa Trên AI***

Đây là dự án xây dựng một ứng dụng web hỗ trợ học tập và nhận diện biển báo giao thông (BBGT) tại Việt Nam. Ứng dụng được phát triển bằng Streamlit, kết hợp mô hình nhận diện đối tượng (YOLOv8) để phát hiện biển báo từ ảnh và mô hình ngôn ngữ lớn (Google Gemini) để cung cấp giải thích chi tiết, trả lời câu hỏi và hỗ trợ ôn tập.

**Tính năng chính**
Nhận diện BBGT: Tải ảnh lên hoặc sử dụng camera để nhận diện 56 loại biển báo giao thông Việt Nam.
Giải thích bằng AI: Tự động cung cấp ý nghĩa, tác dụng và mức phạt (tham khảo) cho các biển báo được nhận diện bằng Google Gemini API.
Tra cứu tài liệu: Xem và tìm kiếm toàn bộ 56 loại biển báo được phân loại theo danh mục.
Kiểm tra kiến thức (Quiz): Thực hiện bài kiểm tra trắc nghiệm 10 câu hỏi ngẫu nhiên. Nếu trả lời sai, AI sẽ tự động giải thích.
Chatbot AI Giao Thông: Trò chuyện trực tiếp với trợ lý AI (Gemini) để hỏi đáp về luật và các tình huống giao thông liên quan đến biển báo.
Quản lý người dùng: Hỗ trợ đăng ký, đăng nhập và lưu lịch sử làm quiz, lịch sử các cuộc trò chuyện.

**Công nghệ sử dụng**
Ngôn ngữ: Python 3.x
Framework Web: Streamlit
Nhận diện Đối tượng: Ultralytics (YOLOv8n)
Mô hình Ngôn ngữ (LLM): Google Gemini (gemini-2.0-flash)
Xử lý ảnh: OpenCV, Pillow (PIL)
Cơ sở dữ liệu: SQLite
Thư viện khác: Numpy, Pandas

**Chuẩn bị và Cài đặt**

**1. Tải mã nguồn (hoặc Clone Repository)**
git clone [https://github.com/ten-repo-cua-ban/ten-du-an.git](https://github.com/ten-repo-cua-ban/ten-du-an.git)
cd ten-du-an

**2. Chuẩn bị các tệp tin cần thiết**
Trước khi chạy, hãy đảm bảo bạn có các tệp dữ liệu quan trọng sau trong thư mục gốc của dự án:
Mô hình YOLOv8:
last85.pt: File trọng số của mô hình YOLOv8n đã được huấn luyện.
Dữ liệu biển báo:
signs_info.json: File JSON chứa thông tin (tên, mô tả, loại) của 56 biển báo.
Ảnh biển báo (cho trang Tài liệu):
Một thư mục có tên signs chứa hình ảnh .png của 56 loại biển báo. Tên tệp phải khớp với key trong signs_info.json (ví dụ: dung_lai.png, cam_bop_coi.png...).

**3. Cài đặt Môi trường (Khuyến nghị)**
Tạo một môi trường ảo để tránh xung đột thư viện:
# Dành cho macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Dành cho Windows
python -m venv venv
.\venv\Scripts\activate

**4. Cài đặt thư viện**

Dự án yêu cầu các thư viện được liệt kê trong requirements.txt.
Nội dung requirements.txt:
streamlit>=1.24
ultralytics>=8.0.0
pillow
numpy
pandas
google-generativeai
opencv-python-headless

**Cài đặt bằng lệnh:**
pip install -r requirements.txt

**5. Cấu hình API Key**

Ứng dụng này cần một API Key từ Google Gemini (Google AI Studio) để hoạt động.
Mở tệp app_v5.5.py.
Tìm đến dòng sau:
GEMINI_API_KEY = "AIzaSyCq6nzJrXliZZyzVDvPmm6juTOEttRjJdQ" # Replace with your actual API key
Thay thế chuỗi "AIzaSy...JdQ" bằng API Key của riêng bạn.
Khởi chạy ứng dụng
Sau khi hoàn tất các bước cài đặt và cấu hình, chạy ứng dụng bằng lệnh sau trong terminal:
streamlit run app_v5.5.py

Streamlit sẽ tự động mở một tab trên trình duyệt của bạn (thường là http://localhost:8501).

**Cấu trúc thư mục**
<img width="1806" height="752" alt="image" src="https://github.com/user-attachments/assets/05b77e3e-f66e-4236-b6a1-c403f514ec74" />



