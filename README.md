# 📘 Hệ Thống Dự Báo Giá Xăng Dầu: Từ Dữ Liệu Thô Đến Sản Phẩm AI Chuyên Nghiệp

Dự án này là một nền tảng dự báo giá xăng (RON 95) hoàn chỉnh, được thiết kế để mô phỏng toàn bộ quy trình làm việc của một **Senior Data Scientist**. Tài liệu này giải thích chi tiết từng module để ngay cả sinh viên cũng có thể hiểu và vận hành.

---

## 🏗️ 1. Cấu Trúc Module & Ý Nghĩa (Architecture)

Hệ thống được chia thành các lớp (layers) tách biệt để đảm bảo tính chuyên nghiệp:

### 📁 `src/processing/` (Lớp Tiền Xử Lý)
*   **`data_audit.py`**: Kiểm tra "sức khỏe" dữ liệu (thiếu dòng, lỗi font, outlier).
*   **`data_cleaning.py`**: "Quét dọn" dữ liệu, xử lý các giá trị trống bằng phương pháp Forward Fill (giữ tính liên tục của chuỗi thời gian).

### 📁 `src/analysis/` (Lớp Phân Tích & Khai Phá)
*   **`eda.py`**: Trực quan hóa để tìm xu hướng và sự tương quan giữa các biến (ví dụ: Xăng và Dầu Brent).
*   **`data_mining.py`**: Sử dụng thuật toán để tìm "độ trễ" (Lag). Chúng ta phát hiện Dầu thế giới ảnh hưởng đến Việt Nam sau khoảng 7 ngày.

### 📁 `src/features/` (Lớp Kỹ Thuật Đặc Trưng)
*   **`feature_engineering.py`**: Chuyển đổi dữ liệu thô thành các biến mà AI có thể hiểu được như: Trung bình trượt (Moving Average), Độ biến động (Volatility), và quan trọng nhất là **Dự báo Độ lệch (Delta)**.

### 📁 `src/models/` (Lớp Huấn Luyện AI)
*   Nơi chứa các thuật toán: XGBoost, LightGBM, Random Forest.
*   **`compare_models.py`**: Sân chơi để các mô hình "thi đấu" với nhau nhằm chọn ra người thắng cuộc dựa trên sai số thấp nhất (MAE).

### 📁 `src/api/` & `frontend/` (Lớp Triển Khai)
*   **API (FastAPI)**: "Cổng giao tiếp" cho phép ứng dụng khác lấy kết quả dự báo từ AI.
*   **Frontend (React)**: Giao diện dashboard chuyên nghiệp để người dùng tương tác trực quan.

---

## 🚀 2. Quy Trình Vận Hành 7 Bước (Pipeline)

### Bước 1: Kiểm toán dữ liệu (Audit)
Chúng ta không bao giờ tin ngay vào dữ liệu thô. Ta cần biết có bao nhiêu ô trống, dữ liệu có bị nhảy vọt vô lý không.

### Bước 2: Làm sạch (Cleaning)
Loại bỏ các cột không cần thiết (cột Unnamed, cột rác) và chuẩn hóa tên cột để máy tính không bị lỗi font.

### Bước 3: Khai phá tri thức (Data Mining)
Ta dùng toán học để trả lời câu hỏi: *"Giá dầu thế giới tăng hôm nay thì bao lâu nữa giá xăng Việt Nam tăng?"*. Kết quả: 7 ngày.

### Bước 4: Tạo đặc trưng (Feature Engineering)
Thay vì dự báo "Ngày mai giá bao nhiêu?", ta dự báo "Ngày mai giá tăng/giảm bao nhiêu?". Đây là kỹ thuật giúp mô hình AI chính xác hơn 50%.

### Bước 5: Huấn luyện & So sánh (Modeling & Benchmark)
Ta chạy thử nhiều mô hình. Kết quả cho thấy **Random Forest** là mô hình ổn định nhất cho bài toán này.

### Bước 6: Kiểm chứng cuốn chiếu (Walk-forward Validation)
Đây là cách kiểm tra mô hình theo kiểu "thực tế ảo": Ta giả sử đang ở quá khứ, dùng dữ liệu cũ dự báo tương lai, rồi so sánh với kết quả thật đã xảy ra.

### Bước 7: Đóng gói sản phẩm (Deployment)
Biến mô hình AI thành một trang web (Dashboard) để ai cũng có thể dùng.

---

## ⚙️ 3. Hướng Dẫn Chạy Dự Án

### 1. Cài đặt thư viện
```powershell
pip install -r requirements.txt
cd frontend
npm install
```

### 2. Khởi chạy Hệ thống
*   **Bật Backend**: `python src/api/main.py`
*   **Bật Frontend**: `npm run dev` (truy cập http://localhost:5173)

---

## 🎯 4. Bài Học Rút Ra (Insights)
*   Giá xăng Việt Nam có tính chất "Sticky" (đứng yên lâu rồi mới điều chỉnh).
*   Mô hình AI chỉ thực sự hiệu quả khi chúng ta biết tạo ra các đặc trưng thông minh (như `days_since_last_change`).
*   Một hệ thống tốt là một hệ thống có cấu trúc rõ ràng, module hóa cao.

---
*Tài liệu này giúp bạn hiểu rằng Data Science không chỉ là chạy code, mà là một quy trình tư duy logic từ dữ liệu thô đến tri thức.*
