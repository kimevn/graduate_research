# 📦 Phương pháp kết hợp rời rạc hoá dữ liệu và trích chọn đặc trưng dựa trên lý thuyết thông tin

## 🧰 Công nghệ sử dụng
- Python >= 3.8  
- Các thư viện như:
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `matplotlib`
  - `ucimlrepo`
  - `math`

## 🚀 Cài đặt
- Tạo virtual environment (khuyên dùng):
  ```bash
  python -m venv venv
  venv\Scripts\activate  
  ```
  
- Cài đặt các thư viện cần thiết:
  ```bash
    pip install numpy pandas scikit-learn matplotlib ucimlrepo math
  ```
  
- Sử dụng dữ liệu từ UCI Machine Learning Repository(tùy vào dataset sẽ có id khác nhau):
  ```bash
  dataset = fetch_ucirepo(id=109)
  X = dataset.data.features
  y = dataset.data.targets
  ```
  
- Chạy mã nguồn:
  ```bash
  python main.py
  ```