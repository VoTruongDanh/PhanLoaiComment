# Web App Phân Tích Sentiment

Web app đơn giản để phân tích sentiment cho comments TikTok với giao diện thân thiện.

## Cài đặt

```bash
pip install -r requirements.txt
```

## Chạy Web App

### Cách 1: Sử dụng script
```bash
python src/run_web.py
```

### Cách 2: Chạy trực tiếp với Streamlit
```bash
cd src
streamlit run app.py
```

Sau khi chạy, trình duyệt sẽ tự động mở tại `http://localhost:8501`

## Tính năng

1. **Upload File CSV**: Upload file CSV chứa cột `text`
2. **Phân Tích Sentiment**: Tự động phân tích và thêm cột `trust`
3. **Xem Kết Quả**: Xem kết quả phân tích trong bảng
4. **Tìm Kiếm & Lọc**: Tìm kiếm và lọc theo sentiment
5. **Thống Kê**: Xem biểu đồ và thống kê phân bố sentiment
6. **Download**: Tải file CSV đã phân tích

## Giao diện

- **Tab 1 - Upload & Phân Tích**: Upload file và chạy phân tích
- **Tab 2 - Kết Quả**: Xem kết quả, tìm kiếm, lọc và download
- **Tab 3 - Thống Kê**: Xem thống kê và biểu đồ phân bố

## Lưu ý

- Lần đầu chạy sẽ tải model (~500MB), mất vài phút
- Nếu có GPU, app sẽ tự động sử dụng để tăng tốc
- File CSV phải có cột `text` chứa comments
