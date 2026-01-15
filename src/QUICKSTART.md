# Hướng Dẫn Sử Dụng Nhanh

## Bước 1: Cài đặt thư viện

```bash
pip install -r src/requirements.txt
```

## Bước 2: Chạy tool

### Cách 1: Chạy script đơn giản (Khuyến nghị)

```bash
python src/run_sentiment.py
```

### Cách 2: Chạy với tham số tùy chỉnh

```bash
python src/sentiment_analyzer.py --input dataset_tiktok-comments-637video-scraper_2026-01-15.csv
```

### Cách 3: Sử dụng trong code Python

```python
from src.sentiment_analyzer import SentimentAnalyzer

analyzer = SentimentAnalyzer()
analyzer.process_csv('dataset_tiktok-comments-637video-scraper_2026-01-15.csv')
```

## Kết quả

Tool sẽ:
- Đọc file CSV
- Phân tích sentiment cho cột `text`
- Tạo/cập nhật cột `trust` với giá trị:
  - **1**: Tích cực
  - **0**: Trung tính  
  - **-1**: Tiêu cực
- Lưu kết quả vào file (ghi đè file đầu vào nếu không chỉ định output)

## Lưu ý

- Lần đầu chạy sẽ tải model (~500MB), mất vài phút
- Nếu có GPU, tool sẽ tự động sử dụng để tăng tốc
- Tool chỉ phân tích các dòng chưa có giá trị trust
