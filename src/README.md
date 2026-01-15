# Tool Phân Tích Sentiment cho Comments TikTok

Tool này phân tích sentiment của comments TikTok và tạo cột `trust` với giá trị:
- **1**: Tích cực (positive)
- **0**: Trung tính (neutral)
- **-1**: Tiêu cực (negative)

## Cài đặt

```bash
pip install -r requirements.txt
```

## Sử dụng

### Cách 1: Chạy trực tiếp với Python

```bash
python src/sentiment_analyzer.py --input dataset_tiktok-comments-637video-scraper_2026-01-15.csv
```

### Cách 2: Sử dụng trong code Python

```python
from src.sentiment_analyzer import SentimentAnalyzer

# Khởi tạo analyzer
analyzer = SentimentAnalyzer()

# Xử lý file CSV
analyzer.process_csv(
    input_file='dataset_tiktok-comments-637video-scraper_2026-01-15.csv',
    output_file='output.csv',  # Tùy chọn
    batch_size=32
)
```

## Tham số

- `--input, -i`: File CSV đầu vào (mặc định: `dataset_tiktok-comments-637video-scraper_2026-01-15.csv`)
- `--output, -o`: File CSV đầu ra (nếu không chỉ định thì ghi đè file đầu vào)
- `--model, -m`: Model sentiment analysis
  - `cardiffnlp/twitter-roberta-base-sentiment-latest` (mặc định, nhanh)
  - `nlptown/bert-base-multilingual-uncased-sentiment` (chính xác hơn, chậm hơn)
- `--batch-size, -b`: Kích thước batch (mặc định: 32)
- `--text-column, -t`: Tên cột chứa text (mặc định: `text`)
- `--trust-column, -c`: Tên cột trust (mặc định: `trust`)

## Ví dụ

```bash
# Sử dụng model mặc định
python src/sentiment_analyzer.py

# Sử dụng model chính xác hơn
python src/sentiment_analyzer.py --model nlptown/bert-base-multilingual-uncased-sentiment

# Chỉ định file đầu ra
python src/sentiment_analyzer.py --input input.csv --output output.csv

# Tăng batch size để xử lý nhanh hơn (nếu có GPU)
python src/sentiment_analyzer.py --batch-size 64
```

## Lưu ý

- Model sẽ được tải xuống lần đầu tiên sử dụng (khoảng 500MB)
- Nếu có GPU, tool sẽ tự động sử dụng GPU để tăng tốc
- Tool chỉ phân tích các dòng chưa có giá trị trust (bỏ qua các dòng đã có)
