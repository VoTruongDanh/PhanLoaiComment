# Tool Phân Tích Dữ Liệu Comments TikTok

Tool này cung cấp phân tích toàn diện về dữ liệu comments TikTok, bao gồm:

## Tính năng

1. **Thông tin cơ bản**: Tổng quan về dataset, số lượng comments, cột dữ liệu, dữ liệu thiếu
2. **Phân tích Sentiment**: Phân bố sentiment (tích cực/trung tính/tiêu cực), tỷ lệ, biểu đồ
3. **Phân tích Engagement**: Thống kê về likes, replies, mối quan hệ với sentiment
4. **Phân tích Thời gian**: Xu hướng comments theo ngày, giờ trong ngày
5. **Phân tích Người dùng**: Top users, phân tích sentiment theo user
6. **Phân tích Video**: Top videos, phân tích sentiment theo video
7. **Phân tích Text**: Độ dài text, mối quan hệ với sentiment
8. **Phân tích Tương quan**: Ma trận tương quan giữa các biến

## Cài đặt

```bash
pip install -r src/requirements.txt
```

## Sử dụng

### Cách 1: Chạy script đơn giản (Khuyến nghị)

```bash
python src/run_analysis.py
```

### Cách 2: Chạy với tham số tùy chỉnh

```bash
python src/data_analysis.py --input dataset_tiktok-comments-637video-scraper_2026-01-15.csv
```

### Cách 3: Sử dụng trong code Python

```python
from src.data_analysis import TikTokDataAnalyzer

# Khởi tạo analyzer
analyzer = TikTokDataAnalyzer('dataset_tiktok-comments-637video-scraper_2026-01-15.csv')

# Chạy phân tích đầy đủ
results = analyzer.run_full_analysis()

# Hoặc chạy từng phần
analyzer.load_data()
analyzer.sentiment_analysis()
analyzer.engagement_analysis()
# ...
```

## Kết quả

Tool sẽ tạo thư mục `analysis_results/` chứa:

- **sentiment_distribution.png**: Biểu đồ phân bố sentiment (bar chart và pie chart)
- **engagement_distribution.png**: Biểu đồ phân bố engagement (likes, replies)
- **time_analysis.png**: Biểu đồ phân tích thời gian (theo ngày và giờ)
- **correlation_heatmap.png**: Ma trận tương quan giữa các biến
- **analysis_report.json**: Báo cáo tổng hợp dạng JSON

## Tham số

- `--input, -i`: File CSV đầu vào (mặc định: `dataset_tiktok-comments-637video-scraper_2026-01-15.csv`)

## Yêu cầu dữ liệu

Dataset cần có các cột sau (tùy chọn, tool sẽ bỏ qua nếu không có):

- **Bắt buộc**: `text` (nội dung comment)
- **Khuyến nghị**: `trust` (sentiment score: 1, 0, -1) - cần chạy `sentiment_analyzer.py` trước
- **Tùy chọn**: 
  - `diggCount` (likes)
  - `replyCommentTotal` (số replies)
  - `createTimeISO` (thời gian)
  - `uniqueId` hoặc `uid` (user ID)
  - `videoWebUrl` (URL video)

## Ví dụ

```bash
# Phân tích file mặc định
python src/run_analysis.py

# Phân tích file khác
python src/data_analysis.py --input path/to/your/file.csv
```

## Lưu ý

- Tool sẽ tự động tạo thư mục `analysis_results/` để lưu kết quả
- Nếu file CSV không có cột `trust`, một số phân tích sẽ bị bỏ qua
- Tool hỗ trợ nhiều encoding (UTF-8, Latin-1, CP1252)
- Các biểu đồ được lưu với độ phân giải cao (300 DPI)
