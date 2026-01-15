# Tóm Tắt Phân Tích Dữ Liệu Comments TikTok

**Ngày phân tích**: 2026-01-16  
**Dataset**: dataset_tiktok-comments-637video-scraper_2026-01-15.csv  
**Tổng số comments**: 4,000

---

## 📊 Tổng Quan

### Thông tin Dataset
- **Tổng số comments**: 4,000
- **Số cột**: 10
- **Dữ liệu thiếu**: 27 comments thiếu text (0.68%)
- **Khoảng thời gian**: 2021-02-25 đến 2026-01-15 (1,784 ngày)

---

## 😊 Phân Tích Sentiment

### Phân Bố Sentiment
- **Tích cực (1)**: 2,004 comments (50.10%)
- **Trung tính (0)**: 689 comments (17.22%)
- **Tiêu cực (-1)**: 1,307 comments (32.67%)

### Nhận Xét
- Tỷ lệ comments tích cực cao hơn tiêu cực (50.10% vs 32.67%)
- Gần 1/3 comments có sentiment tiêu cực, cần chú ý
- Tỷ lệ trung tính thấp (17.22%), cho thấy người dùng có xu hướng thể hiện cảm xúc rõ ràng

---

## 👍 Phân Tích Engagement

### Likes (diggCount)
- **Tổng**: 81,945 likes
- **Trung bình**: 20.49 likes/comment
- **Median**: 1 like/comment
- **Max**: 8,101 likes (comment nổi bật nhất)
- **Phân bố**: Rất lệch, hầu hết comments có ít likes, một số ít có rất nhiều

### Replies (replyCommentTotal)
- **Tổng**: 5,658 replies
- **Trung bình**: 1.41 replies/comment
- **Median**: 0 replies (hầu hết không có reply)
- **Max**: 263 replies (comment gây tranh cãi nhất)

### Engagement theo Sentiment

**Likes:**
- Tiêu cực: TB = 19.09, Median = 1.00
- Trung tính: TB = 24.72, Median = 1.00 (cao nhất!)
- Tích cực: TB = 19.94, Median = 1.00

**Replies:**
- Tiêu cực: TB = 1.83, Median = 0.00 (cao nhất - gây tranh cãi)
- Trung tính: TB = 1.46, Median = 0.00
- Tích cực: TB = 1.12, Median = 0.00

### Nhận Xét
- Comments trung tính có nhiều likes nhất (có thể do tính khách quan)
- Comments tiêu cực có nhiều replies nhất (gây tranh cãi, tạo discussion)
- Comments tích cực có engagement thấp hơn, nhưng vẫn chiếm đa số

---

## ⏰ Phân Tích Thời Gian

### Theo Ngày
- **Trung bình**: 6.13 comments/ngày
- **Min**: 1 comment
- **Max**: 125 comments (ngày có nhiều hoạt động nhất)

### Theo Giờ
- **Giờ cao điểm**: 14h (266 comments)
- **Giờ thấp điểm**: 20h (52 comments)

### Nhận Xét
- Hoạt động tập trung vào buổi chiều (14h)
- Buổi tối (20h) có ít hoạt động nhất
- Phân bố không đều, có ngày có rất nhiều comments (125 comments)

---

## 👥 Phân Tích Người Dùng

- **Tổng số users**: 3,896
- **Trung bình comments/user**: 1.03
- **Top user**: harrynguyen1104 (5 comments)

### Nhận Xét
- Hầu hết users chỉ comment 1 lần (phân bố rất rộng)
- Không có user nào comment quá nhiều (tối đa 5 comments)
- Cộng đồng phân tán, không tập trung vào một số users

---

## 🎬 Phân Tích Video

- **Tổng số video**: 209
- **Trung bình comments/video**: 19.14
- **Video có nhiều comments nhất**: 82 comments

### Top Videos Tích Cực Nhất (>= 10 comments)
1. TB sentiment = 1.00 (15 comments)
2. TB sentiment = 1.00 (17 comments)
3. TB sentiment = 0.97 (36 comments)

### Top Videos Tiêu Cực Nhất (>= 10 comments)
1. TB sentiment = -0.60 (58 comments)
2. TB sentiment = -0.57 (81 comments)
3. TB sentiment = -0.54 (13 comments)

### Nhận Xét
- Một số video có sentiment rất tích cực (100% positive)
- Một số video có sentiment rất tiêu cực (cần xem xét nội dung)
- Video có nhiều comments nhất (81 comments) có sentiment tiêu cực (-0.57)

---

## 📝 Phân Tích Text

### Độ Dài Text
- **Trung bình**: 31.69 ký tự
- **Median**: 22 ký tự
- **Min**: 1 ký tự
- **Max**: 914 ký tự

### Độ Dài theo Sentiment
- **Tiêu cực**: TB = 42.13, Median = 30 ký tự (dài nhất)
- **Trung tính**: TB = 30.92, Median = 21 ký tự
- **Tích cực**: TB = 25.14, Median = 16 ký tự (ngắn nhất)

### Nhận Xét
- Comments tiêu cực thường dài hơn (có thể do giải thích, phàn nàn)
- Comments tích cực thường ngắn gọn (ví dụ: "hay quá", "thích lắm")
- Có comment dài tới 914 ký tự (câu chuyện dài)

---

## 🔗 Phân Tích Tương Quan

### Ma Trận Tương Quan

| Biến | trust | diggCount | replyCommentTotal | text_length |
|------|-------|-----------|-------------------|-------------|
| **trust** | 1.00 | 0.00 | -0.04 | **-0.20** |
| **diggCount** | 0.00 | 1.00 | **0.38** | 0.03 |
| **replyCommentTotal** | -0.04 | **0.38** | 1.00 | 0.09 |
| **text_length** | **-0.20** | 0.03 | 0.09 | 1.00 |

### Nhận Xét
- **Sentiment và độ dài text**: Tương quan âm mạnh (-0.20) - comments dài hơn có xu hướng tiêu cực hơn
- **Likes và Replies**: Tương quan dương mạnh (0.38) - comments có nhiều likes thường có nhiều replies
- **Sentiment và Engagement**: Tương quan rất yếu (0.00, -0.04) - sentiment không ảnh hưởng nhiều đến likes/replies

---

## 💡 Kết Luận & Khuyến Nghị

### Điểm Mạnh
1. ✅ Tỷ lệ comments tích cực cao (50.10%)
2. ✅ Engagement tốt (trung bình 20 likes/comment)
3. ✅ Cộng đồng đa dạng (3,896 users)

### Điểm Cần Cải Thiện
1. ⚠️ Tỷ lệ comments tiêu cực còn cao (32.67%)
2. ⚠️ Comments tiêu cực có nhiều replies (gây tranh cãi)
3. ⚠️ Một số video có sentiment rất tiêu cực cần xem xét

### Khuyến Nghị
1. **Theo dõi các video có sentiment tiêu cực cao** để cải thiện nội dung
2. **Quản lý comments tiêu cực** có nhiều replies để tránh tranh cãi
3. **Khuyến khích comments tích cực** để tăng tỷ lệ positive
4. **Phân tích sâu hơn** về các video có nhiều comments tiêu cực để tìm nguyên nhân

---

## 📁 Files Kết Quả

Tất cả kết quả đã được lưu trong thư mục `analysis_results/`:
- `sentiment_distribution.png` - Biểu đồ phân bố sentiment
- `engagement_distribution.png` - Biểu đồ phân bố engagement
- `time_analysis.png` - Biểu đồ phân tích thời gian
- `correlation_heatmap.png` - Ma trận tương quan
- `analysis_report.json` - Báo cáo chi tiết dạng JSON

---

**Tool được tạo bởi**: Data Analysis Script  
**Phiên bản**: 1.0  
**Ngày**: 2026-01-16
