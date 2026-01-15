"""
Script chạy nhanh để phân tích sentiment
Sử dụng: python src/run_sentiment.py
"""

import sys
import os

# Thêm thư mục src vào path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from sentiment_analyzer import SentimentAnalyzer

def main():
    """Chạy phân tích sentiment cho file CSV"""
    
    # File mặc định
    input_file = 'dataset_tiktok-comments-637video-scraper_2026-01-15.csv'
    
    # Kiểm tra file có tồn tại không
    if not os.path.exists(input_file):
        print(f"Không tìm thấy file: {input_file}")
        print("Vui lòng chỉ định đường dẫn file CSV:")
        input_file = input("Nhập đường dẫn file: ").strip().strip('"')
        
        if not os.path.exists(input_file):
            print(f"File không tồn tại: {input_file}")
            return
    
    print("=" * 60)
    print("TOOL PHÂN TÍCH SENTIMENT CHO COMMENTS TIKTOK")
    print("=" * 60)
    print(f"File đầu vào: {input_file}")
    print()
    
    # Khởi tạo analyzer
    print("Đang khởi tạo sentiment analyzer...")
    analyzer = SentimentAnalyzer()
    print()
    
    # Xử lý file
    try:
        analyzer.process_csv(
            input_file=input_file,
            output_file=None,  # Ghi đè file đầu vào
            batch_size=32
        )
        print("\n✓ Hoàn thành!")
    except Exception as e:
        print(f"\n✗ Lỗi: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
