"""
Tool phân tích sentiment cho comments TikTok
Đánh giá cột text và tạo cột trust:
- 1: tích cực (positive)
- 0: trung tính (neutral)
- -1: tiêu cực (negative)
"""

import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import torch
from tqdm import tqdm
import warnings
import sys
import os
import re

# Fix encoding for Windows console
if sys.platform == 'win32':
    try:
        # Set console to UTF-8
        os.system('chcp 65001 >nul 2>&1')
        sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None
        sys.stderr.reconfigure(encoding='utf-8') if hasattr(sys.stderr, 'reconfigure') else None
    except:
        pass

warnings.filterwarnings('ignore')

# Từ khóa tích cực tiếng Việt
POSITIVE_KEYWORDS = [
    'xinh', 'đẹp', 'cute', 'dễ thương', 'hay', 'tốt', 'tuyệt', 'vui', 'thích', 'yêu',
    'love', 'amazing', 'great', 'good', 'nice', 'beautiful', 'wonderful', 'awesome',
    'thú vị', 'hài', 'vui vẻ', 'hạnh phúc', 'tuyệt vời', 'xuất sắc', 'giỏi', 'tài',
    'khen', 'khen ngợi', 'ủng hộ', 'đồng ý', 'đúng', 'chính xác', 'chuẩn', 'ok', 'okay',
    'không sao', 'k sao', 'ko sao', 'ổn', 'ok', 'fine', 'alright'
]

# Từ khóa tiêu cực tiếng Việt
NEGATIVE_KEYWORDS = [
    'xấu', 'tệ', 'dở', 'tồi', 'kém', 'ghét', 'chán', 'buồn', 'thất vọng', 'tức',
    'bad', 'terrible', 'awful', 'hate', 'disgusting', 'horrible', 'worst', 'stupid',
    'ngu', 'dốt', 'đần', 'lười', 'vô dụng', 'phản đối', 'sai', 'không đúng',
    'chê', 'phê phán', 'chỉ trích', 'tức giận', 'bực', 'khó chịu'
]

# Emoji tích cực
POSITIVE_EMOJIS = ['😊', '😍', '🥰', '😘', '😁', '😂', '🤗', '😄', '😃', '😆', '😉', 
                   '💕', '💖', '💗', '💓', '💞', '❤️', '🧡', '💛', '💚', '💙', 
                   '💜', '🤍', '🖤', '🤎', '💯', '👍', '👏', '🎉', '🎊', '✨', '🌟']

# Emoji tiêu cực
NEGATIVE_EMOJIS = ['😢', '😭', '😤', '😠', '😡', '🤬', '😞', '😔', '😟', '😕', 
                   '🙁', '☹️', '😣', '😖', '😫', '😩', '💔', '👎', '❌', '🚫']


class SentimentAnalyzer:
    """Phân tích sentiment sử dụng model đa ngôn ngữ"""
    
    def __init__(self, model_name='cardiffnlp/twitter-roberta-base-sentiment-latest'):
        """
        Khởi tạo sentiment analyzer
        
        Args:
            model_name: Tên model từ HuggingFace
                       - 'cardiffnlp/twitter-roberta-base-sentiment-latest': Nhanh, hỗ trợ đa ngôn ngữ
                       - 'nlptown/bert-base-multilingual-uncased-sentiment': Chính xác hơn, chậm hơn
        """
        print(f"Đang tải model: {model_name}...")
        self.device = 0 if torch.cuda.is_available() else -1
        try:
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=model_name,
                tokenizer=model_name,
                device=self.device,
                return_all_scores=False,
                truncation=True,
                max_length=512
            )
            print(f"Model đã sẵn sàng (device: {'GPU' if self.device >= 0 else 'CPU'})")
        except Exception as e:
            print(f"Lỗi khi tải model: {e}")
            print("Đang thử model dự phòng...")
            # Fallback to multilingual model
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model='nlptown/bert-base-multilingual-uncased-sentiment',
                device=self.device,
                return_all_scores=False,
                truncation=True,
                max_length=512
            )
            print("Đã tải model dự phòng thành công")
    
    def _check_keywords_and_emojis(self, text):
        """
        Kiểm tra từ khóa và emoji để bổ sung cho phân tích sentiment
        
        Returns:
            tuple: (positive_score, negative_score) từ 0-1
        """
        text_lower = text.lower()
        positive_score = 0
        negative_score = 0
        
        # Kiểm tra từ khóa tích cực
        for keyword in POSITIVE_KEYWORDS:
            if keyword in text_lower:
                positive_score += 0.3
        
        # Kiểm tra từ khóa tiêu cực
        for keyword in NEGATIVE_KEYWORDS:
            if keyword in text_lower:
                negative_score += 0.3
        
        # Kiểm tra emoji tích cực
        for emoji in POSITIVE_EMOJIS:
            if emoji in text:
                positive_score += 0.2
        
        # Kiểm tra emoji tiêu cực
        for emoji in NEGATIVE_EMOJIS:
            if emoji in text:
                negative_score += 0.2
        
        # Xử lý các trường hợp đặc biệt
        if any(phrase in text_lower for phrase in ['không sao', 'k sao', 'ko sao', 'khong sao']):
            positive_score += 0.5
        
        if any(phrase in text_lower for phrase in ['haha', 'hihi', 'hehe']):
            positive_score += 0.3
        
        return min(positive_score, 1.0), min(negative_score, 1.0)
    
    def analyze_text(self, text):
        """
        Phân tích sentiment cho một đoạn text
        
        Args:
            text: Đoạn text cần phân tích
            
        Returns:
            int: 1 (positive), 0 (neutral), -1 (negative)
        """
        if pd.isna(text) or not str(text).strip():
            return 0
        
        try:
            # Giới hạn độ dài text để tránh lỗi
            text = str(text)[:512]
            
            # Kiểm tra từ khóa và emoji trước
            pos_keyword_score, neg_keyword_score = self._check_keywords_and_emojis(text)
            
            # Phân tích bằng model
            result = self.sentiment_pipeline(text)[0]
            label = result['label'].upper()
            score = result.get('score', 0.5)
            
            # Chuyển đổi label thành trust score
            model_score = 0
            if '5 STAR' in label or '4 STAR' in label:
                model_score = 1
            elif '1 STAR' in label or '2 STAR' in label:
                model_score = -1
            elif 'POSITIVE' in label or 'POS' in label:
                model_score = 1
            elif 'NEGATIVE' in label or 'NEG' in label:
                model_score = -1
            
            # Kết hợp kết quả model với từ khóa/emoji
            final_score = model_score
            
            # Đếm số emoji tích cực và tiêu cực
            pos_emoji_count = sum(1 for emoji in POSITIVE_EMOJIS if emoji in text)
            neg_emoji_count = sum(1 for emoji in NEGATIVE_EMOJIS if emoji in text)
            
            # Nếu có nhiều emoji tích cực, ưu tiên tích cực
            if pos_emoji_count >= 2 and model_score <= 0:
                final_score = 1
            # Nếu có emoji tích cực và từ khóa tích cực, ưu tiên tích cực
            elif pos_emoji_count >= 1 and pos_keyword_score > 0.3 and model_score <= 0:
                final_score = 1
            # Nếu có nhiều emoji tiêu cực, ưu tiên tiêu cực
            elif neg_emoji_count >= 2 and model_score >= 0:
                final_score = -1
            # Nếu từ khóa/emoji mạnh, điều chỉnh kết quả
            elif pos_keyword_score > 0.5 and model_score <= 0:
                final_score = 1  # Ưu tiên tích cực nếu có nhiều dấu hiệu tích cực
            elif neg_keyword_score > 0.5 and model_score >= 0:
                final_score = -1  # Ưu tiên tiêu cực nếu có nhiều dấu hiệu tiêu cực
            elif pos_keyword_score > neg_keyword_score + 0.3:
                final_score = 1
            elif neg_keyword_score > pos_keyword_score + 0.3:
                final_score = -1
            
            return final_score
                
        except Exception as e:
            print(f"Lỗi khi phân tích: {text[:50]}... - {str(e)}")
            return 0
    
    def analyze_batch(self, texts, batch_size=32, progress_callback=None):
        """
        Phân tích sentiment cho nhiều texts (nhanh hơn)
        
        Args:
            texts: List hoặc Series các texts
            batch_size: Số lượng texts xử lý cùng lúc
            progress_callback: Hàm callback để cập nhật progress (current, total)
            
        Returns:
            numpy array: Mảng các trust scores
        """
        results = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        # Xử lý theo batch để tăng tốc
        # Sử dụng tqdm nếu không có callback (cho CLI)
        if progress_callback is None:
            try:
                progress_range = tqdm(range(0, len(texts), batch_size), desc="Phân tích sentiment")
            except:
                progress_range = range(0, len(texts), batch_size)
        else:
            progress_range = range(0, len(texts), batch_size)
        
        for batch_idx, i in enumerate(progress_range):
            if progress_callback:
                progress_callback(batch_idx + 1, total_batches)
            batch = texts[i:i+batch_size].tolist()
            
            # Xử lý từng text trong batch để tránh lỗi
            for text in batch:
                if pd.isna(text) or not str(text).strip():
                    results.append(0)
                    continue
                
                try:
                    # Sử dụng analyze_text để có logic cải thiện
                    trust_score = self.analyze_text(text)
                    results.append(trust_score)
                except Exception as e:
                    # Nếu lỗi, mặc định là neutral
                    results.append(0)
        
        return np.array(results)
    
    def process_csv(self, input_file, output_file=None, text_column='text', trust_column='trust', batch_size=32):
        """
        Xử lý file CSV: đọc, phân tích sentiment, và lưu kết quả
        
        Args:
            input_file: Đường dẫn file CSV đầu vào
            output_file: Đường dẫn file CSV đầu ra (nếu None thì ghi đè file đầu vào)
            text_column: Tên cột chứa text
            trust_column: Tên cột trust cần tạo/cập nhật
            batch_size: Số lượng texts xử lý cùng lúc
        """
        print(f"Đang đọc file: {input_file}")
        df = pd.read_csv(input_file)
        
        print(f"Tổng số dòng: {len(df)}")
        print(f"Cột text có {df[text_column].notna().sum()} giá trị không rỗng")
        
        # Kiểm tra xem cột trust đã tồn tại chưa
        if trust_column not in df.columns:
            df[trust_column] = None
        
        # Lọc các dòng cần phân tích (chưa có trust hoặc trust rỗng)
        mask = df[trust_column].isna() | (df[trust_column] == '')
        texts_to_analyze = df.loc[mask, text_column]
        
        if len(texts_to_analyze) == 0:
            print("Tất cả các dòng đã có trust score. Không cần phân tích thêm.")
            return df
        
        print(f"Số dòng cần phân tích: {len(texts_to_analyze)}")
        
        # Phân tích sentiment
        print("Bắt đầu phân tích sentiment...")
        trust_scores = self.analyze_batch(texts_to_analyze, batch_size=batch_size)
        
        # Cập nhật cột trust
        df.loc[mask, trust_column] = trust_scores
        
        # Thống kê kết quả
        print("\n=== Thống kê kết quả ===")
        print(f"Tích cực (1): {(df[trust_column] == 1).sum()} ({((df[trust_column] == 1).sum() / len(df) * 100):.2f}%)")
        print(f"Trung tính (0): {(df[trust_column] == 0).sum()} ({((df[trust_column] == 0).sum() / len(df) * 100):.2f}%)")
        print(f"Tiêu cực (-1): {(df[trust_column] == -1).sum()} ({((df[trust_column] == -1).sum() / len(df) * 100):.2f}%)")
        
        # Lưu file
        if output_file is None:
            output_file = input_file
        
        print(f"\nĐang lưu kết quả vào: {output_file}")
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print("Hoàn thành!")
        
        return df
    
    def process_csv_dataframe(self, df, text_column='text', trust_column='trust', batch_size=32):
        """
        Xử lý DataFrame trực tiếp: phân tích sentiment và thêm cột trust
        
        Args:
            df: DataFrame cần xử lý
            text_column: Tên cột chứa text
            trust_column: Tên cột trust cần tạo/cập nhật
            batch_size: Số lượng texts xử lý cùng lúc
            
        Returns:
            DataFrame: DataFrame đã được thêm cột trust
        """
        # Kiểm tra xem cột trust đã tồn tại chưa
        if trust_column not in df.columns:
            df[trust_column] = None
        
        # Lọc các dòng cần phân tích (chưa có trust hoặc trust rỗng)
        mask = df[trust_column].isna() | (df[trust_column] == '')
        texts_to_analyze = df.loc[mask, text_column]
        
        if len(texts_to_analyze) == 0:
            return df
        
        # Lấy progress callback nếu có
        progress_callback = getattr(self, 'progress_callback', None)
        
        # Phân tích sentiment
        trust_scores = self.analyze_batch(texts_to_analyze, batch_size=batch_size, progress_callback=progress_callback)
        
        # Cập nhật cột trust
        df.loc[mask, trust_column] = trust_scores
        
        return df


def main():
    """Hàm main để chạy tool"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Phân tích sentiment cho comments TikTok')
    parser.add_argument('--input', '-i', 
                       default='dataset_tiktok-comments-637video-scraper_2026-01-15.csv',
                       help='File CSV đầu vào')
    parser.add_argument('--output', '-o', 
                       default=None,
                       help='File CSV đầu ra (nếu không chỉ định thì ghi đè file đầu vào)')
    parser.add_argument('--model', '-m',
                       default='cardiffnlp/twitter-roberta-base-sentiment-latest',
                       choices=['cardiffnlp/twitter-roberta-base-sentiment-latest',
                               'nlptown/bert-base-multilingual-uncased-sentiment'],
                       help='Model sentiment analysis')
    parser.add_argument('--batch-size', '-b',
                       type=int, default=32,
                       help='Kích thước batch (mặc định: 32)')
    parser.add_argument('--text-column', '-t',
                       default='text',
                       help='Tên cột chứa text (mặc định: text)')
    parser.add_argument('--trust-column', '-c',
                       default='trust',
                       help='Tên cột trust (mặc định: trust)')
    
    args = parser.parse_args()
    
    # Khởi tạo analyzer
    analyzer = SentimentAnalyzer(model_name=args.model)
    
    # Xử lý file
    analyzer.process_csv(
        input_file=args.input,
        output_file=args.output,
        text_column=args.text_column,
        trust_column=args.trust_column,
        batch_size=args.batch_size
    )


if __name__ == '__main__':
    main()
