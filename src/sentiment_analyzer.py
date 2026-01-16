"""
Tool phân tích sentiment cho comments TikTok
Đánh giá cột text và tạo cột sentiment:
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
import time

# Try import Gemini
try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False
    print("⚠️  google-generativeai chưa được cài đặt. Chạy: pip install google-generativeai")

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
    'không sao', 'k sao', 'ko sao', 'ổn', 'fine', 'alright', 'xinh quá', 'đẹp quá',
    'ngon', 'ngon lắm', 'thích lắm', 'tuyệt vời', 'xuất sắc'
]

# Từ khóa tiêu cực tiếng Việt (mở rộng)
NEGATIVE_KEYWORDS = [
    'xấu', 'tệ', 'dở', 'tồi', 'kém', 'ghét', 'chán', 'buồn', 'thất vọng', 'tức',
    'bad', 'terrible', 'awful', 'hate', 'disgusting', 'horrible', 'worst', 'stupid',
    'ngu', 'dốt', 'đần', 'lười', 'vô dụng', 'phản đối', 'sai', 'không đúng',
    'chê', 'phê phán', 'chỉ trích', 'tức giận', 'bực', 'khó chịu',
    # Thêm từ khóa tiêu cực phổ biến
    'chịu', 'tẩy chay', 'phốt', 'drama', 'scandal', 'lỗi', 'sai lầm', 'vấn đề',
    'thất bại', 'thua', 'thua lỗ', 'giảm', 'giảm doanh thu', 'tụt dốc',
    'đi vào lòng đất', 'toàn đi vào lòng đất', 'sập', 'phá sản', 'đóng cửa',
    'cứu trợ', 'trích 1k', 'trích tiền', 'lừa đảo', 'lừa dối', 'gian dối',
    'chán ghét', 'mệt mỏi', 'bức xúc', 'tức giận', 'bực bội', 'khó chịu',
    'không tốt', 'không hay', 'không ổn', 'không được', 'dở tệ', 'tệ hại',
    'phản cảm', 'gây sốc', 'sốc', 'kinh khủng', 'khủng khiếp', 'tồi tệ',
    'tội nghiệp', 'đáng thương', 'thất vọng', 'bất ngờ tiêu cực'
]

# Cụm từ tiêu cực (phải match cả cụm)
NEGATIVE_PHRASES = [
    'chịu rồi', 'chịu thôi', 'chịu luôn', 'chịu không nổi',
    'tẩy chay', 'tẩy chay hết', 'tẩy chay luôn',
    'đi vào lòng đất', 'toàn đi vào lòng đất',
    'cứu trợ', 'trích 1k', 'trích tiền cứu trợ',
    'giảm doanh thu', 'giảm an tây',
    'hết vụ', 'hết chiến dịch', 'hết đợt',
    'từ lúc vụ', 'từ vụ',
    'chưa chừa', 'chưa bỏ',
    'tiêu chuẩn kép', 'chuẩn kép',
    'bú fame', 'làm content',
    'tối ngủ có ngon không', 'ngủ có ngon không'
]

# Cụm từ giải thích/thông tin (neutral indicators)
NEUTRAL_PHRASES = [
    'là do', 'là vì', 'chắc là', 'có thể là', 'có lẽ là',
    'nhân viên', 'nv', 'nhân viên bấm', 'nv bấm', 'nhân viên order',
    'đặt qua app', 'order qua app', 'qua app', 'đặt app',
    'note như vậy', 'ghi chú', 'note lại', 'ghi note',
    'thường là', 'thông thường', 'bình thường', 'bthg',
    'không phải', 'không phải do', 'không phải là',
    'mình từng', 'từng làm', 'từng thấy',
    'đó là', 'đây là', 'cái này là', 'cái đó là'
]

# Emoji tích cực
POSITIVE_EMOJIS = ['😊', '😍', '🥰', '😘', '😁', '😂', '🤗', '😄', '😃', '😆', '😉', 
                   '💕', '💖', '💗', '💓', '💞', '❤️', '🧡', '💛', '💚', '💙', 
                   '💜', '🤍', '🖤', '🤎', '💯', '👍', '👏', '🎉', '🎊', '✨', '🌟']

# Emoji tiêu cực
NEGATIVE_EMOJIS = ['😢', '😭', '😤', '😠', '😡', '🤬', '😞', '😔', '😟', '😕', 
                   '🙁', '☹️', '😣', '😖', '😫', '😩', '💔', '👎', '❌', '🚫']


class SentimentAnalyzer:
    """Phân tích sentiment sử dụng model đa ngôn ngữ hoặc Gemini API"""
    
    def __init__(self, model_name='cardiffnlp/twitter-roberta-base-sentiment-latest', 
                 use_gemini=False, gemini_api_key=None):
        """
        Khởi tạo sentiment analyzer
        
        Args:
            model_name: Tên model từ HuggingFace hoặc 'gemini-2.5-flash'
                       - 'cardiffnlp/twitter-roberta-base-sentiment-latest': Nhanh, hỗ trợ đa ngôn ngữ
                       - 'nlptown/bert-base-multilingual-uncased-sentiment': Chính xác hơn, chậm hơn
                       - 'gemini-2.5-flash': Sử dụng Gemini 2.5 Flash API (chính xác nhất)
            use_gemini: Nếu True, sử dụng Gemini thay vì transformer model
            gemini_api_key: API key cho Gemini (hoặc lấy từ env GEMINI_API_KEY)
        """
        self.use_gemini = use_gemini or model_name == 'gemini-2.5-flash'
        self.gemini_model = None
        self.sentiment_pipeline = None
        
        if self.use_gemini:
            if not HAS_GEMINI:
                raise ImportError("google-generativeai chưa được cài đặt. Chạy: pip install google-generativeai")
            
            # Lấy API key từ parameter hoặc environment variable
            api_key = gemini_api_key or os.getenv('GEMINI_API_KEY')
            if not api_key:
                raise ValueError("Cần cung cấp Gemini API key qua parameter gemini_api_key hoặc biến môi trường GEMINI_API_KEY")
            
            print("Đang khởi tạo Gemini 2.5 Flash...")
            try:
                genai.configure(api_key=api_key)
                # Thử dùng gemini-2.0-flash-exp (model mới nhất), fallback về gemini-1.5-flash
                try:
                    self.gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')
                    print("✅ Gemini 2.0 Flash (experimental) đã sẵn sàng")
                except:
                    self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
                    print("✅ Gemini 1.5 Flash đã sẵn sàng")
            except Exception as e:
                print(f"❌ Lỗi khi khởi tạo Gemini: {e}")
                raise
        else:
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
            tuple: (positive_score, negative_score, neutral_indicator) từ 0-1
        """
        text_lower = text.lower()
        positive_score = 0
        negative_score = 0
        neutral_indicator = 0
        
        # Kiểm tra cụm từ tiêu cực trước (quan trọng hơn)
        for phrase in NEGATIVE_PHRASES:
            if phrase in text_lower:
                negative_score += 0.5  # Cụm từ có trọng số cao hơn
        
        # Kiểm tra cụm từ neutral (giải thích/thông tin)
        for phrase in NEUTRAL_PHRASES:
            if phrase in text_lower:
                neutral_indicator += 0.4
        
        # Kiểm tra từ khóa tích cực
        for keyword in POSITIVE_KEYWORDS:
            if keyword in text_lower:
                positive_score += 0.25  # Giảm trọng số từng từ đơn
        
        # Kiểm tra từ khóa tiêu cực
        for keyword in NEGATIVE_KEYWORDS:
            if keyword in text_lower:
                negative_score += 0.25  # Giảm trọng số từng từ đơn
        
        # Kiểm tra emoji tích cực
        for emoji in POSITIVE_EMOJIS:
            if emoji in text:
                positive_score += 0.15  # Giảm trọng số emoji
        
        # Kiểm tra emoji tiêu cực
        for emoji in NEGATIVE_EMOJIS:
            if emoji in text:
                negative_score += 0.15
        
        # Xử lý các trường hợp đặc biệt
        if any(phrase in text_lower for phrase in ['không sao', 'k sao', 'ko sao', 'khong sao']):
            positive_score += 0.4
        
        # Sarcasm detection: "=))", ":))" trong context tiêu cực
        sarcasm_indicators = [':))', '=))', ':)))', '=)))', ':))))', '=))))']
        has_sarcasm = any(indicator in text for indicator in sarcasm_indicators)
        
        # Nếu có sarcasm và có từ tiêu cực -> tiêu cực mạnh hơn
        if has_sarcasm and negative_score > 0:
            negative_score += 0.3
        
        # Nếu có sarcasm và có từ tích cực trong context tiêu cực -> có thể là sarcasm
        if has_sarcasm and positive_score > 0 and negative_score > 0.3:
            positive_score = max(0, positive_score - 0.3)  # Giảm điểm tích cực
        
        return min(positive_score, 1.0), min(negative_score, 1.0), min(neutral_indicator, 1.0)
    
    def analyze_text_gemini(self, text):
        """
        Phân tích sentiment bằng Gemini API
        
        Args:
            text: Đoạn text cần phân tích
            
        Returns:
            int: 1 (positive), 0 (neutral), -1 (negative)
        """
        if pd.isna(text) or not str(text).strip():
            return 0
        
        try:
            text = str(text).strip()
            
            # Prompt cải thiện với examples và hướng dẫn rõ ràng hơn
            prompt = f"""Phân tích cảm xúc comment và trả về CHỈ MỘT SỐ: 1, 0, hoặc -1.

Comment: "{text}"

QUY TẮC:
- 1 (tích cực): Khen, thích, yêu, ủng hộ, vui, hài lòng, tốt, đẹp, ngon, hay
- 0 (trung tính): CHỈ khi là câu hỏi thuần túy, giải thích kỹ thuật, thông tin khách quan KHÔNG có cảm xúc
- -1 (tiêu cực): Chê, ghét, tức, thất vọng, chán, phê phán, sarcasm tiêu cực (=)), :)) với context tiêu cực), từ khóa: chịu, tẩy chay, phốt, drama, cứu trợ, đi vào lòng đất

VÍ DỤ:
"ngon quá" → 1
"đẹp lắm" → 1  
"tẩy chay katinat" → -1
"chịu rồi" → -1
"phốt vụ 1k" → -1
"chiến dịch đi vào lòng đất =))" → -1
"nhân viên bấm note" → 0 (giải thích kỹ thuật)
"đặt qua app như nào?" → 0 (câu hỏi)
"tui thấy hơi ấy :)" → -1 (có cảm xúc tiêu cực)

QUAN TRỌNG: Nếu có BẤT KỲ cảm xúc (dù nhẹ), đừng đánh 0. Chỉ đánh 0 khi thực sự là thông tin khách quan.

Trả về CHỈ số: 1, 0, hoặc -1"""

            response = self.gemini_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2,  # Tăng một chút để linh hoạt hơn
                    max_output_tokens=5,  # Giảm xuống vì chỉ cần số
                )
            )
            
            result_text = response.text.strip()
            
            # Parse kết quả - ưu tiên tìm số đầu tiên
            import re
            numbers = re.findall(r'-?\d+', result_text)
            if numbers:
                score = int(numbers[0])
                if score in [-1, 0, 1]:
                    return score
            
            # Nếu không tìm thấy số, dùng keyword fallback
            pos_keyword_score, neg_keyword_score, neutral_indicator = self._check_keywords_and_emojis(text)
            
            # Nếu có keyword mạnh, ưu tiên keyword
            if neg_keyword_score > 0.5:
                return -1
            elif pos_keyword_score > 0.5 and neg_keyword_score < 0.3:
                return 1
            elif neg_keyword_score > pos_keyword_score + 0.2:
                return -1
            elif pos_keyword_score > neg_keyword_score + 0.2:
                return 1
            # Nếu là giải thích kỹ thuật rõ ràng và không có cảm xúc -> neutral
            elif neutral_indicator > 0.6 and abs(pos_keyword_score - neg_keyword_score) < 0.2:
                return 0
            # Mặc định: nếu không chắc, ưu tiên cảm xúc hơn neutral
            elif neg_keyword_score > 0.2:
                return -1
            elif pos_keyword_score > 0.2:
                return 1
            else:
                return 0
                    
        except Exception as e:
            print(f"Lỗi Gemini khi phân tích: {text[:50]}... - {str(e)}")
            # Fallback to keyword-based với logic cải thiện
            pos_keyword_score, neg_keyword_score, neutral_indicator = self._check_keywords_and_emojis(text)
            if neg_keyword_score > 0.4:
                return -1
            elif pos_keyword_score > 0.4 and neg_keyword_score < 0.3:
                return 1
            elif neg_keyword_score > pos_keyword_score + 0.2:
                return -1
            elif pos_keyword_score > neg_keyword_score + 0.2:
                return 1
            elif neutral_indicator > 0.6 and abs(pos_keyword_score - neg_keyword_score) < 0.2:
                return 0
            else:
                # Nếu không chắc, ưu tiên cảm xúc hơn neutral
                if neg_keyword_score > 0.1:
                    return -1
                elif pos_keyword_score > 0.1:
                    return 1
                return 0
    
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
        
        # Nếu dùng Gemini, gọi phương thức Gemini
        if self.use_gemini and self.gemini_model:
            return self.analyze_text_gemini(text)
        
        try:
            # Giới hạn độ dài text để tránh lỗi
            text = str(text)[:512]
            
            # Kiểm tra từ khóa và emoji trước
            pos_keyword_score, neg_keyword_score, neutral_indicator = self._check_keywords_and_emojis(text)
            
            # Nếu có dấu hiệu neutral mạnh (giải thích/thông tin), ưu tiên neutral
            if neutral_indicator > 0.5 and abs(pos_keyword_score - neg_keyword_score) < 0.4:
                # Nếu là giải thích/thông tin và không có cảm xúc rõ ràng -> neutral
                return 0
            
            # Phân tích bằng model
            result = self.sentiment_pipeline(text)[0]
            label = result['label'].upper()
            score = result.get('score', 0.5)
            
            # Chuyển đổi label thành sentiment score
            model_score = 0
            model_confidence = score
            if '5 STAR' in label or '4 STAR' in label:
                model_score = 1
            elif '1 STAR' in label or '2 STAR' in label:
                model_score = -1
            elif 'POSITIVE' in label or 'POS' in label:
                model_score = 1
            elif 'NEGATIVE' in label or 'NEG' in label:
                model_score = -1
            
            # Kết hợp kết quả model với từ khóa/emoji
            # Ưu tiên keyword score nếu nó mạnh hơn model score
            final_score = model_score
            
            # Nếu keyword score rất mạnh (>0.7), ưu tiên keyword
            if neg_keyword_score > 0.7:
                final_score = -1
            elif pos_keyword_score > 0.7 and neg_keyword_score < 0.3:
                final_score = 1
            # Nếu keyword score khá mạnh (>0.5) và model confidence thấp (<0.6), ưu tiên keyword
            elif neg_keyword_score > 0.5 and model_confidence < 0.6:
                final_score = -1
            elif pos_keyword_score > 0.5 and neg_keyword_score < 0.3 and model_confidence < 0.6:
                final_score = 1
            # Nếu keyword và model conflict, ưu tiên keyword nếu mạnh hơn
            elif neg_keyword_score > pos_keyword_score + 0.4 and model_score >= 0:
                final_score = -1
            elif pos_keyword_score > neg_keyword_score + 0.4 and model_score <= 0:
                final_score = 1
            # Nếu keyword score tương đối và model confidence cao, giữ model
            elif abs(pos_keyword_score - neg_keyword_score) < 0.3 and model_confidence > 0.7:
                final_score = model_score
            # Nếu keyword difference rõ ràng (>0.3), điều chỉnh theo keyword
            elif neg_keyword_score > pos_keyword_score + 0.3:
                final_score = -1
            elif pos_keyword_score > neg_keyword_score + 0.3:
                final_score = 1
            
            # Đếm số emoji tích cực và tiêu cực (bổ sung)
            pos_emoji_count = sum(1 for emoji in POSITIVE_EMOJIS if emoji in text)
            neg_emoji_count = sum(1 for emoji in NEGATIVE_EMOJIS if emoji in text)
            
            # Nếu có nhiều emoji tiêu cực, tăng cường tiêu cực
            if neg_emoji_count >= 2 and final_score >= 0:
                final_score = -1
            # Nếu có nhiều emoji tích cực và không có từ tiêu cực mạnh, tích cực
            elif pos_emoji_count >= 2 and neg_keyword_score < 0.4 and final_score <= 0:
                final_score = 1
            
            # Xử lý trường hợp neutral: nếu có neutral indicator và không có cảm xúc rõ ràng
            if neutral_indicator > 0.4 and abs(final_score) == 1:
                # Nếu là giải thích/thông tin nhưng có cảm xúc -> giảm độ mạnh
                if final_score == 1 and pos_keyword_score < 0.5:
                    final_score = 0
                elif final_score == -1 and neg_keyword_score < 0.5:
                    final_score = 0
            
            return final_score
                
        except Exception as e:
            print(f"Lỗi khi phân tích: {text[:50]}... - {str(e)}")
            return 0
    
    def analyze_batch_gemini(self, texts, batch_size=20, progress_callback=None):
        """
        Phân tích sentiment bằng Gemini với batch processing (nhanh hơn)
        Gửi nhiều comments cùng lúc trong 1 request
        """
        results = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        if progress_callback is None:
            try:
                progress_range = tqdm(range(0, len(texts), batch_size), desc="Phân tích sentiment (Gemini)")
            except:
                progress_range = range(0, len(texts), batch_size)
        else:
            progress_range = range(0, len(texts), batch_size)
        
        for batch_idx, i in enumerate(progress_range):
            if progress_callback:
                progress_callback(batch_idx + 1, total_batches)
            
            batch_texts = texts[i:i+batch_size].tolist()
            batch_scores = []
            
            # Tạo batch prompt cho nhiều comments cùng lúc
            comments_list = []
            for idx, text in enumerate(batch_texts):
                if pd.isna(text) or not str(text).strip():
                    batch_scores.append(0)
                    continue
                
                text_clean = str(text).strip()[:500]  # Giới hạn độ dài
                comments_list.append(f"{idx + 1}. \"{text_clean}\"")
            
            if not comments_list:
                results.extend([0] * len(batch_texts))
                continue
            
            # Prompt tối ưu cho batch processing
            prompt = f"""Phân tích cảm xúc các comments sau và trả về CHỈ CÁC SỐ, mỗi dòng 1 số (1, 0, hoặc -1) tương ứng với từng comment theo thứ tự.

Comments:
{chr(10).join(comments_list)}

QUY TẮC:
- 1: Khen, thích, yêu, ủng hộ, vui, hài lòng, tốt, đẹp, ngon
- 0: CHỈ khi là câu hỏi thuần túy hoặc giải thích kỹ thuật KHÔNG có cảm xúc
- -1: Chê, ghét, tức, thất vọng, chán, phê phán, sarcasm tiêu cực (=)), :)), từ: chịu, tẩy chay, phốt, drama, cứu trợ

QUAN TRỌNG: Nếu có BẤT KỲ cảm xúc (dù nhẹ), đừng đánh 0.

Trả về CHỈ CÁC SỐ, mỗi dòng 1 số, theo thứ tự:
1
0
-1
..."""

            try:
                response = self.gemini_model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.2,
                        max_output_tokens=min(100, len(comments_list) * 5),  # Tối ưu tokens
                    )
                )
                
                result_text = response.text.strip()
                
                # Parse kết quả - tìm tất cả số
                import re
                numbers = re.findall(r'-?\d+', result_text)
                
                # Map kết quả về batch
                result_idx = 0
                for text in batch_texts:
                    if pd.isna(text) or not str(text).strip():
                        batch_scores.append(0)
                    elif result_idx < len(numbers):
                        score = int(numbers[result_idx])
                        if score in [-1, 0, 1]:
                            batch_scores.append(score)
                        else:
                            # Fallback
                            pos_keyword_score, neg_keyword_score, neutral_indicator = self._check_keywords_and_emojis(str(text))
                            if neg_keyword_score > 0.4:
                                batch_scores.append(-1)
                            elif pos_keyword_score > 0.4 and neg_keyword_score < 0.3:
                                batch_scores.append(1)
                            elif neutral_indicator > 0.6:
                                batch_scores.append(0)
                            else:
                                batch_scores.append(0 if abs(pos_keyword_score - neg_keyword_score) < 0.2 else (1 if pos_keyword_score > neg_keyword_score else -1))
                        result_idx += 1
                    else:
                        # Không đủ kết quả, dùng fallback
                        pos_keyword_score, neg_keyword_score, neutral_indicator = self._check_keywords_and_emojis(str(text))
                        if neg_keyword_score > 0.4:
                            batch_scores.append(-1)
                        elif pos_keyword_score > 0.4 and neg_keyword_score < 0.3:
                            batch_scores.append(1)
                        else:
                            batch_scores.append(0 if neutral_indicator > 0.6 else (1 if pos_keyword_score > neg_keyword_score else -1))
                
                results.extend(batch_scores)
                
                # Delay nhỏ giữa các batch
                time.sleep(0.2)  # 200ms giữa các batch thay vì từng cái
                
            except Exception as e:
                print(f"Lỗi Gemini batch: {str(e)[:100]}")
                # Fallback: phân tích từng cái bằng keyword
                for text in batch_texts:
                    if pd.isna(text) or not str(text).strip():
                        results.append(0)
                    else:
                        pos_keyword_score, neg_keyword_score, neutral_indicator = self._check_keywords_and_emojis(str(text))
                        if neg_keyword_score > 0.4:
                            results.append(-1)
                        elif pos_keyword_score > 0.4 and neg_keyword_score < 0.3:
                            results.append(1)
                        elif neutral_indicator > 0.6:
                            results.append(0)
                        else:
                            results.append(0 if abs(pos_keyword_score - neg_keyword_score) < 0.2 else (1 if pos_keyword_score > neg_keyword_score else -1))
        
        return np.array(results)
    
    def analyze_batch(self, texts, batch_size=32, progress_callback=None):
        """
        Phân tích sentiment cho nhiều texts (nhanh hơn)
        
        Args:
            texts: List hoặc Series các texts
            batch_size: Số lượng texts xử lý cùng lúc
            progress_callback: Hàm callback để cập nhật progress (current, total)
            
        Returns:
            numpy array: Mảng các sentiment scores
        """
        # Nếu dùng Gemini, sử dụng batch processing
        if self.use_gemini and self.gemini_model:
            return self.analyze_batch_gemini(texts, batch_size=min(batch_size, 20), progress_callback=progress_callback)
        
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
                    sentiment_score = self.analyze_text(text)
                    results.append(sentiment_score)
                        
                except Exception as e:
                    # Nếu lỗi, mặc định là neutral
                    print(f"Lỗi khi phân tích: {str(e)[:100]}")
                    results.append(0)
        
        return np.array(results)
    
    def process_csv(self, input_file, output_file=None, text_column='text', trust_column='sentiment', batch_size=32):
        """
        Xử lý file CSV: đọc, phân tích sentiment, và lưu kết quả
        
        Args:
            input_file: Đường dẫn file CSV đầu vào
            output_file: Đường dẫn file CSV đầu ra (nếu None thì ghi đè file đầu vào)
            text_column: Tên cột chứa text
            trust_column: Tên cột sentiment cần tạo/cập nhật (mặc định: 'sentiment')
            batch_size: Số lượng texts xử lý cùng lúc
        """
        print(f"Đang đọc file: {input_file}")
        df = pd.read_csv(input_file)
        
        print(f"Tổng số dòng: {len(df)}")
        print(f"Cột text có {df[text_column].notna().sum()} giá trị không rỗng")
        
        # Kiểm tra xem cột sentiment đã tồn tại chưa, nếu không thì thêm vào cuối
        if trust_column not in df.columns:
            df[trust_column] = None
        
        # Lọc các dòng cần phân tích (chưa có sentiment hoặc sentiment rỗng)
        mask = df[trust_column].isna() | (df[trust_column] == '')
        texts_to_analyze = df.loc[mask, text_column]
        
        if len(texts_to_analyze) == 0:
            print("Tất cả các dòng đã có sentiment score. Không cần phân tích thêm.")
            return df
        
        print(f"Số dòng cần phân tích: {len(texts_to_analyze)}")
        
        # Phân tích sentiment
        print("Bắt đầu phân tích sentiment...")
        sentiment_scores = self.analyze_batch(texts_to_analyze, batch_size=batch_size)
        
        # Cập nhật cột sentiment
        df.loc[mask, trust_column] = sentiment_scores
        
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
    
    def process_csv_dataframe(self, df, text_column='text', trust_column='sentiment', batch_size=32):
        """
        Xử lý DataFrame trực tiếp: phân tích sentiment và thêm cột sentiment
        
        Args:
            df: DataFrame cần xử lý
            text_column: Tên cột chứa text
            trust_column: Tên cột sentiment cần tạo/cập nhật (mặc định: 'sentiment')
            batch_size: Số lượng texts xử lý cùng lúc
            
        Returns:
            DataFrame: DataFrame đã được thêm cột sentiment
        """
        # Kiểm tra xem cột sentiment đã tồn tại chưa, nếu không thì thêm vào cuối
        if trust_column not in df.columns:
            df[trust_column] = None
        
        # Lọc các dòng cần phân tích (chưa có sentiment hoặc sentiment rỗng)
        mask = df[trust_column].isna() | (df[trust_column] == '')
        texts_to_analyze = df.loc[mask, text_column]
        
        if len(texts_to_analyze) == 0:
            return df
        
        # Lấy progress callback nếu có
        progress_callback = getattr(self, 'progress_callback', None)
        
        # Phân tích sentiment
        sentiment_scores = self.analyze_batch(texts_to_analyze, batch_size=batch_size, progress_callback=progress_callback)
        
        # Cập nhật cột sentiment
        df.loc[mask, trust_column] = sentiment_scores
        
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
                       default='sentiment',
                       help='Tên cột sentiment (mặc định: sentiment)')
    
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
