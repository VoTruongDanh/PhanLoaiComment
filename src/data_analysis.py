"""
Comprehensive Data Analysis Script for TikTok Comments Dataset
Phân tích toàn diện dữ liệu comments TikTok
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import sys
import os
import re
from collections import Counter
import json

# Fix encoding for Windows console
if sys.platform == 'win32':
    try:
        os.system('chcp 65001 >nul 2>&1')
        sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None
        sys.stderr.reconfigure(encoding='utf-8') if hasattr(sys.stderr, 'reconfigure') else None
    except:
        pass

warnings.filterwarnings('ignore')

# Set style for plots
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        plt.style.use('default')
sns.set_palette("husl")

class TikTokDataAnalyzer:
    """Phân tích dữ liệu comments TikTok"""
    
    def __init__(self, csv_file):
        """
        Khởi tạo analyzer
        
        Args:
            csv_file: Đường dẫn file CSV
        """
        self.csv_file = csv_file
        self.df = None
        self.output_dir = 'analysis_results'
        
        # Tạo thư mục output nếu chưa có
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def load_data(self):
        """Đọc dữ liệu từ file CSV"""
        print(f"Đang đọc file: {self.csv_file}")
        try:
            self.df = pd.read_csv(self.csv_file, encoding='utf-8-sig')
        except:
            try:
                self.df = pd.read_csv(self.csv_file, encoding='latin-1')
            except:
                self.df = pd.read_csv(self.csv_file, encoding='cp1252')
        
        print(f"Đã tải {len(self.df)} dòng dữ liệu")
        print(f"Các cột: {list(self.df.columns)}")
        return self.df
    
    def basic_info(self):
        """Thông tin cơ bản về dataset"""
        print("\n" + "="*80)
        print("1. THÔNG TIN CƠ BẢN VỀ DATASET")
        print("="*80)
        
        print(f"\nTổng số comments: {len(self.df):,}")
        print(f"Số cột: {len(self.df.columns)}")
        print(f"\nCác cột trong dataset:")
        for col in self.df.columns:
            print(f"  - {col}")
        
        print(f"\nThông tin về dữ liệu thiếu:")
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df) * 100).round(2)
        missing_df = pd.DataFrame({
            'Số lượng thiếu': missing,
            'Tỷ lệ %': missing_pct
        })
        missing_df = missing_df[missing_df['Số lượng thiếu'] > 0]
        if len(missing_df) > 0:
            print(missing_df.to_string())
        else:
            print("  Không có dữ liệu thiếu")
        
        print(f"\nThông tin về kiểu dữ liệu:")
        print(self.df.dtypes.to_string())
        
        return {
            'total_comments': len(self.df),
            'total_columns': len(self.df.columns),
            'missing_data': missing.to_dict()
        }
    
    def sentiment_analysis(self):
        """Phân tích sentiment"""
        print("\n" + "="*80)
        print("2. PHÂN TÍCH SENTIMENT")
        print("="*80)
        
        if 'trust' not in self.df.columns:
            print("⚠️  Cột 'trust' không tồn tại. Vui lòng chạy sentiment_analyzer.py trước.")
            return None
        
        # Chuyển đổi trust sang số nếu cần
        self.df['trust'] = pd.to_numeric(self.df['trust'], errors='coerce')
        
        # Thống kê sentiment
        sentiment_counts = self.df['trust'].value_counts().sort_index()
        sentiment_labels = {-1: 'Tiêu cực', 0: 'Trung tính', 1: 'Tích cực'}
        
        print("\nPhân bố sentiment:")
        total = len(self.df)
        for val, label in sentiment_labels.items():
            count = sentiment_counts.get(val, 0)
            pct = (count / total * 100) if total > 0 else 0
            print(f"  {label} ({val}): {count:,} ({pct:.2f}%)")
        
        # Tính tỷ lệ tích cực
        positive_ratio = (self.df['trust'] == 1).sum() / total if total > 0 else 0
        negative_ratio = (self.df['trust'] == -1).sum() / total if total > 0 else 0
        
        print(f"\nTỷ lệ tích cực: {positive_ratio*100:.2f}%")
        print(f"Tỷ lệ tiêu cực: {negative_ratio*100:.2f}%")
        print(f"Tỷ lệ trung tính: {(1 - positive_ratio - negative_ratio)*100:.2f}%")
        
        # Vẽ biểu đồ
        self._plot_sentiment_distribution()
        
        return {
            'sentiment_counts': sentiment_counts.to_dict(),
            'positive_ratio': positive_ratio,
            'negative_ratio': negative_ratio
        }
    
    def _plot_sentiment_distribution(self):
        """Vẽ biểu đồ phân bố sentiment"""
        sentiment_counts = self.df['trust'].value_counts().sort_index()
        sentiment_labels = {-1: 'Tiêu cực', 0: 'Trung tính', 1: 'Tích cực'}
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Bar chart
        labels = [sentiment_labels.get(val, str(val)) for val in sentiment_counts.index]
        colors = ['#dc2626', '#6b7280', '#059669']
        ax1.bar(labels, sentiment_counts.values, color=colors)
        ax1.set_title('Phân Bố Sentiment (Bar Chart)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Số lượng', fontsize=12)
        ax1.set_xlabel('Sentiment', fontsize=12)
        for i, v in enumerate(sentiment_counts.values):
            ax1.text(i, v, f'{v:,}', ha='center', va='bottom', fontsize=10)
        
        # Pie chart
        ax2.pie(sentiment_counts.values, labels=labels, autopct='%1.1f%%', 
                colors=colors, startangle=90)
        ax2.set_title('Phân Bố Sentiment (Pie Chart)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/sentiment_distribution.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Đã lưu biểu đồ: {self.output_dir}/sentiment_distribution.png")
        plt.close()
    
    def engagement_analysis(self):
        """Phân tích engagement (likes, replies)"""
        print("\n" + "="*80)
        print("3. PHÂN TÍCH ENGAGEMENT")
        print("="*80)
        
        # Kiểm tra các cột engagement
        engagement_cols = {}
        if 'diggCount' in self.df.columns:
            engagement_cols['Likes'] = 'diggCount'
        if 'replyCommentTotal' in self.df.columns:
            engagement_cols['Replies'] = 'replyCommentTotal'
        
        if not engagement_cols:
            print("⚠️  Không tìm thấy cột engagement (diggCount, replyCommentTotal)")
            return None
        
        results = {}
        
        for label, col in engagement_cols.items():
            # Chuyển đổi sang số
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            
            print(f"\n{label} ({col}):")
            print(f"  Tổng: {self.df[col].sum():,.0f}")
            print(f"  Trung bình: {self.df[col].mean():.2f}")
            print(f"  Median: {self.df[col].median():.2f}")
            print(f"  Min: {self.df[col].min():.0f}")
            print(f"  Max: {self.df[col].max():,.0f}")
            print(f"  Std: {self.df[col].std():.2f}")
            
            results[label] = {
                'total': float(self.df[col].sum()),
                'mean': float(self.df[col].mean()),
                'median': float(self.df[col].median()),
                'min': float(self.df[col].min()),
                'max': float(self.df[col].max()),
                'std': float(self.df[col].std())
            }
        
        # Phân tích engagement theo sentiment
        if 'trust' in self.df.columns:
            print("\nEngagement theo Sentiment:")
            for label, col in engagement_cols.items():
                print(f"\n{label} theo Sentiment:")
                sentiment_engagement = self.df.groupby('trust')[col].agg(['mean', 'median', 'sum'])
                for sentiment in [-1, 0, 1]:
                    sentiment_label = {-1: 'Tiêu cực', 0: 'Trung tính', 1: 'Tích cực'}.get(sentiment, str(sentiment))
                    if sentiment in sentiment_engagement.index:
                        mean_val = sentiment_engagement.loc[sentiment, 'mean']
                        median_val = sentiment_engagement.loc[sentiment, 'median']
                        total_val = sentiment_engagement.loc[sentiment, 'sum']
                        print(f"  {sentiment_label}: TB={mean_val:.2f}, Median={median_val:.2f}, Tổng={total_val:,.0f}")
        
        # Vẽ biểu đồ
        self._plot_engagement(engagement_cols)
        
        return results
    
    def _plot_engagement(self, engagement_cols):
        """Vẽ biểu đồ engagement"""
        n_cols = len(engagement_cols)
        if n_cols == 0:
            return
        
        fig, axes = plt.subplots(1, n_cols, figsize=(6*n_cols, 6))
        if n_cols == 1:
            axes = [axes]
        
        for idx, (label, col) in enumerate(engagement_cols.items()):
            ax = axes[idx]
            # Box plot
            data = self.df[col].dropna()
            # Chỉ vẽ nếu có dữ liệu
            if len(data) > 0:
                # Log scale nếu cần
                if data.max() / data.min() > 100:
                    data = np.log1p(data)
                    ax.set_ylabel(f'Log({label})', fontsize=12)
                else:
                    ax.set_ylabel(label, fontsize=12)
                
                ax.boxplot(data, vert=True)
                ax.set_title(f'Phân Bố {label}', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/engagement_distribution.png', dpi=300, bbox_inches='tight')
        print(f"✓ Đã lưu biểu đồ: {self.output_dir}/engagement_distribution.png")
        plt.close()
    
    def time_analysis(self):
        """Phân tích theo thời gian"""
        print("\n" + "="*80)
        print("4. PHÂN TÍCH THEO THỜI GIAN")
        print("="*80)
        
        if 'createTimeISO' not in self.df.columns:
            print("⚠️  Không tìm thấy cột 'createTimeISO'")
            return None
        
        # Chuyển đổi sang datetime
        self.df['datetime'] = pd.to_datetime(self.df['createTimeISO'], errors='coerce')
        
        # Lọc bỏ các giá trị không hợp lệ
        valid_dates = self.df['datetime'].notna()
        if valid_dates.sum() == 0:
            print("⚠️  Không có dữ liệu thời gian hợp lệ")
            return None
        
        df_time = self.df[valid_dates].copy()
        
        # Thông tin thời gian
        print(f"\nThời gian sớm nhất: {df_time['datetime'].min()}")
        print(f"Thời gian muộn nhất: {df_time['datetime'].max()}")
        print(f"Khoảng thời gian: {(df_time['datetime'].max() - df_time['datetime'].min()).days} ngày")
        
        # Phân tích theo ngày
        df_time['date'] = df_time['datetime'].dt.date
        daily_counts = df_time['date'].value_counts().sort_index()
        
        print(f"\nSố lượng comments theo ngày:")
        print(f"  Trung bình: {daily_counts.mean():.2f} comments/ngày")
        print(f"  Min: {daily_counts.min()} comments")
        print(f"  Max: {daily_counts.max()} comments")
        
        # Phân tích theo giờ
        df_time['hour'] = df_time['datetime'].dt.hour
        hourly_counts = df_time['hour'].value_counts().sort_index()
        
        print(f"\nGiờ có nhiều comments nhất: {hourly_counts.idxmax()}h ({hourly_counts.max()} comments)")
        print(f"Giờ có ít comments nhất: {hourly_counts.idxmin()}h ({hourly_counts.min()} comments)")
        
        # Vẽ biểu đồ
        self._plot_time_analysis(df_time, daily_counts, hourly_counts)
        
        return {
            'date_range': {
                'start': str(df_time['datetime'].min()),
                'end': str(df_time['datetime'].max()),
                'days': int((df_time['datetime'].max() - df_time['datetime'].min()).days)
            },
            'daily_stats': {
                'mean': float(daily_counts.mean()),
                'min': int(daily_counts.min()),
                'max': int(daily_counts.max())
            }
        }
    
    def _plot_time_analysis(self, df_time, daily_counts, hourly_counts):
        """Vẽ biểu đồ phân tích thời gian"""
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Daily comments
        axes[0].plot(daily_counts.index, daily_counts.values, marker='o', linewidth=2, markersize=4)
        axes[0].set_title('Số Lượng Comments Theo Ngày', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Ngày', fontsize=12)
        axes[0].set_ylabel('Số lượng comments', fontsize=12)
        axes[0].grid(True, alpha=0.3)
        axes[0].tick_params(axis='x', rotation=45)
        
        # Hourly comments
        axes[1].bar(hourly_counts.index, hourly_counts.values, color='steelblue', alpha=0.7)
        axes[1].set_title('Số Lượng Comments Theo Giờ Trong Ngày', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Giờ', fontsize=12)
        axes[1].set_ylabel('Số lượng comments', fontsize=12)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/time_analysis.png', dpi=300, bbox_inches='tight')
        print(f"✓ Đã lưu biểu đồ: {self.output_dir}/time_analysis.png")
        plt.close()
    
    def user_analysis(self):
        """Phân tích người dùng"""
        print("\n" + "="*80)
        print("5. PHÂN TÍCH NGƯỜI DÙNG")
        print("="*80)
        
        user_cols = {}
        if 'uniqueId' in self.df.columns:
            user_cols['uniqueId'] = 'uniqueId'
        elif 'uid' in self.df.columns:
            user_cols['uid'] = 'uid'
        
        if not user_cols:
            print("⚠️  Không tìm thấy cột user ID")
            return None
        
        user_col = list(user_cols.values())[0]
        
        # Thống kê users
        total_users = self.df[user_col].nunique()
        total_comments = len(self.df)
        
        print(f"\nTổng số users: {total_users:,}")
        print(f"Tổng số comments: {total_comments:,}")
        print(f"Trung bình comments/user: {total_comments/total_users:.2f}")
        
        # Top users
        user_comment_counts = self.df[user_col].value_counts()
        print(f"\nTop 10 users có nhiều comments nhất:")
        for i, (user, count) in enumerate(user_comment_counts.head(10).items(), 1):
            print(f"  {i}. {user}: {count} comments")
        
        # Phân tích sentiment theo user (nếu có)
        if 'trust' in self.df.columns:
            print(f"\nPhân tích sentiment theo user:")
            user_sentiment = self.df.groupby(user_col)['trust'].agg(['mean', 'count'])
            user_sentiment = user_sentiment[user_sentiment['count'] >= 5]  # Chỉ users có >= 5 comments
            user_sentiment = user_sentiment.sort_values('mean', ascending=False)
            
            print(f"\nTop 5 users tích cực nhất (>= 5 comments):")
            for i, (user, row) in enumerate(user_sentiment.head(5).iterrows(), 1):
                print(f"  {i}. {user}: TB sentiment = {row['mean']:.2f} ({int(row['count'])} comments)")
            
            print(f"\nTop 5 users tiêu cực nhất (>= 5 comments):")
            for i, (user, row) in enumerate(user_sentiment.tail(5).iterrows(), 1):
                print(f"  {i}. {user}: TB sentiment = {row['mean']:.2f} ({int(row['count'])} comments)")
        
        return {
            'total_users': int(total_users),
            'avg_comments_per_user': float(total_comments/total_users),
            'top_users': user_comment_counts.head(10).to_dict()
        }
    
    def video_analysis(self):
        """Phân tích theo video"""
        print("\n" + "="*80)
        print("6. PHÂN TÍCH THEO VIDEO")
        print("="*80)
        
        if 'videoWebUrl' not in self.df.columns:
            print("⚠️  Không tìm thấy cột 'videoWebUrl'")
            return None
        
        # Đếm số video
        total_videos = self.df['videoWebUrl'].nunique()
        total_comments = len(self.df)
        
        print(f"\nTổng số video: {total_videos:,}")
        print(f"Tổng số comments: {total_comments:,}")
        print(f"Trung bình comments/video: {total_comments/total_videos:.2f}")
        
        # Top videos
        video_comment_counts = self.df['videoWebUrl'].value_counts()
        print(f"\nTop 10 videos có nhiều comments nhất:")
        for i, (video, count) in enumerate(video_comment_counts.head(10).items(), 1):
            print(f"  {i}. {count} comments")
            print(f"     URL: {video[:80]}...")
        
        # Phân tích sentiment theo video
        if 'trust' in self.df.columns:
            print(f"\nPhân tích sentiment theo video:")
            video_sentiment = self.df.groupby('videoWebUrl')['trust'].agg(['mean', 'count'])
            video_sentiment = video_sentiment[video_sentiment['count'] >= 10]  # Chỉ videos có >= 10 comments
            video_sentiment = video_sentiment.sort_values('mean', ascending=False)
            
            print(f"\nTop 5 videos tích cực nhất (>= 10 comments):")
            for i, (video, row) in enumerate(video_sentiment.head(5).iterrows(), 1):
                print(f"  {i}. TB sentiment = {row['mean']:.2f} ({int(row['count'])} comments)")
            
            print(f"\nTop 5 videos tiêu cực nhất (>= 10 comments):")
            for i, (video, row) in enumerate(video_sentiment.tail(5).iterrows(), 1):
                print(f"  {i}. TB sentiment = {row['mean']:.2f} ({int(row['count'])} comments)")
        
        return {
            'total_videos': int(total_videos),
            'avg_comments_per_video': float(total_comments/total_videos),
            'top_videos': video_comment_counts.head(10).to_dict()
        }
    
    def text_analysis(self):
        """Phân tích text"""
        print("\n" + "="*80)
        print("7. PHÂN TÍCH TEXT")
        print("="*80)
        
        if 'text' not in self.df.columns:
            print("⚠️  Không tìm thấy cột 'text'")
            return None
        
        # Thống kê độ dài text
        self.df['text_length'] = self.df['text'].astype(str).str.len()
        
        print(f"\nThống kê độ dài text:")
        print(f"  Trung bình: {self.df['text_length'].mean():.2f} ký tự")
        print(f"  Median: {self.df['text_length'].median():.2f} ký tự")
        print(f"  Min: {self.df['text_length'].min()} ký tự")
        print(f"  Max: {self.df['text_length'].max()} ký tự")
        
        # Top comments dài nhất
        print(f"\nTop 5 comments dài nhất:")
        longest = self.df.nlargest(5, 'text_length')
        for i, row in enumerate(longest.iterrows(), 1):
            text = str(row[1]['text'])[:100]
            print(f"  {i}. {len(str(row[1]['text']))} ký tự: {text}...")
        
        # Phân tích độ dài theo sentiment
        if 'trust' in self.df.columns:
            print(f"\nĐộ dài text theo sentiment:")
            sentiment_length = self.df.groupby('trust')['text_length'].agg(['mean', 'median'])
            for sentiment in [-1, 0, 1]:
                sentiment_label = {-1: 'Tiêu cực', 0: 'Trung tính', 1: 'Tích cực'}.get(sentiment, str(sentiment))
                if sentiment in sentiment_length.index:
                    mean_len = sentiment_length.loc[sentiment, 'mean']
                    median_len = sentiment_length.loc[sentiment, 'median']
                    print(f"  {sentiment_label}: TB={mean_len:.2f}, Median={median_len:.2f} ký tự")
        
        return {
            'text_length_stats': {
                'mean': float(self.df['text_length'].mean()),
                'median': float(self.df['text_length'].median()),
                'min': int(self.df['text_length'].min()),
                'max': int(self.df['text_length'].max())
            }
        }
    
    def correlation_analysis(self):
        """Phân tích tương quan"""
        print("\n" + "="*80)
        print("8. PHÂN TÍCH TƯƠNG QUAN")
        print("="*80)
        
        # Chọn các cột số
        numeric_cols = []
        if 'trust' in self.df.columns:
            numeric_cols.append('trust')
        if 'diggCount' in self.df.columns:
            numeric_cols.append('diggCount')
        if 'replyCommentTotal' in self.df.columns:
            numeric_cols.append('replyCommentTotal')
        if 'text_length' in self.df.columns:
            numeric_cols.append('text_length')
        
        if len(numeric_cols) < 2:
            print("⚠️  Không đủ cột số để phân tích tương quan")
            return None
        
        # Tính correlation
        corr_df = self.df[numeric_cols].corr()
        
        print("\nMa trận tương quan:")
        print(corr_df.to_string())
        
        # Vẽ heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_df, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                    square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Ma Trận Tương Quan', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/correlation_heatmap.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Đã lưu biểu đồ: {self.output_dir}/correlation_heatmap.png")
        plt.close()
        
        return corr_df.to_dict()
    
    def generate_report(self, results):
        """Tạo báo cáo tổng hợp"""
        print("\n" + "="*80)
        print("9. TẠO BÁO CÁO TỔNG HỢP")
        print("="*80)
        
        report = {
            'dataset_info': {
                'file': self.csv_file,
                'total_comments': len(self.df),
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            },
            'results': results
        }
        
        # Lưu report dạng JSON
        report_file = f'{self.output_dir}/analysis_report.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"✓ Đã lưu báo cáo: {report_file}")
        
        return report
    
    def run_full_analysis(self):
        """Chạy phân tích đầy đủ"""
        print("="*80)
        print("PHÂN TÍCH DỮ LIỆU COMMENTS TIKTOK")
        print("="*80)
        
        # Load data
        self.load_data()
        
        results = {}
        
        # 1. Basic info
        results['basic_info'] = self.basic_info()
        
        # 2. Sentiment analysis
        results['sentiment'] = self.sentiment_analysis()
        
        # 3. Engagement analysis
        results['engagement'] = self.engagement_analysis()
        
        # 4. Time analysis
        results['time'] = self.time_analysis()
        
        # 5. User analysis
        results['users'] = self.user_analysis()
        
        # 6. Video analysis
        results['videos'] = self.video_analysis()
        
        # 7. Text analysis
        results['text'] = self.text_analysis()
        
        # 8. Correlation analysis
        results['correlation'] = self.correlation_analysis()
        
        # 9. Generate report
        self.generate_report(results)
        
        print("\n" + "="*80)
        print("HOÀN THÀNH PHÂN TÍCH!")
        print("="*80)
        print(f"\nKết quả đã được lưu trong thư mục: {self.output_dir}/")
        print("  - sentiment_distribution.png")
        print("  - engagement_distribution.png")
        print("  - time_analysis.png")
        print("  - correlation_heatmap.png")
        print("  - analysis_report.json")
        
        return results


def main():
    """Hàm main"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Phân tích dữ liệu comments TikTok')
    parser.add_argument('--input', '-i',
                       default='dataset_tiktok-comments-637video-scraper_2026-01-15.csv',
                       help='File CSV đầu vào')
    
    args = parser.parse_args()
    
    # Kiểm tra file có tồn tại không
    if not os.path.exists(args.input):
        print(f"❌ Không tìm thấy file: {args.input}")
        print("Vui lòng kiểm tra đường dẫn file.")
        return
    
    # Chạy phân tích
    analyzer = TikTokDataAnalyzer(args.input)
    analyzer.run_full_analysis()


if __name__ == '__main__':
    main()
