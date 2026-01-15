"""
Script để chạy web app
Sử dụng: python src/run_web.py
"""

import subprocess
import sys
import os

def main():
    """Chạy Streamlit web app"""
    # Đảm bảo đang ở đúng thư mục
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Chạy streamlit
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])

if __name__ == '__main__':
    main()
