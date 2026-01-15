"""
Web App để phân tích sentiment cho comments TikTok
UI được thiết kế theo phong cách Antigravity Kit - Experimental & Futuristic
"""

import streamlit as st
import pandas as pd
import numpy as np
from sentiment_analyzer import SentimentAnalyzer
import io
from datetime import datetime
import os

# Thử import plotly cho biểu đồ đẹp hơn
try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except:
    HAS_PLOTLY = False

# Cấu hình trang
st.set_page_config(
    page_title="Sentiment Analysis | TikTok Comments",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Antigravity Kit Experimental Style
st.markdown("""
<style>
    /* Import Modern Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600&display=swap');
    
    /* Reset & Base */
    * {
        box-sizing: border-box;
        margin: 0;
        padding: 0;
    }
    
    /* Color System - Antigravity Experimental */
    :root {
        /* Neon Gradients */
        --neon-cyan: #00f5ff;
        --neon-purple: #b026ff;
        --neon-blue: #0066ff;
        --neon-lime: #39ff14;
        --neon-pink: #ff00ff;
        
        --gradient-cyan: linear-gradient(135deg, #00f5ff 0%, #0066ff 100%);
        --gradient-purple: linear-gradient(135deg, #b026ff 0%, #0066ff 100%);
        --gradient-neon: linear-gradient(135deg, #00f5ff 0%, #b026ff 50%, #ff00ff 100%);
        --gradient-success: linear-gradient(135deg, #39ff14 0%, #00f5ff 100%);
        --gradient-danger: linear-gradient(135deg, #ff00ff 0%, #b026ff 100%);
        
        /* Deep Dark Backgrounds */
        --bg-deep: #000000;
        --bg-dark: #0a0a0f;
        --bg-card: rgba(15, 15, 25, 0.6);
        --bg-glass: rgba(255, 255, 255, 0.03);
        --bg-hover: rgba(255, 255, 255, 0.05);
        
        /* Text Colors */
        --text-primary: #ffffff;
        --text-secondary: #a0a0b0;
        --text-muted: #606070;
        
        /* Borders & Glows */
        --border-subtle: rgba(0, 245, 255, 0.1);
        --border-neon: rgba(0, 245, 255, 0.3);
        --glow-cyan: 0 0 20px rgba(0, 245, 255, 0.3), 0 0 40px rgba(0, 245, 255, 0.1);
        --glow-purple: 0 0 20px rgba(176, 38, 255, 0.3), 0 0 40px rgba(176, 38, 255, 0.1);
        --glow-soft: 0 8px 32px rgba(0, 0, 0, 0.4);
        
        /* Shadows */
        --shadow-float: 0 20px 60px rgba(0, 0, 0, 0.5);
        --shadow-glow: 0 0 30px rgba(0, 245, 255, 0.2);
    }
    
    /* Main Container - Deep Black */
    .main {
        padding: 0 !important;
        background: var(--bg-deep);
        color: var(--text-primary);
        min-height: 100vh;
        position: relative;
        overflow-x: hidden;
    }
    
    /* Remove default Streamlit padding */
    .main .block-container {
        padding-top: 0 !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        padding-bottom: 1rem !important;
        max-width: 100% !important;
    }
    
    /* Reduce main container padding */
    [data-testid="stMainBlockContainer"] {
        padding: 2rem 1rem !important;
        max-width: 1400px !important;
        margin: 0 auto !important;
    }
    
    /* Reduce vertical block padding */
    [data-testid="stVerticalBlock"] {
        padding: 0 !important;
    }
    
    /* Remove all margins from first element */
    .main > div:first-child {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }
    
    /* Animated Background Gradient */
    .main::before {
        content: '';
        position: fixed;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(0, 245, 255, 0.05) 0%, transparent 70%);
        animation: float 20s ease-in-out infinite;
        pointer-events: none;
        z-index: 0;
    }
    
    @keyframes float {
        0%, 100% { transform: translate(0, 0) rotate(0deg); }
        50% { transform: translate(30px, -30px) rotate(180deg); }
    }
    
    /* Navigation Bar - Top - Fixed at absolute top */
    .nav-bar {
        background: rgba(15, 15, 20, 0.8) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        padding: 1rem 2rem !important;
        margin: 1rem auto 0 auto !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 16px !important;
        display: flex !important;
        align-items: center;
        justify-content: space-between;
        position: fixed !important;
        top: 0 !important;
        left: 50% !important;
        transform: translateX(-50%) !important;
        z-index: 99999 !important;
        width: 70% !important;
        box-sizing: border-box !important;
        overflow: hidden !important;
        flex-wrap: nowrap;
        gap: 2rem;
        height: auto !important;
        min-height: auto !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3), 0 0 0 1px rgba(255, 255, 255, 0.05) inset !important;
    }
    
    /* Override ALL possible spacing sources */
    body, html {
        margin: 0 !important;
        padding: 0 !important;
    }
    
    .stApp, .stApp > div, [data-testid="stAppViewContainer"] {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }
    
    /* Add spacer for fixed nav - increased height */
    .main {
        padding-top: 80px !important;
    }
    
    /* Remove spacing from first markdown */
    .stMarkdown:first-of-type {
        margin-top: 0 !important;
        padding-top: 0 !important;
        margin-bottom: 0 !important;
        padding-bottom: 0 !important;
    }
    
    .stMarkdown:first-of-type > div {
        margin-top: 0 !important;
        padding-top: 0 !important;
        margin-bottom: 0 !important;
        padding-bottom: 0 !important;
    }
    
    /* Reduce size of nav-bar container */
    .stMarkdown:first-of-type [data-testid="stMarkdownContainer"] {
        margin: 0 !important;
        padding: 0 !important;
    }
    
    .stMarkdown:first-of-type .nav-bar {
        margin: 0 !important;
    }
    
    .nav-logo {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.125rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00f5ff 0%, #00ff88 50%, #ffd700 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        white-space: nowrap;
        flex-shrink: 0;
        line-height: 1.2;
    }
    
    .nav-logo span:first-child {
        font-size: 1rem;
        background: linear-gradient(135deg, #00f5ff 0%, #00ff88 25%, #ffd700 50%, #ff6b6b 75%, #00f5ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        filter: drop-shadow(0 0 8px rgba(0, 245, 255, 0.5));
    }
    
    .nav-logo span:last-child {
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    
    .nav-links {
        display: flex;
        gap: 2rem;
        align-items: center;
        flex-wrap: nowrap;
        justify-content: center;
        flex: 1;
        flex-shrink: 0;
    }
    
    .nav-link {
        font-family: 'Inter', sans-serif;
        font-size: 0.875rem;
        font-weight: 400;
        color: rgba(255, 255, 255, 0.7);
        text-decoration: none;
        transition: all 0.3s ease;
        letter-spacing: 0.2px;
        white-space: nowrap;
        flex-shrink: 0;
        padding: 0.5rem 0;
        line-height: 1.2;
        position: relative;
    }
    
    .nav-link:hover {
        color: rgba(255, 255, 255, 1);
    }
    
    .nav-link::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        width: 0;
        height: 2px;
        background: linear-gradient(90deg, #00f5ff, #00ff88);
        transition: width 0.3s ease;
    }
    
    .nav-link:hover::after {
        width: 100%;
    }
    
    .nav-github {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        text-decoration: none;
        color: rgba(255, 255, 255, 0.9);
        font-family: 'Inter', sans-serif;
        font-size: 0.875rem;
        font-weight: 500;
        transition: all 0.3s ease;
        flex-shrink: 0;
    }
    
    .nav-github:hover {
        background: rgba(255, 255, 255, 0.1);
        border-color: rgba(255, 255, 255, 0.2);
        color: rgba(255, 255, 255, 1);
        transform: translateY(-1px);
    }
    
    .nav-github svg {
        width: 18px;
        height: 18px;
        fill: currentColor;
    }
    
    /* Responsive navigation */
    @media (max-width: 1200px) {
        .nav-bar {
            padding: 0.75rem 1.25rem !important;
        }
        
        .nav-logo {
            font-size: 0.9375rem;
        }
        
        .nav-links {
            gap: 0.875rem;
        }
        
        .nav-link {
            font-size: 0.75rem;
        }
    }
    
    @media (max-width: 768px) {
        .nav-bar {
            padding: 0.75rem 1rem !important;
            gap: 0.5rem;
        }
        
        .nav-logo {
            font-size: 0.875rem;
        }
        
        .nav-logo span:last-child {
            display: none;
        }
        
        .nav-links {
            gap: 0.75rem;
        }
        
        .nav-link {
            font-size: 0.6875rem;
        }
    }
    
    @media (max-width: 640px) {
        .nav-links {
            gap: 0.5rem;
        }
        
        .nav-link {
            font-size: 0.625rem;
        }
        
        .nav-link:nth-child(3) {
            display: none;
        }
    }
    
    /* Hero Section - Main Content */
    .hero-section {
        padding: 4rem 2rem;
        max-width: 1200px;
        margin: 0 auto;
        text-align: center;
    }
    
    .hero-badge {
        display: inline-block;
        background: var(--bg-glass);
        backdrop-filter: blur(10px);
        border: 1px solid var(--border-subtle);
        border-radius: 20px;
        padding: 0.5rem 1rem;
        margin-bottom: 2rem;
        font-family: 'Inter', sans-serif;
        font-size: 0.875rem;
        color: var(--text-secondary);
    }
    
    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 4rem;
        font-weight: 600;
        letter-spacing: -2px;
        line-height: 1.1;
        margin: 0 0 1.5rem 0;
    }
    
    .hero-title .gradient-text {
        background: var(--gradient-neon);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .hero-description {
        font-family: 'Inter', sans-serif;
        font-size: 1.25rem;
        font-weight: 300;
        color: var(--text-secondary);
        max-width: 700px;
        margin: 0 auto 3rem auto;
        line-height: 1.6;
        letter-spacing: 0.3px;
    }
    
    /* Cards - Glass Morphism Floating */
    .stat-card {
        background: var(--bg-glass);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        padding: 2rem;
        border-radius: 12px;
        border: 1px solid var(--border-subtle);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        box-shadow: var(--shadow-float);
    }
    
    .stat-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: var(--gradient-cyan);
        opacity: 0;
        transition: opacity 0.4s ease;
    }
    
    .stat-card:hover {
        background: var(--bg-hover);
        border-color: var(--border-neon);
        transform: translateY(-8px) scale(1.02);
        box-shadow: var(--glow-cyan), var(--shadow-float);
    }
    
    .stat-card:hover::before {
        opacity: 1;
    }
    
    .stat-card .label {
        font-family: 'Inter', sans-serif;
        font-size: 0.6875rem;
        font-weight: 500;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 2px;
        margin: 0 0 1rem 0;
    }
    
    .stat-card .value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.5rem;
        font-weight: 600;
        color: var(--text-primary);
        margin: 0;
        line-height: 1.2;
        background: var(--gradient-cyan);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .stat-card .sub-value {
        font-family: 'Inter', sans-serif;
        font-size: 0.875rem;
        font-weight: 400;
        color: var(--text-secondary);
        margin-top: 0.5rem;
        letter-spacing: 0.5px;
    }
    
    /* Buttons - Neon Experimental */
    .stButton>button {
        font-family: 'Inter', sans-serif;
        background: transparent;
        color: var(--neon-cyan);
        border: 1px solid var(--border-neon);
        border-radius: 8px;
        padding: 0.875rem 2rem;
        font-weight: 500;
        font-size: 0.9375rem;
        letter-spacing: 0.5px;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: var(--glow-soft);
        position: relative;
        overflow: hidden;
        background: var(--bg-glass);
        backdrop-filter: blur(10px);
    }
    
    .stButton>button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: var(--gradient-cyan);
        opacity: 0.1;
        transition: left 0.5s ease;
    }
    
    .stButton>button:hover {
        color: var(--text-primary);
        border-color: var(--neon-cyan);
        transform: translateY(-2px);
        box-shadow: var(--glow-cyan), var(--shadow-float);
    }
    
    .stButton>button:hover::before {
        left: 100%;
    }
    
    /* Hide Tabs Buttons - Make them invisible but clickable */
    .stTabs [data-baseweb="tab-list"] {
        position: fixed !important;
        top: -100px !important;
        left: 0 !important;
        width: 100% !important;
        height: 1px !important;
        opacity: 0 !important;
        pointer-events: auto !important;
        z-index: 100000 !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        pointer-events: auto !important;
        opacity: 0 !important;
        height: 1px !important;
        overflow: hidden !important;
    }
    
    /* Show tab content */
    .stTabs [role="tabpanel"] {
        display: block !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-family: 'Inter', sans-serif;
        padding: 1rem 2rem;
        font-weight: 400;
        color: var(--text-secondary);
        border: none;
        background: transparent;
        transition: all 0.3s ease;
        font-size: 0.9375rem;
        letter-spacing: 0.3px;
    }
    
    .stTabs [aria-selected="true"] {
        color: var(--neon-cyan);
        border-bottom: 2px solid;
        border-image: var(--gradient-cyan);
        border-image-slice: 1;
        background: transparent;
    }
    
    /* Sidebar - Glass Experimental */
    [data-testid="stSidebar"] {
        background: var(--bg-glass);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-right: 1px solid var(--border-subtle);
    }
    
    [data-testid="stSidebar"] * {
        color: var(--text-primary);
    }
    
    /* File Uploader - Floating */
    .uploadedFile {
        border: 2px dashed var(--border-subtle);
        border-radius: 12px;
        padding: 3rem;
        background: var(--bg-glass);
        backdrop-filter: blur(10px);
        transition: all 0.4s ease;
        position: relative;
    }
    
    .uploadedFile:hover {
        border-color: var(--border-neon);
        background: var(--bg-hover);
        box-shadow: var(--glow-soft);
    }
    
    /* Progress Bar - Neon */
    .stProgress > div > div > div {
        background: var(--gradient-cyan);
        box-shadow: var(--glow-cyan);
    }
    
    /* Dataframe - Glass */
    .dataframe {
        border-radius: 12px;
        border: 1px solid var(--border-subtle);
        overflow: hidden;
        background: var(--bg-glass);
        backdrop-filter: blur(10px);
    }
    
    /* Inputs - Experimental */
    .stTextInput>div>div>input,
    .stSelectbox>div>div {
        background: var(--bg-glass);
        backdrop-filter: blur(10px);
        border: 1px solid var(--border-subtle);
        border-radius: 8px;
        color: var(--text-primary);
        font-family: 'Inter', sans-serif;
    }
    
    .stTextInput>div>div>input:focus,
    .stSelectbox>div>div:focus {
        border-color: var(--border-neon);
        box-shadow: var(--glow-cyan);
        outline: none;
    }
    
    /* Messages - Glass Experimental */
    .stSuccess {
        background: rgba(57, 255, 20, 0.1);
        backdrop-filter: blur(10px);
        border-left: 2px solid var(--neon-lime);
        border-radius: 8px;
        padding: 1rem;
        color: var(--text-primary);
        box-shadow: 0 0 20px rgba(57, 255, 20, 0.2);
    }
    
    .stError {
        background: rgba(255, 0, 255, 0.1);
        backdrop-filter: blur(10px);
        border-left: 2px solid var(--neon-pink);
        border-radius: 8px;
        padding: 1rem;
        color: var(--text-primary);
        box-shadow: 0 0 20px rgba(255, 0, 255, 0.2);
    }
    
    .stInfo {
        background: rgba(0, 245, 255, 0.1);
        backdrop-filter: blur(10px);
        border-left: 2px solid var(--neon-cyan);
        border-radius: 8px;
        padding: 1rem;
        color: var(--text-primary);
        box-shadow: 0 0 20px rgba(0, 245, 255, 0.2);
    }
    
    .stWarning {
        background: rgba(176, 38, 255, 0.1);
        backdrop-filter: blur(10px);
        border-left: 2px solid var(--neon-purple);
        border-radius: 8px;
        padding: 1rem;
        color: var(--text-primary);
        box-shadow: 0 0 20px rgba(176, 38, 255, 0.2);
    }
    
    /* Spacing - Generous */
    .element-container {
        margin-bottom: 2rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Force remove all top spacing - Strong CSS */
    .stApp > header {
        display: none !important;
        height: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    
    .stApp {
        padding-top: 0 !important;
        margin-top: 0 !important;
    }
    
    .stApp > div:first-child {
        padding-top: 0 !important;
        margin-top: 0 !important;
    }
    
    /* Remove all Streamlit default spacing */
    [data-testid="stAppViewContainer"] {
        padding-top: 0 !important;
        margin-top: 0 !important;
    }
    
    [data-testid="stHeader"] {
        display: none !important;
        height: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    
    /* Remove spacing from markdown container */
    .element-container:first-child {
        margin-top: 0 !important;
        padding-top: 0 !important;
        margin-bottom: 0 !important;
        padding-bottom: 0 !important;
    }
    
    /* Remove spacing from first two element containers (CSS and JS) */
    .element-container:nth-child(1),
    .element-container:nth-child(2) {
        margin: 0 !important;
        padding: 0 !important;
        height: auto !important;
    }
    
    /* Remove spacing from stMarkdown in first containers */
    .element-container:first-child .stMarkdown,
    .element-container:nth-child(2) .stMarkdown {
        margin: 0 !important;
        padding: 0 !important;
    }
    
    .element-container:first-child .stMarkdown > div,
    .element-container:nth-child(2) .stMarkdown > div {
        margin: 0 !important;
        padding: 0 !important;
    }
    
    /* Remove spacing from stElementContainer */
    [data-testid="stElementContainer"]:first-child,
    [data-testid="stElementContainer"]:nth-child(2) {
        margin: 0 !important;
        padding: 0 !important;
        height: auto !important;
    }
    
    /* Navigation bar - absolutely no margin */
    .nav-bar {
        margin-top: 0 !important;
        margin-bottom: 0 !important;
        padding-top: 0.875rem !important;
        padding-bottom: 0.875rem !important;
        position: relative !important;
    }
    
    /* Force first markdown to have no spacing */
    .stMarkdown:first-child {
        margin-top: 0 !important;
        padding-top: 0 !important;
        margin-bottom: 0 !important;
        padding-bottom: 0 !important;
    }
    
    .stMarkdown:first-child > div {
        margin-top: 0 !important;
        padding-top: 0 !important;
        margin-bottom: 0 !important;
        padding-bottom: 0 !important;
    }
    
    /* Remove spacing from stMarkdownContainer */
    [data-testid="stMarkdownContainer"]:first-child,
    [data-testid="stMarkdownContainer"]:nth-child(1),
    [data-testid="stMarkdownContainer"]:nth-child(2) {
        margin: 0 !important;
        padding: 0 !important;
    }
    
    /* Custom Scrollbar - Neon */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bg-dark);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--gradient-cyan);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--neon-cyan);
        box-shadow: var(--glow-cyan);
    }
    
    /* Section Divider - Subtle */
    .section-divider {
        height: 1px;
        background: var(--border-subtle);
        margin: 3rem 0;
        position: relative;
    }
    
    .section-divider::before {
        content: '';
        position: absolute;
        left: 0;
        top: 0;
        width: 100px;
        height: 1px;
        background: var(--gradient-cyan);
        opacity: 0.5;
    }
    
    /* Floating Animation */
    @keyframes float-up {
        0% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0); }
    }
    
    .float-animation {
        animation: float-up 3s ease-in-out infinite;
    }
</style>
""", unsafe_allow_html=True)

# JavaScript to inject nav-bar directly into body and remove all spacing
st.markdown("""
<script>
    (function() {
        // Inject nav-bar directly into body immediately
        function injectNavBar() {
            // Remove existing nav-bar if any
            const existing = document.querySelector('.nav-bar');
            if (existing) existing.remove();
            
            // Create nav-bar element
            const navBar = document.createElement('div');
            navBar.className = 'nav-bar';
            navBar.innerHTML = `
                <div class="nav-logo">
                    <span>▲</span>
                    <span>Sentiment Analysis</span>
                </div>
                <div class="nav-links">
                    <a href="#tab-upload-results" class="nav-link">Upload & Results</a>
                    <a href="#tab-analytics" class="nav-link">Analytics</a>
                    <a href="#tab-settings" class="nav-link">Settings</a>
                </div>
                <a href="https://github.com" target="_blank" class="nav-github">
                    <svg viewBox="0 0 24 24" fill="currentColor">
                        <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
                    </svg>
                    <span>GitHub</span>
                </a>
            `;
            
            // Insert at the very beginning of body
            document.body.insertBefore(navBar, document.body.firstChild);
            
            // Force styles
            navBar.style.position = 'fixed';
            navBar.style.top = '0';
            navBar.style.left = '0';
            navBar.style.right = '0';
            navBar.style.margin = '0';
            navBar.style.padding = '1rem 2rem';
            navBar.style.zIndex = '99999';
            navBar.style.width = '100vw';
        }
        
        function removeAllSpacing() {
            // Remove Streamlit header
            const header = document.querySelector('[data-testid="stHeader"]');
            if (header) header.remove();
            
            // Remove all padding/margin
            const selectors = [
                '[data-testid="stAppViewContainer"]',
                '.stApp',
                '.main',
                '.main .block-container',
                '.element-container:first-child',
                '.element-container:nth-child(2)',
                '.stMarkdown:first-child',
                '[data-testid="stElementContainer"]:first-child',
                '[data-testid="stElementContainer"]:nth-child(2)',
                '[data-testid="stMarkdownContainer"]:first-child',
                '[data-testid="stMarkdownContainer"]:nth-child(2)'
            ];
            
            selectors.forEach(selector => {
                document.querySelectorAll(selector).forEach(el => {
                    el.style.paddingTop = '0';
                    el.style.marginTop = '0';
                    el.style.paddingBottom = '0';
                    el.style.marginBottom = '0';
                    el.style.padding = '0';
                    el.style.margin = '0';
                    el.style.height = 'auto';
                });
            });
            
            // Specifically target first two element containers
            const firstContainers = document.querySelectorAll('.element-container:nth-child(-n+2)');
            firstContainers.forEach(el => {
                el.style.margin = '0';
                el.style.padding = '0';
                el.style.height = 'auto';
                const markdown = el.querySelector('.stMarkdown');
                if (markdown) {
                    markdown.style.margin = '0';
                    markdown.style.padding = '0';
                    const markdownDiv = markdown.querySelector('div');
                    if (markdownDiv) {
                        markdownDiv.style.margin = '0';
                        markdownDiv.style.padding = '0';
                    }
                }
            });
            
            // Remove spacing from body/html
            document.body.style.margin = '0';
            document.body.style.padding = '0';
            document.documentElement.style.margin = '0';
            document.documentElement.style.padding = '0';
            
            // Add padding to main for fixed nav
            const main = document.querySelector('.main');
            const navBar = document.querySelector('.nav-bar');
            if (main && navBar) {
                main.style.paddingTop = navBar.offsetHeight + 'px';
            }
            
            // Remove spacing from tabs
            const tabs = document.querySelector('.stTabs');
            if (tabs) {
                tabs.style.marginTop = '0';
                tabs.style.paddingTop = '0';
                const tabsFirstChild = tabs.querySelector('div:first-child');
                if (tabsFirstChild) {
                    tabsFirstChild.style.marginTop = '0';
                    tabsFirstChild.style.paddingTop = '0';
                }
            }
            
            // Remove spacer div if exists
            const spacer = document.querySelector('div[style*="height: 60px"]');
            if (spacer) {
                spacer.remove();
            }
        }
        
        // Handle navigation link clicks - Use query params
        function setupNavLinks() {
            // Remove old event listeners by cloning
            const navLinks = document.querySelectorAll('.nav-link');
            navLinks.forEach(link => {
                // Clone to remove old listeners
                const newLink = link.cloneNode(true);
                link.parentNode.replaceChild(newLink, link);
                
                newLink.addEventListener('click', function(e) {
                    e.preventDefault();
                    e.stopPropagation();
                    
                    const href = this.getAttribute('href');
                    const targetId = href.replace('#', '');
                    
                    if (targetId) {
                        // Map ID to tab name and index
                        const idToTabMap = {
                            'tab-upload-results': { name: 'Upload & Results', index: 1 },
                            'tab-analytics': { name: 'Analytics', index: 2 },
                            'tab-settings': { name: 'Settings', index: 3 }
                        };
                        
                        const tabInfo = idToTabMap[targetId];
                        
                        if (tabInfo) {
                            // First, click the tab button to switch tab
                            setTimeout(() => {
                                let tabButtons = document.querySelectorAll('[data-baseweb="tab"]');
                                if (!tabButtons || tabButtons.length === 0) {
                                    tabButtons = document.querySelectorAll('button[role="tab"]');
                                }
                                
                                if (tabButtons && tabButtons.length > tabInfo.index && tabButtons[tabInfo.index]) {
                                    tabButtons[tabInfo.index].click();
                                }
                            }, 50);
                            
                            // Then scroll to the target element
                            setTimeout(() => {
                                const targetElement = document.getElementById(targetId);
                                if (targetElement) {
                                    const navBar = document.querySelector('.nav-bar');
                                    const offset = navBar ? navBar.offsetHeight + 30 : 100;
                                    
                                    // Get element position
                                    const rect = targetElement.getBoundingClientRect();
                                    const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
                                    const targetPosition = rect.top + scrollTop - offset;
                                    
                                    window.scrollTo({
                                        top: targetPosition,
                                        behavior: 'smooth'
                                    });
                                } else {
                                    // Retry after a bit more time
                                    setTimeout(() => {
                                        const targetElement = document.getElementById(targetId);
                                        if (targetElement) {
                                            const navBar = document.querySelector('.nav-bar');
                                            const offset = navBar ? navBar.offsetHeight + 30 : 100;
                                            const rect = targetElement.getBoundingClientRect();
                                            const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
                                            const targetPosition = rect.top + scrollTop - offset;
                                            
                                            window.scrollTo({
                                                top: targetPosition,
                                                behavior: 'smooth'
                                            });
                                        }
                                    }, 500);
                                }
                            }, 300);
                        }
                    }
                });
            });
            
            // Handle logo click - go to Introduction tab
            const navLogo = document.querySelector('.nav-logo');
            if (navLogo) {
                navLogo.style.cursor = 'pointer';
                // Remove old listener
                const newLogo = navLogo.cloneNode(true);
                navLogo.parentNode.replaceChild(newLogo, navLogo);
                
                newLogo.addEventListener('click', function(e) {
                    e.preventDefault();
                    e.stopPropagation();
                    
                    // Click Introduction tab
                    setTimeout(() => {
                        let tabButtons = document.querySelectorAll('[data-baseweb="tab"]');
                        if (!tabButtons || tabButtons.length === 0) {
                            tabButtons = document.querySelectorAll('button[role="tab"]');
                        }
                        if (tabButtons && tabButtons[0]) {
                            tabButtons[0].click();
                        }
                    }, 50);
                    
                    // Scroll to Introduction section
                    setTimeout(() => {
                        const targetElement = document.getElementById('tab-introduction');
                        if (targetElement) {
                            const navBar = document.querySelector('.nav-bar');
                            const offset = navBar ? navBar.offsetHeight + 20 : 100;
                            const elementPosition = targetElement.getBoundingClientRect().top;
                            const offsetPosition = elementPosition + window.pageYOffset - offset;
                            
                            window.scrollTo({
                                top: offsetPosition,
                                behavior: 'smooth'
                            });
                        } else {
                            window.scrollTo({ top: 0, behavior: 'smooth' });
                        }
                    }, 300);
                });
            }
        }
        
        // Run immediately
        injectNavBar();
        removeAllSpacing();
        setupNavLinks();
        
        // Run on DOM ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', function() {
                injectNavBar();
                removeAllSpacing();
                setupNavLinks();
            });
        }
        
        // Run multiple times
        setTimeout(() => { injectNavBar(); removeAllSpacing(); setupNavLinks(); }, 10);
        setTimeout(() => { injectNavBar(); removeAllSpacing(); setupNavLinks(); }, 50);
        setTimeout(() => { injectNavBar(); removeAllSpacing(); setupNavLinks(); }, 100);
        setTimeout(() => { injectNavBar(); removeAllSpacing(); setupNavLinks(); }, 300);
        
        // Watch for changes
        const observer = new MutationObserver(function() {
            injectNavBar();
            removeAllSpacing();
            setupNavLinks();
        });
        observer.observe(document.body, { childList: true, subtree: true });
    })();
</script>
""", unsafe_allow_html=True)

# Navigation Bar - Top
st.markdown("""
<div class="nav-bar">
    <div class="nav-logo">
        <span>▲</span>
        <span>Sentiment Analysis</span>
    </div>
    <div class="nav-links">
        <a href="#tab-upload-results" class="nav-link">Upload & Results</a>
        <a href="#tab-analytics" class="nav-link">Analytics</a>
        <a href="#tab-settings" class="nav-link">Settings</a>
    </div>
    <a href="https://github.com" target="_blank" class="nav-github">
        <svg viewBox="0 0 24 24" fill="currentColor">
            <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
        </svg>
        <span>GitHub</span>
    </a>
</div>
""", unsafe_allow_html=True)

# Sidebar - Minimal (only help)
with st.sidebar:
    with st.expander("Hướng Dẫn"):
        st.markdown("""
        1. Upload file CSV có cột **text**
        2. Nhấn nút phân tích
        3. Xem kết quả & tải file
        
        **Sentiment Values:**
        - **1** = Tích cực
        - **0** = Trung tính
        - **-1** = Tiêu cực
        """)

# Session State
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = None
if 'results_df' not in st.session_state:
    st.session_state.results_df = None
if 'model_choice' not in st.session_state:
    st.session_state.model_choice = "nlptown/bert-base-multilingual-uncased-sentiment"
if 'batch_size' not in st.session_state:
    st.session_state.batch_size = 32
if 'text_column' not in st.session_state:
    st.session_state.text_column = 'text'
if 'trust_column' not in st.session_state:
    st.session_state.trust_column = 'trust'
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = 'Introduction'

# Get tab from query params or session state
query_params = st.query_params
if 'tab' in query_params:
    st.session_state.current_tab = query_params['tab']

# Tabs - Hidden, controlled by navigation bar
tab_intro, tab1, tab2, tab3 = st.tabs(["Introduction", "Upload & Results", "Analytics", "Settings"])

# Set active tab based on session state
tab_map = {
    'Introduction': 0,
    'Upload & Results': 1,
    'Analytics': 2,
    'Settings': 3
}

# Tab Introduction - Default page
with tab_intro:
    st.markdown('<div id="tab-introduction"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; padding: 3rem 2rem;">
        <div style="font-size: 3rem; font-weight: 700; margin-bottom: 1rem;">
            <span style="background: linear-gradient(135deg, #00f5ff 0%, #00ff88 50%, #ffd700 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                ⚡ AI Sentiment Analysis Tool
            </span>
        </div>
        <div style="font-size: 1.25rem; color: rgba(255, 255, 255, 0.7); margin-bottom: 2rem; max-width: 800px; margin-left: auto; margin-right: auto;">
            Phân tích cảm xúc cho Comments TikTok
        </div>
        <div style="font-size: 1rem; color: rgba(255, 255, 255, 0.6); line-height: 1.8; max-width: 700px; margin-left: auto; margin-right: auto; margin-bottom: 3rem;">
            Một công cụ phân tích sentiment toàn diện sử dụng AI để đánh giá cảm xúc của comments TikTok. 
            Nhanh chóng, chính xác và dễ sử dụng với các model transformer hiện đại.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div style="height: 2rem;"></div>', unsafe_allow_html=True)
    
    # How to use
    st.markdown("""
    <div style="max-width: 800px; margin: 0 auto;">
        <h2 style="text-align: center; margin-bottom: 2rem; font-size: 2rem;">Cách sử dụng</h2>
        <div style="background: rgba(255, 255, 255, 0.05); border-radius: 12px; padding: 2rem; border: 1px solid rgba(255, 255, 255, 0.1);">
            <div style="margin-bottom: 1.5rem;">
                <div style="display: flex; align-items: start; gap: 1rem;">
                    <div style="background: linear-gradient(135deg, #00f5ff, #00ff88); border-radius: 50%; width: 2rem; height: 2rem; display: flex; align-items: center; justify-content: center; font-weight: 700; flex-shrink: 0;">1</div>
                    <div>
                        <div style="font-weight: 600; margin-bottom: 0.5rem;">Upload file CSV</div>
                        <div style="color: rgba(255, 255, 255, 0.6);">Chọn file CSV có cột 'text' chứa comments TikTok</div>
                    </div>
                </div>
            </div>
            <div style="margin-bottom: 1.5rem;">
                <div style="display: flex; align-items: start; gap: 1rem;">
                    <div style="background: linear-gradient(135deg, #00f5ff, #00ff88); border-radius: 50%; width: 2rem; height: 2rem; display: flex; align-items: center; justify-content: center; font-weight: 700; flex-shrink: 0;">2</div>
                    <div>
                        <div style="font-weight: 600; margin-bottom: 0.5rem;">Phân tích sentiment</div>
                        <div style="color: rgba(255, 255, 255, 0.6);">Nhấn nút "Phân Tích Sentiment" và chờ kết quả</div>
                    </div>
                </div>
            </div>
            <div>
                <div style="display: flex; align-items: start; gap: 1rem;">
                    <div style="background: linear-gradient(135deg, #00f5ff, #00ff88); border-radius: 50%; width: 2rem; height: 2rem; display: flex; align-items: center; justify-content: center; font-weight: 700; flex-shrink: 0;">3</div>
                    <div>
                        <div style="font-weight: 600; margin-bottom: 0.5rem;">Xem kết quả & Tải file</div>
                        <div style="color: rgba(255, 255, 255, 0.6);">Xem thống kê, biểu đồ và tải file CSV đã được phân tích</div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div style="height: 3rem;"></div>', unsafe_allow_html=True)
    
    # Sentiment Values
    st.markdown("""
    <div style="max-width: 600px; margin: 0 auto; text-align: center;">
        <h3 style="margin-bottom: 1.5rem; font-size: 1.5rem;">Giá trị Sentiment</h3>
        <div style="display: flex; gap: 1rem; justify-content: center; flex-wrap: wrap;">
            <div style="background: rgba(0, 255, 136, 0.1); border: 1px solid rgba(0, 255, 136, 0.3); border-radius: 8px; padding: 1rem 1.5rem;">
                <div style="font-size: 1.5rem; font-weight: 700; color: #00ff88;">1</div>
                <div style="color: rgba(255, 255, 255, 0.7); margin-top: 0.5rem;">Tích cực</div>
            </div>
            <div style="background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 8px; padding: 1rem 1.5rem;">
                <div style="font-size: 1.5rem; font-weight: 700; color: rgba(255, 255, 255, 0.5);">0</div>
                <div style="color: rgba(255, 255, 255, 0.7); margin-top: 0.5rem;">Trung tính</div>
            </div>
            <div style="background: rgba(255, 107, 107, 0.1); border: 1px solid rgba(255, 107, 107, 0.3); border-radius: 8px; padding: 1rem 1.5rem;">
                <div style="font-size: 1.5rem; font-weight: 700; color: #ff6b6b;">-1</div>
                <div style="color: rgba(255, 255, 255, 0.7); margin-top: 0.5rem;">Tiêu cực</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Tab 1: Upload & Results (2 cột 3/7)
with tab1:
    st.markdown('<div id="tab-upload-results"></div>', unsafe_allow_html=True)
    col_upload, col_results = st.columns([3, 7])
    
    # Cột Upload (30%)
    with col_upload:
        st.markdown("### Upload")
        uploaded_file = st.file_uploader(
            "Chọn file CSV",
            type=['csv'],
            help="File CSV phải có cột 'text' chứa comments"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                # File Info - Floating Cards
                st.markdown(f"""
                <div class="stat-card">
                    <div class="label">Tổng số dòng</div>
                    <div class="value">{len(df):,}</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="stat-card" style="margin-top: 1rem;">
                    <div class="label">Số cột</div>
                    <div class="value">{len(df.columns)}</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="stat-card" style="margin-top: 1rem;">
                    <div class="label">Kích thước</div>
                    <div class="value">{uploaded_file.size / 1024:.1f}</div>
                    <div class="sub-value">KB</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Validation
                if 'text' not in df.columns:
                    st.error("File CSV không có cột 'text'!")
                else:
                    text_count = df['text'].notna().sum()
                    st.success(f"File hợp lệ! Có **{text_count:,}** comments")
                    
                    # Preview
                    with st.expander("Preview dữ liệu"):
                        st.dataframe(
                            df[['text']].head(5) if 'text' in df.columns else df.head(5),
                            use_container_width=True,
                            height=200
                        )
                    
                    # Analysis Button
                    st.markdown("---")
                    if st.button("Phân Tích Sentiment", type="primary", use_container_width=True):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        try:
                            status_text.text("Đang khởi tạo model...")
                            analyzer = SentimentAnalyzer(model_name=st.session_state.model_choice)
                            st.session_state.analyzer = analyzer
                            progress_bar.progress(20)
                            
                            status_text.text("Đang phân tích sentiment...")
                            
                            def update_progress(current, total):
                                progress = 20 + int((current / total) * 70)
                                progress_bar.progress(progress)
                                status_text.text(f"Đang phân tích: {current}/{total} batches...")
                            
                            analyzer.progress_callback = update_progress
                            
                            results_df = analyzer.process_csv_dataframe(
                                df.copy(),
                                text_column=st.session_state.text_column,
                                trust_column=st.session_state.trust_column,
                                batch_size=st.session_state.batch_size
                            )
                            
                            progress_bar.progress(100)
                            status_text.text("Hoàn thành!")
                            
                            st.session_state.results_df = results_df
                            st.success("Phân tích hoàn thành!")
                            st.balloons()
                            
                        except Exception as e:
                            st.error(f"Lỗi: {str(e)}")
                            st.exception(e)
                        finally:
                            progress_bar.empty()
                            status_text.empty()
                            
            except Exception as e:
                st.error(f"Lỗi khi đọc file: {str(e)}")
                st.exception(e)
    
    # Cột Results (70%)
    with col_results:
        st.markdown("### Results")
        if st.session_state.results_df is not None:
            df = st.session_state.results_df
            
            if 'trust' in df.columns:
                # Stats - Floating Cards
                total = len(df)
                positive = (df['trust'] == 1).sum()
                neutral = (df['trust'] == 0).sum()
                negative = (df['trust'] == -1).sum()
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown(f"""
                    <div class="stat-card">
                        <div class="label">Tổng số</div>
                        <div class="value">{total:,}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    pct = positive/total*100
                    st.markdown(f"""
                    <div class="stat-card" style="border-top: 2px solid var(--neon-lime);">
                        <div class="label">Tích cực</div>
                        <div class="value" style="background: var(--gradient-success); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{positive:,}</div>
                        <div class="sub-value">{pct:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col3:
                    pct = neutral/total*100
                    st.markdown(f"""
                    <div class="stat-card" style="border-top: 2px solid var(--text-muted);">
                        <div class="label">Trung tính</div>
                        <div class="value" style="color: var(--text-muted);">{neutral:,}</div>
                        <div class="sub-value">{pct:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col4:
                    pct = negative/total*100
                    st.markdown(f"""
                    <div class="stat-card" style="border-top: 2px solid var(--neon-pink);">
                        <div class="label">Tiêu cực</div>
                        <div class="value" style="background: var(--gradient-danger); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{negative:,}</div>
                        <div class="sub-value">{pct:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                
                # Data Table
                display_cols = ['text', 'trust']
                other_cols = [col for col in df.columns if col not in ['text', 'trust']]
                if other_cols:
                    display_cols = ['text'] + other_cols[:3] + ['trust']
                
                st.dataframe(
                    df[display_cols] if all(col in df.columns for col in display_cols) else df,
                    use_container_width=True,
                    height=400
                )
                
                # Download
                st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                csv = df.to_csv(index=False, encoding='utf-8-sig')
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"sentiment_results_{timestamp}.csv"
                
                st.download_button(
                    label="Tải File CSV",
                    data=csv,
                    file_name=filename,
                    mime="text/csv",
                    use_container_width=True
                )
                
                # Filter
                st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                with col1:
                    search_text = st.text_input("Tìm kiếm", placeholder="Nhập từ khóa...")
                with col2:
                    filter_sentiment = st.selectbox(
                        "Lọc sentiment",
                        ["Tất cả", "Tích cực (1)", "Trung tính (0)", "Tiêu cực (-1)"]
                    )
                
                if search_text or filter_sentiment != "Tất cả":
                    filtered_df = df.copy()
                    
                    if search_text:
                        filtered_df = filtered_df[filtered_df['text'].str.contains(search_text, case=False, na=False)]
                    
                    if filter_sentiment == "Tích cực (1)":
                        filtered_df = filtered_df[filtered_df['trust'] == 1]
                    elif filter_sentiment == "Trung tính (0)":
                        filtered_df = filtered_df[filtered_df['trust'] == 0]
                    elif filter_sentiment == "Tiêu cực (-1)":
                        filtered_df = filtered_df[filtered_df['trust'] == -1]
                    
                    st.caption(f"Kết quả: {len(filtered_df)} dòng")
                    st.dataframe(
                        filtered_df[display_cols] if all(col in filtered_df.columns for col in display_cols) else filtered_df,
                        use_container_width=True,
                        height=300
                    )
            else:
                st.warning("Chưa có cột 'trust' trong kết quả")
        else:
            st.info("Vui lòng upload file và chạy phân tích")

# Tab 2: Analytics
with tab2:
    st.markdown('<div id="tab-analytics"></div>', unsafe_allow_html=True)
    if st.session_state.results_df is not None:
        df = st.session_state.results_df
        
        if 'trust' in df.columns:
            # Stats
            total = len(df)
            positive = (df['trust'] == 1).sum()
            neutral = (df['trust'] == 0).sum()
            negative = (df['trust'] == -1).sum()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Tổng số", total)
            with col2:
                st.metric("Tích cực", positive, f"{positive/total*100:.1f}%")
            with col3:
                st.metric("Trung tính", neutral, f"{neutral/total*100:.1f}%")
            with col4:
                st.metric("Tiêu cực", negative, f"{negative/total*100:.1f}%")
            
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Phân Bố Sentiment**")
                if HAS_PLOTLY:
                    fig_bar = go.Figure(data=[
                        go.Bar(
                            x=['Tích cực', 'Trung tính', 'Tiêu cực'],
                            y=[positive, neutral, negative],
                            marker_color=['#39ff14', '#a0a0b0', '#ff00ff'],
                            text=[positive, neutral, negative],
                            textposition='auto',
                        )
                    ])
                    fig_bar.update_layout(
                        height=400,
                        template='plotly_dark',
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        showlegend=False,
                        margin=dict(l=0, r=0, t=0, b=0),
                        font=dict(color='#ffffff', size=12)
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
                else:
                    chart_data = pd.DataFrame({
                        'Sentiment': ['Tích cực', 'Trung tính', 'Tiêu cực'],
                        'Số lượng': [positive, neutral, negative]
                    })
                    st.bar_chart(chart_data.set_index('Sentiment'), height=400)
            
            with col2:
                st.markdown("**Tỷ Lệ Sentiment**")
                if HAS_PLOTLY:
                    fig_pie = px.pie(
                        values=[positive, neutral, negative],
                        names=['Tích cực', 'Trung tính', 'Tiêu cực'],
                        color_discrete_map={
                            'Tích cực': '#39ff14',
                            'Trung tính': '#a0a0b0',
                            'Tiêu cực': '#ff00ff'
                        },
                        hole=0.4
                    )
                    fig_pie.update_traces(
                        textposition='inside',
                        textinfo='percent+label',
                        textfont=dict(color='#ffffff', size=12)
                    )
                    fig_pie.update_layout(
                        height=400,
                        template='plotly_dark',
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        margin=dict(l=0, r=0, t=0, b=0),
                        font=dict(color='#ffffff', size=12)
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                else:
                    st.bar_chart(pd.DataFrame({
                        'Sentiment': ['Tích cực', 'Trung tính', 'Tiêu cực'],
                        'Count': [positive, neutral, negative]
                    }).set_index('Sentiment'))
            
            # Examples
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Ví dụ Comments Tích Cực**")
                positive_examples = df[df['trust'] == 1]['text'].head(5).tolist()
                if positive_examples:
                    for i, text in enumerate(positive_examples, 1):
                        st.markdown(f"""
                        <div style='background: rgba(57, 255, 20, 0.1); backdrop-filter: blur(10px); padding: 1rem; border-radius: 8px; border-left: 2px solid #39ff14; margin-bottom: 0.75rem; box-shadow: 0 0 20px rgba(57, 255, 20, 0.2);'>
                            <p style='margin: 0; font-size: 0.875rem; color: var(--text-primary); font-family: Inter, sans-serif;'>{text[:120]}{'...' if len(text) > 120 else ''}</p>
                        </div>
                        """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("**Ví dụ Comments Tiêu Cực**")
                negative_examples = df[df['trust'] == -1]['text'].head(5).tolist()
                if negative_examples:
                    for i, text in enumerate(negative_examples, 1):
                        st.markdown(f"""
                        <div style='background: rgba(255, 0, 255, 0.1); backdrop-filter: blur(10px); padding: 1rem; border-radius: 8px; border-left: 2px solid #ff00ff; margin-bottom: 0.75rem; box-shadow: 0 0 20px rgba(255, 0, 255, 0.2);'>
                            <p style='margin: 0; font-size: 0.875rem; color: var(--text-primary); font-family: Inter, sans-serif;'>{text[:120]}{'...' if len(text) > 120 else ''}</p>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.warning("Chưa có cột 'trust' trong kết quả")
    else:
        st.info("Vui lòng upload file và chạy phân tích ở tab **Upload & Results**")

# Tab 3: Settings
with tab3:
    st.markdown('<div id="tab-settings"></div>', unsafe_allow_html=True)
    st.markdown("### Cài Đặt Cột Dữ Liệu")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.session_state.text_column = st.text_input(
            "Tên cột chứa text",
            value=st.session_state.text_column,
            help="Tên cột trong CSV chứa comments cần phân tích"
        )
    
    with col2:
        st.session_state.trust_column = st.text_input(
            "Tên cột trust (kết quả)",
            value=st.session_state.trust_column,
            help="Tên cột sẽ chứa kết quả sentiment (1, 0, -1)"
        )
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    st.markdown("### Cài Đặt Model")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.session_state.model_choice = st.selectbox(
            "Model Phân Tích",
            ["nlptown/bert-base-multilingual-uncased-sentiment", 
             "cardiffnlp/twitter-roberta-base-sentiment-latest"],
            index=0 if st.session_state.model_choice == "nlptown/bert-base-multilingual-uncased-sentiment" else 1,
            help="Model đa ngôn ngữ hỗ trợ tốt hơn cho tiếng Việt"
        )
        
        st.info("""
        **nlptown/bert-base-multilingual-uncased-sentiment:**
        - Chính xác hơn, hỗ trợ đa ngôn ngữ tốt
        - Chậm hơn, tốn RAM hơn
        
        **cardiffnlp/twitter-roberta-base-sentiment-latest:**
        - Nhanh hơn, nhẹ hơn
        - Hỗ trợ đa ngôn ngữ cơ bản
        """)
    
    with col2:
        st.session_state.batch_size = st.slider(
            "Batch Size",
            min_value=8,
            max_value=64,
            value=st.session_state.batch_size,
            step=8,
            help="Lớn hơn = nhanh hơn nhưng tốn RAM hơn"
        )
        
        st.caption(f"Đang sử dụng: **{st.session_state.batch_size}** items/batch")
        
        if st.session_state.batch_size >= 48:
            st.warning("⚠️ Batch size lớn có thể gây lỗi out of memory. Nên dùng GPU.")
        elif st.session_state.batch_size <= 16:
            st.info("💡 Batch size nhỏ an toàn nhưng chậm hơn.")
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    st.markdown("### Thông Tin Model")
    
    if st.session_state.analyzer is not None:
        st.success("✅ Model đã được khởi tạo")
        st.info(f"**Model hiện tại:** {st.session_state.model_choice}")
        st.info(f"**Device:** {'GPU' if st.session_state.analyzer.device >= 0 else 'CPU'}")
    else:
        st.warning("⚠️ Model chưa được khởi tạo. Model sẽ được tải khi bạn chạy phân tích lần đầu.")
        st.info("""
        **Lưu ý:**
        - Lần đầu chạy sẽ tải model (~500MB), mất vài phút
        - Model sẽ được cache, các lần sau sẽ nhanh hơn
        - Nếu có GPU, tool sẽ tự động sử dụng để tăng tốc
        """)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    st.markdown("### Tùy Chọn Nâng Cao")
    
    with st.expander("Thông Tin Kỹ Thuật"):
        st.markdown("""
        **Sentiment Analysis Model:**
        - Sử dụng transformer models từ HuggingFace
        - Hỗ trợ đa ngôn ngữ (tiếng Việt, tiếng Anh)
        - Kết hợp với keyword và emoji detection
        
        **Sentiment Values:**
        - **1** = Tích cực (Positive)
        - **0** = Trung tính (Neutral)
        - **-1** = Tiêu cực (Negative)
        
        **Xử lý:**
        - Tự động bỏ qua các dòng đã có trust score
        - Chỉ phân tích các dòng chưa có giá trị
        - Hỗ trợ resume nếu bị gián đoạn
        """)
    
    with st.expander("Tối Ưu Hiệu Suất"):
        st.markdown("""
        **Để tăng tốc độ:**
        1. Tăng batch size (nếu có đủ RAM)
        2. Sử dụng GPU (nếu có)
        3. Chọn model nhẹ hơn (twitter-roberta)
        
        **Để tăng độ chính xác:**
        1. Sử dụng model đa ngôn ngữ (bert-multilingual)
        2. Giảm batch size để xử lý kỹ hơn
        3. Kiểm tra và làm sạch dữ liệu trước khi phân tích
        """)
    
    with st.expander("Xử Lý Lỗi"):
        st.markdown("""
        **Lỗi thường gặp:**
        - **Out of Memory:** Giảm batch size
        - **Model không tải được:** Kiểm tra kết nối internet
        - **File không đọc được:** Kiểm tra encoding (UTF-8)
        - **Cột không tìm thấy:** Kiểm tra tên cột trong Settings
        
        **Giải pháp:**
        - Refresh trang và thử lại
        - Kiểm tra log trong console
        - Thử với file nhỏ hơn trước
        """)

# Footer - Minimal Experimental
st.markdown("""
<div style='text-align: center; padding: 3rem 0; color: var(--text-secondary); border-top: 1px solid var(--border-subtle); margin-top: 4rem; position: relative;'>
    <p style='margin: 0; font-size: 0.875rem; font-weight: 400; font-family: Inter, sans-serif; letter-spacing: 1px;'>Sentiment Analysis Tool</p>
    <p style='margin: 0.5rem 0 0 0; font-size: 0.75rem; color: var(--text-muted); font-family: Inter, sans-serif;'>Experimental AI • Future Design</p>
</div>
""", unsafe_allow_html=True)
