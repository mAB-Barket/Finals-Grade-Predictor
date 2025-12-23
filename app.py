"""
AI Student Performance Predictor
================================
A beautiful, modern Streamlit application for predicting student grades
using Machine Learning with relative grading system.

Developed for Air University
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ============================================================
# PAGE CONFIG & THEME
# ============================================================
st.set_page_config(
    page_title="AI Grade Predictor | Air University",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern Premium Color Scheme - Deep Purple & Teal with Gold accents
st.markdown("""
<style>
/* Import Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Poppins:wght@400;500;600;700&display=swap');

/* Root Variables - Premium Deep Purple & Teal Theme */
:root {
    --bg-primary: #0a0a1a;
    --bg-secondary: #12122a;
    --bg-card: #1a1a3e;
    --bg-card-hover: #252550;
    --accent-primary: #6366f1;      /* Indigo */
    --accent-secondary: #14b8a6;    /* Teal */
    --accent-gold: #f59e0b;         /* Gold */
    --accent-success: #22c55e;      /* Green */
    --accent-danger: #ef4444;       /* Red */
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --border-color: rgba(99, 102, 241, 0.2);
    --glow-primary: rgba(99, 102, 241, 0.4);
    --glow-secondary: rgba(20, 184, 166, 0.4);
}

/* Global Styles */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

.stApp {
    background: linear-gradient(135deg, var(--bg-primary) 0%, var(--bg-secondary) 50%, #0d1f2d 100%);
}

/* Hide Streamlit Branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Main Container */
.main .block-container {
    padding: 2rem 3rem;
    max-width: 1400px;
}

/* Sidebar Styling */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #12122a 0%, #1a1a3e 100%);
    border-right: 1px solid var(--border-color);
}

section[data-testid="stSidebar"] > div {
    padding: 1.5rem 1rem;
}

/* Custom Header */
.main-header {
    text-align: center;
    padding: 2rem 0;
    margin-bottom: 2rem;
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(20, 184, 166, 0.1) 100%);
    border-radius: 20px;
    border: 1px solid var(--border-color);
    backdrop-filter: blur(10px);
}

.main-title {
    font-family: 'Poppins', sans-serif;
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #6366f1 0%, #14b8a6 50%, #f59e0b 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.5rem;
    letter-spacing: -0.02em;
}

.main-subtitle {
    color: var(--text-secondary);
    font-size: 1.1rem;
    font-weight: 400;
}

.university-badge {
    display: inline-block;
    background: linear-gradient(135deg, var(--accent-gold) 0%, #d97706 100%);
    color: #0a0a1a;
    padding: 6px 16px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-top: 1rem;
    letter-spacing: 0.5px;
}

/* Card Styles */
.metric-card {
    background: linear-gradient(145deg, var(--bg-card) 0%, rgba(26, 26, 62, 0.8) 100%);
    border: 1px solid var(--border-color);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    transition: all 0.3s ease;
    backdrop-filter: blur(10px);
}

.metric-card:hover {
    transform: translateY(-2px);
    border-color: var(--accent-primary);
    box-shadow: 0 8px 32px var(--glow-primary);
}

.metric-label {
    color: var(--text-secondary);
    font-size: 0.85rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 0.5rem;
}

.metric-value {
    font-size: 2.5rem;
    font-weight: 700;
    background: linear-gradient(135deg, var(--accent-primary) 0%, var(--accent-secondary) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* Status Badges */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 12px 24px;
    border-radius: 50px;
    font-weight: 700;
    font-size: 1rem;
    letter-spacing: 0.5px;
    animation: pulse-glow 2s ease-in-out infinite;
}

.status-pass {
    background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);
    color: #052e16;
    box-shadow: 0 4px 20px rgba(34, 197, 94, 0.4);
}

.status-fail {
    background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
    color: #450a0a;
    box-shadow: 0 4px 20px rgba(239, 68, 68, 0.4);
}

@keyframes pulse-glow {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.9; transform: scale(1.02); }
}

/* Grade Display */
.grade-display {
    text-align: center;
    padding: 2rem;
    background: linear-gradient(145deg, var(--bg-card) 0%, rgba(26, 26, 62, 0.6) 100%);
    border-radius: 20px;
    border: 1px solid var(--border-color);
}

.predicted-score {
    font-size: 4rem;
    font-weight: 800;
    line-height: 1;
    margin-bottom: 0.5rem;
}

.score-pass {
    background: linear-gradient(135deg, #22c55e 0%, #14b8a6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.score-fail {
    background: linear-gradient(135deg, #ef4444 0%, #f97316 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.grade-letter {
    font-size: 3rem;
    font-weight: 800;
    padding: 0.5rem 1.5rem;
    border-radius: 12px;
    display: inline-block;
    margin-top: 1rem;
}

.grade-a { background: linear-gradient(135deg, #22c55e 0%, #10b981 100%); color: #052e16; }
.grade-b { background: linear-gradient(135deg, #3b82f6 0%, #6366f1 100%); color: #0c1f3f; }
.grade-c { background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); color: #422006; }
.grade-d { background: linear-gradient(135deg, #f97316 0%, #ea580c 100%); color: #431407; }
.grade-f { background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); color: #450a0a; }

/* Section Headers */
.section-header {
    color: var(--text-primary);
    font-family: 'Poppins', sans-serif;
    font-size: 1.3rem;
    font-weight: 600;
    margin: 1.5rem 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid var(--border-color);
    display: flex;
    align-items: center;
    gap: 10px;
}

/* Recommendation Cards */
.recommendation-card {
    background: linear-gradient(145deg, rgba(99, 102, 241, 0.1) 0%, rgba(20, 184, 166, 0.1) 100%);
    border: 1px solid var(--accent-primary);
    border-radius: 16px;
    padding: 1.5rem;
    margin: 1rem 0;
}

.recommendation-title {
    color: var(--accent-secondary);
    font-weight: 600;
    font-size: 1.1rem;
    margin-bottom: 0.5rem;
}

/* Weightage Info Cards */
.weightage-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin: 1rem 0;
}

.weightage-item {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
    transition: all 0.3s ease;
}

.weightage-item:hover {
    border-color: var(--accent-secondary);
    box-shadow: 0 4px 16px var(--glow-secondary);
}

.weightage-label {
    color: var(--text-muted);
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.weightage-value {
    color: var(--accent-secondary);
    font-size: 1.5rem;
    font-weight: 700;
}

/* Sidebar Sections */
.sidebar-section {
    background: rgba(99, 102, 241, 0.05);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 1rem;
}

.sidebar-title {
    color: var(--accent-primary);
    font-weight: 600;
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Attendance Status Styles */
.attendance-container {
    background: linear-gradient(145deg, var(--bg-card) 0%, rgba(26, 26, 62, 0.8) 100%);
    border-radius: 16px;
    padding: 1.25rem;
    border: 1px solid var(--border-color);
    margin-bottom: 1rem;
    transition: all 0.3s ease;
}

.attendance-container.eligible {
    border-color: var(--accent-success);
    box-shadow: 0 4px 20px rgba(34, 197, 94, 0.15);
}

.attendance-container.ineligible {
    border-color: var(--accent-danger);
    box-shadow: 0 4px 20px rgba(239, 68, 68, 0.15);
}

.attendance-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.75rem;
}

.attendance-title {
    color: var(--text-primary);
    font-weight: 600;
    font-size: 0.95rem;
    display: flex;
    align-items: center;
    gap: 8px;
}

.attendance-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.3px;
}

.attendance-badge.eligible {
    background: linear-gradient(135deg, rgba(34, 197, 94, 0.2) 0%, rgba(16, 163, 127, 0.2) 100%);
    color: var(--accent-success);
    border: 1px solid rgba(34, 197, 94, 0.3);
}

.attendance-badge.ineligible {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(249, 115, 22, 0.2) 100%);
    color: var(--accent-danger);
    border: 1px solid rgba(239, 68, 68, 0.3);
}

.attendance-value {
    font-size: 2rem;
    font-weight: 700;
    text-align: center;
    margin: 0.5rem 0;
}

.attendance-value.eligible {
    background: linear-gradient(135deg, var(--accent-success) 0%, var(--accent-secondary) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.attendance-value.ineligible {
    background: linear-gradient(135deg, var(--accent-danger) 0%, #f97316 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.attendance-progress {
    background: rgba(100, 116, 139, 0.2);
    border-radius: 10px;
    height: 8px;
    overflow: hidden;
    margin: 0.75rem 0;
}

.attendance-progress-bar {
    height: 100%;
    border-radius: 10px;
    transition: width 0.4s ease;
}

.attendance-progress-bar.eligible {
    background: linear-gradient(90deg, var(--accent-success) 0%, var(--accent-secondary) 100%);
}

.attendance-progress-bar.ineligible {
    background: linear-gradient(90deg, var(--accent-danger) 0%, #f97316 100%);
}

.attendance-info {
    font-size: 0.8rem;
    color: var(--text-muted);
    text-align: center;
    margin-top: 0.5rem;
}

.attendance-warning {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(249, 115, 22, 0.1) 100%);
    border: 1px solid rgba(239, 68, 68, 0.3);
    border-radius: 12px;
    padding: 1rem;
    margin-top: 1rem;
    text-align: center;
}

.attendance-warning-title {
    color: var(--accent-danger);
    font-weight: 600;
    font-size: 0.95rem;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
}

.attendance-warning-text {
    color: var(--text-secondary);
    font-size: 0.85rem;
    line-height: 1.5;
}

/* Ineligibility Banner */
.ineligibility-banner {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(249, 115, 22, 0.1) 100%);
    border: 2px solid var(--accent-danger);
    border-radius: 20px;
    padding: 2rem;
    margin: 2rem 0;
    text-align: center;
    animation: pulse-border 2s ease-in-out infinite;
}

@keyframes pulse-border {
    0%, 100% { border-color: rgba(239, 68, 68, 0.8); }
    50% { border-color: rgba(239, 68, 68, 0.4); }
}

.ineligibility-icon {
    font-size: 4rem;
    margin-bottom: 1rem;
}

.ineligibility-title {
    color: var(--accent-danger);
    font-family: 'Poppins', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    margin-bottom: 0.75rem;
}

.ineligibility-subtitle {
    color: var(--text-secondary);
    font-size: 1.1rem;
    margin-bottom: 1rem;
}

.ineligibility-attendance {
    display: inline-block;
    background: rgba(239, 68, 68, 0.2);
    color: var(--accent-danger);
    padding: 0.5rem 1.5rem;
    border-radius: 30px;
    font-size: 1.5rem;
    font-weight: 700;
    margin: 1rem 0;
}

.ineligibility-note {
    color: var(--text-muted);
    font-size: 0.9rem;
    margin-top: 1rem;
    max-width: 500px;
    margin-left: auto;
    margin-right: auto;
}

/* Input Styling */
.stSlider > div[data-baseweb="slider"] > div {
    background: transparent !important;
}

.stSlider > div > div > div[role="slider"] {
    background-color: var(--accent-primary) !important;
    border-color: var(--accent-primary) !important;
}

/* Modern Delete Button */
.delete-btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    border-radius: 8px;
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(249, 115, 22, 0.1) 100%);
    border: 1px solid rgba(239, 68, 68, 0.3);
    color: var(--accent-danger);
    font-size: 0.9rem;
    cursor: pointer;
    transition: all 0.3s ease;
}

.delete-btn:hover {
    background: linear-gradient(135deg, var(--accent-danger) 0%, #dc2626 100%);
    color: white;
    border-color: var(--accent-danger);
    transform: scale(1.1);
    box-shadow: 0 4px 15px rgba(239, 68, 68, 0.4);
}

.quiz-item, .assignment-item {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: rgba(99, 102, 241, 0.05);
    border: 1px solid var(--border-color);
    border-radius: 10px;
    padding: 0.6rem 0.8rem;
    margin-bottom: 0.5rem;
    transition: all 0.3s ease;
}

.quiz-item:hover, .assignment-item:hover {
    border-color: var(--accent-primary);
    background: rgba(99, 102, 241, 0.1);
}

.item-info {
    display: flex;
    flex-direction: column;
}

.item-name {
    color: var(--text-primary);
    font-weight: 600;
    font-size: 0.9rem;
}

.item-score {
    color: var(--accent-secondary);
    font-size: 0.8rem;
}

/* Result Popup Animation */
.result-popup {
    animation: popIn 0.6s cubic-bezier(0.68, -0.55, 0.265, 1.55) forwards;
}

@keyframes popIn {
    0% {
        opacity: 0;
        transform: scale(0.5) translateY(30px);
    }
    50% {
        transform: scale(1.05) translateY(-10px);
    }
    100% {
        opacity: 1;
        transform: scale(1) translateY(0);
    }
}

/* Success Animation - Celebration */
.celebration-popup {
    position: relative;
    animation: celebrate 0.8s ease forwards;
}

@keyframes celebrate {
    0% {
        opacity: 0;
        transform: scale(0.3);
    }
    50% {
        transform: scale(1.1);
    }
    70% {
        transform: scale(0.95);
    }
    100% {
        opacity: 1;
        transform: scale(1);
    }
}

/* Confetti Effect */
.confetti-container {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
    z-index: 9999;
    overflow: hidden;
}

.confetti {
    position: absolute;
    width: 10px;
    height: 10px;
    animation: confetti-fall 3s ease-out forwards;
}

@keyframes confetti-fall {
    0% {
        opacity: 1;
        transform: translateY(-100%) rotate(0deg);
    }
    100% {
        opacity: 0;
        transform: translateY(100vh) rotate(720deg);
    }
}

/* Fail Animation - Shake */
.fail-popup {
    animation: shake 0.6s ease-in-out;
}

@keyframes shake {
    0%, 100% { transform: translateX(0); }
    10%, 30%, 50%, 70%, 90% { transform: translateX(-5px); }
    20%, 40%, 60%, 80% { transform: translateX(5px); }
}

/* Glow Effect for Grade Letter */
.grade-glow {
    animation: gradeGlow 2s ease-in-out infinite;
}

@keyframes gradeGlow {
    0%, 100% {
        box-shadow: 0 0 20px currentColor;
    }
    50% {
        box-shadow: 0 0 40px currentColor, 0 0 60px currentColor;
    }
}

/* Pulse animation for status badge */
.status-pulse {
    animation: statusPulse 1.5s ease-in-out infinite;
}

@keyframes statusPulse {
    0%, 100% {
        transform: scale(1);
        box-shadow: 0 4px 20px currentColor;
    }
    50% {
        transform: scale(1.05);
        box-shadow: 0 6px 30px currentColor;
    }
}

/* Score counter animation */
.score-counter {
    animation: countUp 1s ease-out forwards;
}

@keyframes countUp {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* Flying emoji animation */
.flying-emoji {
    position: fixed;
    font-size: 3rem;
    animation: flyUp 2s ease-out forwards;
    pointer-events: none;
    z-index: 10000;
}

@keyframes flyUp {
    0% {
        opacity: 1;
        transform: translateY(0) scale(1);
    }
    100% {
        opacity: 0;
        transform: translateY(-200px) scale(1.5);
    }
}

/* Button Styling */
.stButton > button {
    background: linear-gradient(135deg, var(--accent-primary) 0%, #4f46e5 100%);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.75rem 2rem;
    font-weight: 600;
    font-size: 1rem;
    transition: all 0.3s ease;
    width: 100%;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #4f46e5 0%, #4338ca 100%);
    box-shadow: 0 8px 24px var(--glow-primary);
    transform: translateY(-2px);
}

/* Primary Action Button */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--accent-secondary) 0%, #0d9488 100%);
}

.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #0d9488 0%, #0f766e 100%);
    box-shadow: 0 8px 24px var(--glow-secondary);
}

/* Footer */
.footer {
    text-align: center;
    padding: 2rem 0;
    margin-top: 3rem;
    border-top: 1px solid var(--border-color);
    color: var(--text-muted);
    font-size: 0.85rem;
}

.footer-logo {
    font-weight: 700;
    background: linear-gradient(135deg, var(--accent-primary) 0%, var(--accent-secondary) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* Plotly Chart Container */
.plotly-chart {
    background: var(--bg-card);
    border-radius: 16px;
    padding: 1rem;
    border: 1px solid var(--border-color);
}

/* Animations */
@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.animate-in {
    animation: fadeInUp 0.6s ease forwards;
}

/* Logo Container */
.logo-container {
    text-align: center;
    padding: 1rem 0;
    margin-bottom: 1rem;
}

.logo-icon {
    font-size: 3rem;
    display: block;
    margin-bottom: 0.5rem;
}

.logo-text {
    font-family: 'Poppins', sans-serif;
    font-weight: 700;
    font-size: 1.1rem;
    background: linear-gradient(135deg, var(--accent-primary) 0%, var(--accent-secondary) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_model():
    try:
        return joblib.load('grade_predictor_model.pkl')
    except Exception as e:
        return None

model = load_model()

# ============================================================
# GRADING CONFIGURATION
# ============================================================
WEIGHTS = {
    'finals': 0.40,
    'mids': 0.25,
    'quizzes': 0.15,
    'assignments': 0.10,
    'presentation': 0.05,
    'internal': 0.05
}

GPA_SCALE = {
    'A': 4.0, 'A-': 3.7,
    'B+': 3.3, 'B': 3.0, 'B-': 2.7,
    'C+': 2.3, 'C': 2.0, 'C-': 1.7,
    'D+': 1.3, 'D': 1.0,
    'F': 0.0
}

# Class statistics from training data (for relative grading)
CLASS_MEAN = 72.47
CLASS_STD = 9.51

def get_relative_grade(total_score):
    """Assign grade based on relative position in class distribution."""
    z_score = (total_score - CLASS_MEAN) / CLASS_STD
    
    if z_score >= 1.5: return 'A'
    elif z_score >= 1.0: return 'A-'
    elif z_score >= 0.5: return 'B+'
    elif z_score >= 0.0: return 'B'
    elif z_score >= -0.5: return 'B-'
    elif z_score >= -1.0: return 'C+'
    elif z_score >= -1.5: return 'C'
    elif z_score >= -2.0: return 'C-'
    elif z_score >= -2.25: return 'D+'
    elif z_score >= -2.5: return 'D'
    else: return 'F'

def get_grade_class(grade):
    """Get CSS class for grade styling."""
    if grade.startswith('A'): return 'grade-a'
    elif grade.startswith('B'): return 'grade-b'
    elif grade.startswith('C'): return 'grade-c'
    elif grade.startswith('D'): return 'grade-d'
    else: return 'grade-f'

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    # Logo
    st.markdown("""
    <div class="logo-container">
        <span class="logo-icon">🎓</span>
        <span class="logo-text">Grade Predictor</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Student Details Section - Attendance with Professional Styling
    is_eligible = True  # Will be set based on attendance
    
    attendance = st.slider("📊 Attendance (%)", 0, 100, 85, help="Minimum 75% required for exam eligibility")
    is_eligible = attendance >= 75
    eligibility_class = "eligible" if is_eligible else "ineligible"
    eligibility_text = "ELIGIBLE" if is_eligible else "NOT ELIGIBLE"
    eligibility_icon = "✓" if is_eligible else "✗"
    
    st.markdown(f"""
    <div class="attendance-container {eligibility_class}">
        <div class="attendance-header">
            <span class="attendance-title">📊 Attendance Status</span>
            <span class="attendance-badge {eligibility_class}">{eligibility_icon} {eligibility_text}</span>
        </div>
        <div class="attendance-value {eligibility_class}">{attendance}%</div>
        <div class="attendance-progress">
            <div class="attendance-progress-bar {eligibility_class}" style="width: {attendance}%;"></div>
        </div>
        <div class="attendance-info">Minimum 75% required for exam eligibility</div>
    </div>
    """, unsafe_allow_html=True)
    
    if not is_eligible:
        st.markdown(f"""
        <div class="attendance-warning">
            <div class="attendance-warning-title">⚠️ Exam Eligibility Warning</div>
            <div class="attendance-warning-text">
                Your attendance is below the required 75% threshold.<br>
                You need <strong>{75 - attendance}% more</strong> attendance to become eligible.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    hours_studied = st.slider("Study Hours/Week", 0, 40, 8)
    prev_gpa = st.number_input("Previous GPA", 0.0, 4.0, 3.0, step=0.1)
    
    st.markdown("---")
    
    # Initialize session states
    if 'quizzes' not in st.session_state:
        st.session_state['quizzes'] = {}
        st.session_state['quiz_counter'] = 0
    if 'assignments' not in st.session_state:
        st.session_state['assignments'] = {}
        st.session_state['assign_counter'] = 0
    
    # Quizzes Section
    st.markdown("""
    <div class="sidebar-title">
        📝 Quizzes (15%)
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("Add Quiz", expanded=False):
        q_total = st.number_input("Total Marks", 1, 100, 10, key="q_total")
        q_obtained = st.number_input("Obtained", 0, q_total, int(q_total * 0.7), key="q_obtained")
        if st.button("➕ Add Quiz", key="add_quiz"):
            st.session_state['quiz_counter'] += 1
            qid = f"Q{st.session_state['quiz_counter']}"
            st.session_state['quizzes'][qid] = {'total': q_total, 'obtained': q_obtained}
            st.rerun()
    
    # Display quizzes with modern styling
    if st.session_state['quizzes']:
        # Convert to list to handle deletion during iteration safely
        current_quizzes = list(st.session_state['quizzes'].items())
        
        for index, (qid, q) in enumerate(current_quizzes):
            # QID might be 'Q1', 'Q3' etc. we want to display as 'Quiz 1', 'Quiz 2' dynamically or just keep Q1..Qn consistent
            # But requirement is "if delete quiz 1 then quiz 2 will have to be quiz 1".
            # So actual stored keys should probably be regenerated or we just display them with new numbers.
            # To strictly follow "quiz 2 will have to be quiz 1", we must rebuild the state on delete.
            
            pct = (q['obtained'] / q['total']) * 100
            pct_color = "#22c55e" if pct >= 70 else "#f59e0b" if pct >= 50 else "#ef4444"
            
            quiz_col1, quiz_col2 = st.columns([4, 1])
            with quiz_col1:
                st.markdown(f'''
                <div class="quiz-item">
                    <div class="item-info">
                        <span class="item-name">{qid}</span>
                        <span class="item-score" style="color: {pct_color};">{q['obtained']}/{q['total']} ({pct:.0f}%)</span>
                    </div>
                </div>
                ''', unsafe_allow_html=True)
            with quiz_col2:
                # Modern 'X' button
                if st.button("✕", key=f"del_q_{qid}", help=f"Remove {qid}"):
                    # Logic to renumber
                    # 1. Pop the deleted item
                    del st.session_state['quizzes'][qid]
                    
                    # 2. Rebuild dict with new keys Q1, Q2...
                    old_values = list(st.session_state['quizzes'].values())
                    st.session_state['quizzes'] = {}
                    st.session_state['quiz_counter'] = 0
                    
                    for val in old_values:
                        st.session_state['quiz_counter'] += 1
                        new_key = f"Q{st.session_state['quiz_counter']}"
                        st.session_state['quizzes'][new_key] = val
                        
                    st.rerun()
    
    # Calculate quiz average
    if st.session_state['quizzes']:
        quiz_pcts = [(q['obtained'] / q['total']) * 100 for q in st.session_state['quizzes'].values()]
        quiz_avg = sum(quiz_pcts) / len(quiz_pcts)
    else:
        quiz_avg = 0.0
    
    st.metric("Quiz Average", f"{quiz_avg:.1f}%")
    
    st.markdown("---")
    
    # Assignments Section
    st.markdown("""
    <div class="sidebar-title">
        📋 Assignments (10%)
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("Add Assignment", expanded=False):
        a_total = st.number_input("Total Marks", 1, 100, 10, key="a_total")
        a_obtained = st.number_input("Obtained", 0, a_total, int(a_total * 0.75), key="a_obtained")
        if st.button("➕ Add Assignment", key="add_assign"):
            st.session_state['assign_counter'] += 1
            aid = f"A{st.session_state['assign_counter']}"
            st.session_state['assignments'][aid] = {'total': a_total, 'obtained': a_obtained}
            st.rerun()
    
    # Display assignments with modern styling
    if st.session_state['assignments']:
        current_assignments = list(st.session_state['assignments'].items())
        
        for index, (aid, a) in enumerate(current_assignments):
            pct = (a['obtained'] / a['total']) * 100
            pct_color = "#22c55e" if pct >= 70 else "#f59e0b" if pct >= 50 else "#ef4444"
            
            assign_col1, assign_col2 = st.columns([4, 1])
            with assign_col1:
                st.markdown(f'''
                <div class="assignment-item">
                    <div class="item-info">
                        <span class="item-name">{aid}</span>
                        <span class="item-score" style="color: {pct_color};">{a['obtained']}/{a['total']} ({pct:.0f}%)</span>
                    </div>
                </div>
                ''', unsafe_allow_html=True)
            with assign_col2:
                if st.button("✕", key=f"del_a_{aid}", help=f"Remove {aid}"):
                    # Renumbering logic
                    del st.session_state['assignments'][aid]
                    
                    old_values = list(st.session_state['assignments'].values())
                    st.session_state['assignments'] = {}
                    st.session_state['assign_counter'] = 0
                    
                    for val in old_values:
                        st.session_state['assign_counter'] += 1
                        new_key = f"A{st.session_state['assign_counter']}"
                        st.session_state['assignments'][new_key] = val
                        
                    st.rerun()
    
    # Calculate assignment average
    if st.session_state['assignments']:
        assign_pcts = [(a['obtained'] / a['total']) * 100 for a in st.session_state['assignments'].values()]
        assignment_avg = sum(assign_pcts) / len(assign_pcts)
    else:
        assignment_avg = 0.0
    
    st.metric("Assignment Average", f"{assignment_avg:.1f}%")
    
    st.markdown("---")
    
    # Midterm Section
    st.markdown("""
    <div class="sidebar-title">
        📚 Midterm (25%)
    </div>
    """, unsafe_allow_html=True)
    
    # mid_total is fixed at 50 as per requirement
    mid_total = 50
    st.caption("Total Marks: 50")
    mid_obtained = st.slider("Midterm Obtained", 0, mid_total, int(mid_total * 0.7), key="mid_obtained")
    midterm = (mid_obtained / mid_total) * 100
    st.metric("Midterm Score", f"{midterm:.1f}%")
    
    st.markdown("---")
    
    # Presentation & Internal
    st.markdown("""
    <div class="sidebar-title">
        🎤 Other Components (10%)
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("Presentation & Internal", expanded=False):
        pres_score = st.slider("Presentation (%)", 0, 100, 70, key="pres")
        internal_score = st.slider("Internal Marks (%)", 0, 100, 70, key="internal")
    
    presentation_marks = st.session_state.get('pres', 70)
    internal_marks = st.session_state.get('internal', 70)

# ============================================================
# MAIN CONTENT
# ============================================================

# Header
st.markdown("""
<div class="main-header animate-in">
    <h1 class="main-title">🎓 AI Grade Predictor</h1>
    <p class="main-subtitle">Powered by Machine Learning with Relative Grading System</p>
    <span class="university-badge">AIR UNIVERSITY</span>
</div>
""", unsafe_allow_html=True)

# Weightage Info
st.markdown('<div class="section-header">📊 Grading Weightages</div>', unsafe_allow_html=True)

col1, col2, col3, col4, col5, col6 = st.columns(6)
with col1:
    st.markdown("""
    <div class="weightage-item">
        <div class="weightage-value">40%</div>
        <div class="weightage-label">Finals</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="weightage-item">
        <div class="weightage-value">25%</div>
        <div class="weightage-label">Midterm</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div class="weightage-item">
        <div class="weightage-value">15%</div>
        <div class="weightage-label">Quizzes</div>
    </div>
    """, unsafe_allow_html=True)
with col4:
    st.markdown("""
    <div class="weightage-item">
        <div class="weightage-value">10%</div>
        <div class="weightage-label">Assignments</div>
    </div>
    """, unsafe_allow_html=True)
with col5:
    st.markdown("""
    <div class="weightage-item">
        <div class="weightage-value">5%</div>
        <div class="weightage-label">Presentation</div>
    </div>
    """, unsafe_allow_html=True)
with col6:
    st.markdown("""
    <div class="weightage-item">
        <div class="weightage-value">5%</div>
        <div class="weightage-label">Internal</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Predict Button
col_btn = st.columns([1, 2, 1])
with col_btn[1]:
    predict_clicked = st.button("🔮 Predict My Grade", use_container_width=True, type="primary")

# ============================================================
# PREDICTION LOGIC
# ============================================================
if predict_clicked:
    # Check attendance eligibility first
    if attendance < 75:
        st.error("🚫 NOT ELIGIBLE: Attendance is below 75%")
        st.markdown("""
        <div class="ineligibility-banner">
            <div class="ineligibility-icon">🚫</div>
            <div class="ineligibility-title">Not Eligible for Examination</div>
            <div class="ineligibility-subtitle">Your attendance does not meet the minimum requirement</div>
            <div class="ineligibility-attendance">""" + str(attendance) + """%</div>
            <div class="ineligibility-note">
                <strong>University Policy:</strong> Students must maintain a minimum of 75% attendance 
                to be eligible to sit for final examinations. Please contact your academic advisor 
                for guidance on improving your attendance.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Show what attendance is needed
        needed = 75 - attendance
        st.markdown(f"""
        <div class="recommendation-card">
            <div class="recommendation-title">📈 How to Become Eligible</div>
            <p style="color: var(--text-secondary);">
                You need to increase your attendance by <strong style="color: var(--accent-danger);">{needed}%</strong> 
                to reach the minimum 75% requirement.
            </p>
            <p style="color: var(--text-muted); font-size: 0.85rem; margin-top: 0.5rem;">
                Contact your course instructor or academic office for attendance rectification options.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.toast("🚫 You are not eligible for the exam!", icon="🚫")
        
    elif model is None:
        st.error("❌ Model not found. Please run `python train_model.py` first.")
    else:
        # Build feature vector
        features = np.array([[
            attendance,
            hours_studied,
            prev_gpa,
            quiz_avg,
            midterm,
            assignment_avg,
            presentation_marks,
            internal_marks
        ]], dtype=float)
        
        # Handle feature count mismatch
        expected = getattr(model, 'n_features_in_', 8)
        if features.shape[1] > expected:
            features = features[:, :expected]
        elif features.shape[1] < expected:
            pad = np.zeros((1, expected - features.shape[1]))
            features = np.hstack([features, pad])
        
        # Predict
        predicted_final = model.predict(features)[0]
        predicted_final = max(0, min(100, predicted_final))  # Clamp to 0-100
        
        # Calculate total weighted score
        total_weighted = (
            predicted_final * WEIGHTS['finals'] +
            midterm * WEIGHTS['mids'] +
            quiz_avg * WEIGHTS['quizzes'] +
            assignment_avg * WEIGHTS['assignments'] +
            presentation_marks * WEIGHTS['presentation'] +
            internal_marks * WEIGHTS['internal']
        )
        
        # Get grade
        grade = get_relative_grade(total_weighted)
        gpa = GPA_SCALE[grade]
        grade_class = get_grade_class(grade)
        
        # Determine pass/fail
        passing = grade != 'F'
        status_class = "status-pass" if passing else "status-fail"
        score_class = "score-pass" if passing else "score-fail"
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown('<div class="section-header">📈 Prediction Results</div>', unsafe_allow_html=True)
            
            # Determine status text
            status_text = "✅ PASSING" if passing else "❌ FAILING"
            popup_class = "celebration-popup" if passing else "fail-popup"
            
            # Show balloons/snow and toast based on result
            if passing:
                st.balloons()
                st.toast(f"🎉 Predicted Grade: {grade} (GPA: {gpa})", icon="🎉")
            else:
                st.toast(f"⚠️ At Risk: Predicted Failing Grade", icon="⚠️")

            # Minified HTML to prevent Markdown code block issues
            result_html = f"""<div class="grade-display result-popup {popup_class}">
<div class="metric-label">Predicted Final Exam Score</div>
<div class="predicted-score score-counter {score_class}">{predicted_final:.1f}</div>
<div style="color: var(--text-secondary); margin-bottom: 1rem;">out of 100</div>
<div class="metric-label">Total Weighted Score</div>
<div style="font-size: 2rem; font-weight: 700; color: var(--accent-secondary);">{total_weighted:.1f}%</div>
<div class="grade-letter grade-glow {grade_class}">{grade}</div>
<div style="color: var(--text-secondary); margin-top: 0.5rem;">GPA: {gpa}</div>
<div style="margin-top: 1.5rem;">
<span class="status-badge status-pulse {status_class}">{status_text}</span>
</div></div>"""
            
            st.markdown(result_html, unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="section-header">💡 AI Recommendations</div>', unsafe_allow_html=True)
            
            if passing:
                if grade in ['A', 'A-']:
                    st.success("🌟 **Excellent Performance!**\n\nYou're on track for a top grade. Keep up the great work!")
                elif grade.startswith('B'):
                    st.info("📈 **Good Performance!**\n\nTo reach an A grade, try:\n- Increase study hours by 3-5 hours/week\n- Focus on improving midterm score")
                else:
                    st.warning("💪 **Room for Improvement**\n\nTo improve your grade:\n- Increase study hours significantly\n- Seek help for difficult topics\n- Don't miss any assignments")
            else:
                st.error("⚠️ **Action Required!**\n\nYou need to improve to pass. Consider:\n- Meeting with your instructor\n- Forming study groups\n- Increasing study hours dramatically")
                
                # Simulate improvements
                for extra_hours in range(1, 20):
                    test_features = features.copy()
                    test_features[0, 1] = hours_studied + extra_hours
                    new_pred = model.predict(test_features)[0]
                    new_total = (
                        new_pred * WEIGHTS['finals'] +
                        midterm * WEIGHTS['mids'] +
                        quiz_avg * WEIGHTS['quizzes'] +
                        assignment_avg * WEIGHTS['assignments'] +
                        presentation_marks * WEIGHTS['presentation'] +
                        internal_marks * WEIGHTS['internal']
                    )
                    new_grade = get_relative_grade(new_total)
                    if new_grade != 'F':
                        st.markdown(f"""
                        <div class="recommendation-card">
                            <div class="recommendation-title">📚 Recommended Action</div>
                            <p>Increase study hours from <strong>{hours_studied}h</strong> to <strong>{hours_studied + extra_hours}h</strong> per week.</p>
                            <p>Projected Grade: <strong>{new_grade}</strong> (GPA: {GPA_SCALE[new_grade]})</p>
                        </div>
                        """, unsafe_allow_html=True)
                        break
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Performance Chart
        st.markdown('<div class="section-header">📊 Performance Breakdown</div>', unsafe_allow_html=True)
        
        # Create radar chart
        categories = ['Attendance', 'Study Hours', 'Quizzes', 'Assignments', 'Midterm', 'Presentation', 'Internal']
        values = [
            attendance,
            (hours_studied / 40) * 100,  # Normalize to 100
            quiz_avg,
            assignment_avg,
            midterm,
            presentation_marks,
            internal_marks
        ]
        values.append(values[0])  # Close the radar
        categories.append(categories[0])
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            fillcolor='rgba(99, 102, 241, 0.3)',
            line=dict(color='#6366f1', width=2),
            name='Your Performance'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    tickfont=dict(color='#94a3b8'),
                    gridcolor='rgba(100, 116, 139, 0.2)'
                ),
                angularaxis=dict(
                    tickfont=dict(color='#f8fafc', size=12),
                    gridcolor='rgba(100, 116, 139, 0.2)'
                ),
                bgcolor='rgba(26, 26, 62, 0.5)'
            ),
            showlegend=False,
            # Disable interactions
            hovermode=False,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=400,
            margin=dict(l=80, r=80, t=40, b=40)
        )
        
        col_chart1, col_chart2 = st.columns([1, 1])
        
        with col_chart1:
            st.plotly_chart(fig, use_container_width=True, config={'staticPlot': True})
        
        with col_chart2:
            # Bar chart for component scores
            components = ['Finals (Pred)', 'Midterm', 'Quizzes', 'Assignments', 'Presentation', 'Internal']
            scores = [predicted_final, midterm, quiz_avg, assignment_avg, presentation_marks, internal_marks]
            weighted_contributions = [
                predicted_final * 0.40,
                midterm * 0.25,
                quiz_avg * 0.15,
                assignment_avg * 0.10,
                presentation_marks * 0.05,
                internal_marks * 0.05
            ]
            
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                x=components,
                y=weighted_contributions,
                marker=dict(
                    color=['#6366f1', '#14b8a6', '#f59e0b', '#22c55e', '#ec4899', '#8b5cf6'],
                    line=dict(width=0)
                ),
                text=[f'{c:.1f}' for c in weighted_contributions],
                textposition='outside',
                textfont=dict(color='#f8fafc')
            ))
            
            fig2.update_layout(
                title=dict(
                    text='Weighted Contribution to Final Grade',
                    font=dict(color='#f8fafc', size=14)
                ),
                xaxis=dict(
                    tickfont=dict(color='#94a3b8'),
                    gridcolor='rgba(100, 116, 139, 0.1)'
                ),
                yaxis=dict(
                    tickfont=dict(color='#94a3b8'),
                    gridcolor='rgba(100, 116, 139, 0.2)',
                    title='Points'
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=400,
                margin=dict(l=40, r=40, t=60, b=40)
            )
            
            st.plotly_chart(fig2, use_container_width=True, config={'staticPlot': True})

# Footer
st.markdown("""
<div class="footer">
    <span class="footer-logo">AI Grade Predictor</span> | Air University | 
    Built with ❤️ using Streamlit & Machine Learning
</div>
""", unsafe_allow_html=True)
