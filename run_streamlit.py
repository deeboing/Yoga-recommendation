"""
Streamlit launcher for Yoga Recommendation System
Run this file directly with: python run_streamlit.py
or use: streamlit run yoga_combined.py
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Import the combined system and run Streamlit
from yoga_combined import run_streamlit_standalone

if __name__ == "__main__":
    run_streamlit_standalone()
