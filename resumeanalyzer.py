# Streamlit run resumeanalyzer.py

# resumeanalyzer.py

import streamlit as st
from dotenv import load_dotenv

# This must be the very first Streamlit command
st.set_page_config(page_title="Smart Resume & JD Analyzer Chatbot", layout="wide")

# Load environment variables
load_dotenv()

# Import our modules (ensure these files are in the same directory)
from resumebot import run_smart_resume_analyzer
from resumebotai import run_ai_resume_analyzer

# Sidebar for model selection (with a unique key)
st.sidebar.title("Resume Analyzer Options")
model_type = st.sidebar.radio("Select your model:", ["Smart Resume Analyzer", "AI Resume Analyzer"], key="model_type_radio")
st.sidebar.markdown(f"**Selected Model:** {model_type}")

# Route to the selected module
if model_type == "Smart Resume Analyzer":
    run_smart_resume_analyzer()
else:
    run_ai_resume_analyzer()
