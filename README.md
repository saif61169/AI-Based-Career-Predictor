# AI Based Career Predictor

An AI-powered career guidance tool that predicts the top 3 most suitable job roles for a candidate based on their resume.

## Features
- Upload your resume
- Extracts skills and features
- Predicts top 3 job roles using Google's Gemma 2B IT model
- Provides a 1-line explanation for each prediction

## Installation
- Add your authentication key from HuggingFace in the code to access the LLM before running the code.
```bash
pip install -r requirements.txt
streamlit run app.py

