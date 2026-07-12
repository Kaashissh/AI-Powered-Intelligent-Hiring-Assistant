# AI-Powered-Intelligent-Hiring-Assistant
An end-to-end AI system that evaluates candidate resumes against job requirements using machine learning and deep learning models, and generates personalized, explainable feedback through a retrieval-based intelligent text generation pipeline. It also features a conversational interface for candidate interaction and insights.

## Features

1. Multi-signal AI scoring — combines semantic similarity, keyword (TF-IDF) matching, and rule-based skill extraction into one weighted, explainable score
2. Detailed breakdown — see exactly how much each signal (semantic / TF-IDF / skills) contributed to the overall result
3. Skill gap analysis — clear lists of matched vs. missing skills relative to the job description
4. Actionable recommendations — auto-generated suggestions for improving a low-scoring resume
5. File upload support — analyze resumes directly from PDF or DOCX, or paste plain text
6. Interactive web UI — Streamlit app with gauge and radar chart visualizations
7. REST API — FastAPI backend with endpoints for single, batch, and file-upload analysis
8. Conversational assistant — a retrieval-based chatbot that answers questions like "what's my score?" or "what skills am I missing?", grounded in your own analysis result
8. Configurable — scoring weights, thresholds, and the skills database are all centralized in one config file


## INSTALLATION 

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt
python download_nltk_data.py
streamlit run app.py


## How Scoring Works

The overall match score is a weighted combination of three signals, computed in resume_matcher.py:

SignalTechniqueWeightSemantic similaritySentence embeddings + cosine similarity40%TF-IDF similarityKeyword/phrase overlap30%Skill matchOverlap between detected skills in resume vs. job description30%

The combined score maps to a category:


≥ 0.80 — Excellent Match
≥ 0.60 — Good Match
≥ 0.40 — Fair Match
< 0.40 — Poor Match
