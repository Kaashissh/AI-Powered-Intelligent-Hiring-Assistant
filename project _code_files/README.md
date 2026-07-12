Here we get the complete code files of this project.
app.py — Streamlit web app that lets a user paste/upload a resume and job description and view the AI match analysis and chatbot.
api.py — FastAPI REST backend exposing the same resume-matching and chatbot functionality as HTTP endpoints.
resume_matcher.py — Core ML engine (ResumeJobMatcher class) that computes semantic, TF-IDF, and skill-match scores between a resume and job description.
chatbot.py — Retrieval-based conversational assistant (ResumeChatbot class) that answers candidate questions grounded in the analysis results.
config.py — Centralized configuration: model settings, scoring weights, match thresholds, and a large skills/domains database.
download_nltk_data.py — Standalone utility script that downloads the NLTK corpora (punkt, stopwords, wordnet) required by the matcher.
requirements.txt — Python dependency manifest listing all packages needed to run the project.
resume_matcher_model.pkl — Serialized (pickled) TF-IDF vectorizer and skills list saved from a previous training/save step.
project_summary.docx — Word document describing the project's overview, features, and architecture for documentation/submission purposes.
