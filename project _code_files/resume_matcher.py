# resume_matcher.py - Backend ML Pipeline

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings('ignore')

class ResumeJobMatcher:
    def __init__(self):
        """Initialize the Resume-Job Matching System"""
        self.sentence_model = None
        self.tfidf_vectorizer = None
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.skills_keywords = [
            'python', 'java', 'javascript', 'react', 'node.js', 'sql', 'mongodb',
            'machine learning', 'deep learning', 'tensorflow', 'pytorch', 'scikit-learn',
            'aws', 'azure', 'docker', 'kubernetes', 'git', 'agile', 'scrum'
        ]
        
    def download_nltk_data(self):
        """Download required NLTK data"""
        try:
            nltk.download('punkt')
            nltk.download('stopwords') 
            nltk.download('wordnet')
            print("NLTK data downloaded successfully")
        except Exception as e:
            print(f"Error downloading NLTK data: {e}")
    
    def load_models(self):
        """Load pre-trained models"""
        try:
            # Load sentence transformer for semantic similarity
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("Sentence transformer loaded successfully")
            
            # Initialize TF-IDF vectorizer
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            print("TF-IDF vectorizer initialized")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            
    def preprocess_text(self, text):
        """Clean and preprocess text"""
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [
            self.lemmatizer.lemmatize(token) 
            for token in tokens 
            if token not in self.stop_words and len(token) > 2
        ]
        
        return ' '.join(tokens)
    
    def extract_skills(self, text):
        """Extract skills from text"""
        text_lower = text.lower()
        found_skills = []
        
        for skill in self.skills_keywords:
            if skill.lower() in text_lower:
                found_skills.append(skill)
                
        return found_skills
    
    def calculate_similarity_scores(self, resume_text, job_description):
        """Calculate multiple similarity scores"""
        
        # Preprocess texts
        resume_clean = self.preprocess_text(resume_text)
        job_clean = self.preprocess_text(job_description)
        
        # 1. Semantic Similarity using Sentence Transformers
        resume_embedding = self.sentence_model.encode([resume_clean])
        job_embedding = self.sentence_model.encode([job_clean])
        semantic_similarity = cosine_similarity(resume_embedding, job_embedding)[0][0]
        
        # 2. TF-IDF Similarity
        tfidf_matrix = self.tfidf_vectorizer.fit_transform([resume_clean, job_clean])
        tfidf_similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        # 3. Skills Match
        resume_skills = set(self.extract_skills(resume_text))
        job_skills = set(self.extract_skills(job_description))
        
        if job_skills:
            skill_match_score = len(resume_skills.intersection(job_skills)) / len(job_skills)
        else:
            skill_match_score = 0.0
            
        # 4. Combined Score (weighted average)
        weights = {
            'semantic': 0.4,
            'tfidf': 0.3,
            'skills': 0.3
        }
        
        combined_score = (
            weights['semantic'] * semantic_similarity +
            weights['tfidf'] * tfidf_similarity +
            weights['skills'] * skill_match_score
        )
        
        return {
            'semantic_similarity': float(semantic_similarity),
            'tfidf_similarity': float(tfidf_similarity),
            'skill_match_score': float(skill_match_score),
            'combined_score': float(combined_score),
            'matched_skills': list(resume_skills.intersection(job_skills)),
            'missing_skills': list(job_skills - resume_skills)
        }
    
    def get_match_category(self, score):
        """Categorize match score"""
        if score >= 0.8:
            return "Excellent Match"
        elif score >= 0.6:
            return "Good Match"
        elif score >= 0.4:
            return "Fair Match"
        else:
            return "Poor Match"
    
    def analyze_resume(self, resume_text, job_description):
        """Main function to analyze resume against job description"""
        
        # Calculate similarities
        scores = self.calculate_similarity_scores(resume_text, job_description)
        
        # Get match category
        match_category = self.get_match_category(scores['combined_score'])
        
        # Generate recommendations
        recommendations = self.generate_recommendations(scores)
        
        result = {
            'match_category': match_category,
            'overall_score': scores['combined_score'],
            'detailed_scores': {
                'semantic_similarity': scores['semantic_similarity'],
                'tfidf_similarity': scores['tfidf_similarity'],
                'skill_match': scores['skill_match_score']
            },
            'matched_skills': scores['matched_skills'],
            'missing_skills': scores['missing_skills'],
            'recommendations': recommendations
        }
        
        return result
    
    def generate_recommendations(self, scores):
        """Generate improvement recommendations"""
        recommendations = []
        
        if scores['skill_match_score'] < 0.5:
            recommendations.append("Consider highlighting more relevant technical skills")
            recommendations.append("Add missing key skills mentioned in job requirements")
        
        if scores['semantic_similarity'] < 0.6:
            recommendations.append("Align resume content more closely with job description language")
            recommendations.append("Use industry-specific keywords and terminology")
        
        if scores['combined_score'] < 0.7:
            recommendations.append("Tailor resume more specifically to this role")
            recommendations.append("Emphasize relevant experience and achievements")
        
        return recommendations
    
    def save_model(self, filepath='resume_matcher_model.pkl'):
        """Save the trained model"""
        model_data = {
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'skills_keywords': self.skills_keywords,
            'sentence_model_name': 'all-MiniLM-L6-v2'  # Save model name instead of object
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='resume_matcher_model.pkl'):
        """Load the trained model"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.tfidf_vectorizer = model_data['tfidf_vectorizer']
            self.skills_keywords = model_data['skills_keywords']
            
            # Reload sentence transformer
            self.sentence_model = SentenceTransformer(model_data['sentence_model_name'])
            
            print(f"Model loaded from {filepath}")
        except FileNotFoundError:
            print(f"Model file {filepath} not found. Please train the model first.")
        except Exception as e:
            print(f"Error loading model: {e}")

# Usage example and model training
if __name__ == "__main__":
    # Initialize matcher
    matcher = ResumeJobMatcher()
    
    # Download required data and load models
    matcher.download_nltk_data()
    matcher.load_models()
    
    # Example usage
    sample_resume = """
    John Doe
    Software Engineer
    
    Experience:
    - 3 years of Python development
    - Built web applications using React and Node.js
    - Experience with SQL databases and MongoDB
    - Familiar with machine learning libraries like scikit-learn
    - Worked with Docker and AWS cloud services
    
    Skills: Python, JavaScript, React, SQL, Machine Learning, AWS
    """
    
    sample_job = """
    We are looking for a Software Engineer with:
    - Strong Python programming skills
    - Experience with React and JavaScript
    - Knowledge of SQL databases
    - Familiarity with machine learning and AI
    - Cloud experience (AWS preferred)
    - Docker containerization experience
    """
    
    # Analyze resume
    result = matcher.analyze_resume(sample_resume, sample_job)
    
    print("=== Resume Analysis Results ===")
    print(f"Match Category: {result['match_category']}")
    print(f"Overall Score: {result['overall_score']:.2f}")
    print(f"Matched Skills: {result['matched_skills']}")
    print(f"Missing Skills: {result['missing_skills']}")
    print("Recommendations:")
    for rec in result['recommendations']:
        print(f"- {rec}")
    
    # Save the model
    matcher.save_model()
    print("\nModel training and saving completed!")