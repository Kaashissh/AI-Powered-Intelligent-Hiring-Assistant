# app.py - Streamlit Frontend

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from resume_matcher import ResumeJobMatcher
from chatbot import ResumeChatbot
import io
import PyPDF2
import docx

# Page config
st.set_page_config(
    page_title="AI Resume Matcher",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .score-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: black;
        text-align: center;
        margin: 0.5rem 0;
    }
    .recommendation-box {
        background-color: #f0f2f6;
        color: black;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_matcher_model():
    """Load the AI matcher model (cached)"""
    matcher = ResumeJobMatcher()
    matcher.download_nltk_data()
    matcher.load_models()
    
    # Try to load saved model, otherwise use fresh instance
    try:
        matcher.load_model('resume_matcher_model.pkl')
    except:
        st.warning("Pre-trained model not found. Using fresh model.")
    
    return matcher

@st.cache_resource
def load_chatbot():
    """Load the conversational assistant (cached)"""
    return ResumeChatbot()

def extract_text_from_pdf(uploaded_file):
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

def extract_text_from_docx(uploaded_file):
    """Extract text from DOCX file"""
    try:
        doc = docx.Document(uploaded_file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading DOCX: {e}")
        return ""

def create_score_visualization(scores):
    """Create visualization for scores"""
    
    # Radar chart for detailed scores
    categories = ['Semantic Similarity', 'TF-IDF Similarity', 'Skill Match']
    values = [
        scores['semantic_similarity'],
        scores['tfidf_similarity'], 
        scores['skill_match']
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Scores',
        line_color='rgb(31, 119, 180)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=False,
        title="Detailed Score Breakdown",
        font_size=12
    )
    
    return fig

def create_gauge_chart(score):
    """Create gauge chart for overall score"""
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = score * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Overall Match Score (%)"},
        delta = {'reference': 70},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 40], 'color': "lightgray"},
                {'range': [40, 60], 'color': "yellow"},
                {'range': [60, 80], 'color': "orange"},
                {'range': [80, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def main():
    """Main Streamlit app"""
    
    # Header
    st.markdown('<h1 class="main-header">AI-Powered Resume Matcher</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    Upload your resume and paste a job description to get AI-powered matching analysis 
    with detailed scores and improvement recommendations.
    """)
    
    # Initialize session state
    if 'matcher' not in st.session_state:
        with st.spinner("Loading AI models..."):
            st.session_state.matcher = load_matcher_model()

    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = load_chatbot()

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    if 'last_result' not in st.session_state:
        st.session_state.last_result = None
    
    # Sidebar
    with st.sidebar:
        st.header("Instructions")
        st.markdown("""
        1. **Upload Resume**: PDF, DOCX, or paste text
        2. **Job Description**: Paste the job requirements
        3. **Analyze**: Get AI-powered matching results
        4. **Review**: Check scores and recommendations
        """)
        
        st.header("Model Information")
        st.info("""
        **Models Used:**
        - Sentence Transformers (all-MiniLM-L6-v2)
        - TF-IDF Vectorization
        - Custom Skills Matching
        """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Resume Input")
        
        # File upload option
        uploaded_file = st.file_uploader(
            "Upload Resume (PDF/DOCX)", 
            type=['pdf', 'docx'],
            help="Upload your resume in PDF or DOCX format"
        )
        
        resume_text = ""
        
        if uploaded_file is not None:
            if uploaded_file.type == "application/pdf":
                resume_text = extract_text_from_pdf(uploaded_file)
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                resume_text = extract_text_from_docx(uploaded_file)
        
        # Text area for manual input
        resume_text = st.text_area(
            "Or paste resume text here:",
            value=resume_text,
            height=300,
            placeholder="Paste your resume content here..."
        )
    
    with col2:
        st.header("Job Description")
        
        job_description = st.text_area(
            "Paste job description:",
            height=300,
            placeholder="Paste the job description here..."
        )
        
        # Company details (optional)
        with st.expander("Additional Job Details (Optional)"):
            company_name = st.text_input("Company Name")
            job_title = st.text_input("Job Title")
            experience_level = st.selectbox(
                "Experience Level",
                ["Entry Level", "Mid Level", "Senior Level", "Executive"]
            )
    
    # Analysis button
    st.markdown("---")
    
    if st.button("Analyze Resume Match", type="primary", use_container_width=True):
        
        if not resume_text.strip():
            st.error("Please provide resume text!")
            return
            
        if not job_description.strip():
            st.error("Please provide job description!")
            return
        
        with st.spinner("Analyzing resume match with AI..."):
            
            # Perform analysis
            result = st.session_state.matcher.analyze_resume(
                resume_text, 
                job_description
            )
            
            # Save result so the chat assistant below can reference it
            st.session_state.last_result = result
            
            # Display results
            st.markdown("---")
            st.header("Analysis Results")
            
            # Overall score and category
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                score_color = "green" if result['overall_score'] > 0.7 else "orange" if result['overall_score'] > 0.5 else "red"
                st.markdown(f"""
                <div class="score-card">
                    <h2>{result['match_category']}</h2>
                    <h1 style="color: white;">{result['overall_score']:.1%}</h1>
                    <p>Overall Match Score</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Detailed visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Gauge chart
                gauge_fig = create_gauge_chart(result['overall_score'])
                st.plotly_chart(gauge_fig, use_container_width=True)
            
            with col2:
                # Radar chart
                radar_fig = create_score_visualization(result['detailed_scores'])
                st.plotly_chart(radar_fig, use_container_width=True)
            
            # Skills analysis
            st.header("🛠️ Skills Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Matched Skills")
                if result['matched_skills']:
                    for skill in result['matched_skills']:
                        st.success(f"✓ {skill}")
                else:
                    st.info("No specific skills matched")
            
            with col2:
                st.subheader("Missing Skills")
                if result['missing_skills']:
                    for skill in result['missing_skills']:
                        st.error(f"✗ {skill}")
                else:
                    st.success("All key skills present!")
            
            # Recommendations
            st.header("Recommendations")
            
            if result['recommendations']:
                for i, rec in enumerate(result['recommendations'], 1):
                    st.markdown(f"""
                    <div class="recommendation-box">
                        <strong>{i}.</strong> {rec}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("Great job! Your resume is well-matched to this position.")
            
            # Detailed scores table
            st.header("Detailed Scores")
            
            scores_df = pd.DataFrame({
                'Metric': ['Semantic Similarity', 'TF-IDF Similarity', 'Skill Match', 'Overall Score'],
                'Score': [
                    f"{result['detailed_scores']['semantic_similarity']:.1%}",
                    f"{result['detailed_scores']['tfidf_similarity']:.1%}",
                    f"{result['detailed_scores']['skill_match']:.1%}",
                    f"{result['overall_score']:.1%}"
                ],
                'Description': [
                    'Semantic meaning alignment',
                    'Keyword/phrase matching',
                    'Technical skills overlap',
                    'Weighted combined score'
                ]
            })
            
            st.dataframe(scores_df, use_container_width=True)
            
            # Download results
            st.header("Export Results")
            
            # Prepare results for download
            results_text = f"""
Resume Analysis Results
======================

Match Category: {result['match_category']}
Overall Score: {result['overall_score']:.1%}

Detailed Scores:
- Semantic Similarity: {result['detailed_scores']['semantic_similarity']:.1%}
- TF-IDF Similarity: {result['detailed_scores']['tfidf_similarity']:.1%}
- Skill Match: {result['detailed_scores']['skill_match']:.1%}

Matched Skills: {', '.join(result['matched_skills']) if result['matched_skills'] else 'None'}
Missing Skills: {', '.join(result['missing_skills']) if result['missing_skills'] else 'None'}

Recommendations:
{chr(10).join([f"- {rec}" for rec in result['recommendations']])}
"""
            
            st.download_button(
                label="📄 Download Analysis Report",
                data=results_text,
                file_name="resume_analysis_report.txt",
                mime="text/plain"
            )

    # ------------------------------------------------------------------
    # Conversational Assistant
    # ------------------------------------------------------------------
    st.markdown("---")
    st.header("💬 Ask the AI Assistant")
    st.caption(
        "Ask about your match score, matched/missing skills, or how to "
        "improve your resume. Run an analysis above first for grounded answers."
    )

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_message = st.chat_input("Ask about your resume analysis...")
    if user_message:
        st.session_state.chat_history.append({"role": "user", "content": user_message})
        reply = st.session_state.chatbot.respond(
            user_message, context=st.session_state.last_result
        )
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        st.rerun()

if __name__ == "__main__":
    main()