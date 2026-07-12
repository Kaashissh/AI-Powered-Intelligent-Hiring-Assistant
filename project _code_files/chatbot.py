# chatbot.py - Retrieval-based Conversational Assistant
#
# Lets a candidate ask natural-language questions about their resume
# analysis (e.g. "what's my score?", "what skills am I missing?") AND
# general resume/career questions that don't need an analysis at all
# (e.g. "how long should my resume be?", "what is ATS?").
#
# How it classifies a message (two signals combined, not just one):
#   1. TF-IDF + cosine similarity against example phrasings per intent.
#   2. A keyword-overlap bonus per intent, so short/blunt questions
#      ("score?", "missing skills") still match confidently even
#      when they don't closely resemble the example sentences.
# The intent with the highest combined score wins, as long as it
# clears a (fairly low) minimum bar — this is what fixes the "only
# ever falls back to the same 3 answers" problem: it's not that there
# were only 3 intents, it's that most messages weren't confident
# enough to pick any of them.

import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SIMILARITY_THRESHOLD = 0.08  # lower bar than before -> fewer "unknown" fallbacks

# ---------------------------------------------------------------------
# Intents that need a prior analysis result (context) to answer well.
# ---------------------------------------------------------------------
CONTEXT_INTENTS = [
    {
        "intent": "ask_score",
        "keywords": ["score", "overall", "match score", "percent", "percentage"],
        "examples": [
            "what is my match score", "how well did I score", "what score did I get",
            "tell me my overall score", "how good is my match", "score", "what's my score",
            "give me my score", "did I pass", "what's the result",
        ],
    },
    {
        "intent": "ask_category",
        "keywords": ["category", "rating", "rated", "classification"],
        "examples": [
            "what match category am I in", "is this a good match", "how would you rate my match",
            "what category was my resume rated", "am I a good fit", "is it a good match",
        ],
    },
    {
        "intent": "ask_missing_skills",
        "keywords": ["missing", "lack", "don't have", "gap", "need to add"],
        "examples": [
            "what skills am I missing", "which skills should I add", "what do I need to improve my resume",
            "what skills are missing from my resume", "what should I learn to match this job",
            "missing skills", "what am I lacking", "what gaps do I have",
        ],
    },
    {
        "intent": "ask_matched_skills",
        "keywords": ["matched", "have", "already", "strengths", "good at"],
        "examples": [
            "which skills matched", "what skills did I already have", "which of my skills matched the job",
            "what strengths does my resume have", "matched skills", "what do I already have",
        ],
    },
    {
        "intent": "ask_semantic_score",
        "keywords": ["semantic", "meaning", "context"],
        "examples": [
            "what is my semantic similarity score", "how semantically similar is my resume",
            "what does semantic similarity mean", "semantic score",
        ],
    },
    {
        "intent": "ask_tfidf_score",
        "keywords": ["tfidf", "tf-idf", "keyword match", "keyword score"],
        "examples": [
            "what is my tfidf score", "how did the keyword matching go", "what is the tf-idf similarity",
            "keyword score",
        ],
    },
    {
        "intent": "ask_skill_score",
        "keywords": ["skill match", "skill score"],
        "examples": [
            "what is my skill match score", "how many skills matched percentage wise",
            "skill match score",
        ],
    },
    {
        "intent": "ask_recommendations",
        "keywords": ["improve", "better", "suggestion", "advice", "recommend", "increase"],
        "examples": [
            "how can I improve my resume", "how do I make my resume better", "tips to improve my resume",
            "how can I increase my match score", "any suggestions for my resume", "what are your recommendations",
            "how do I get a higher score", "what should I fix",
        ],
    },
]

# ---------------------------------------------------------------------
# Intents that DON'T need a prior analysis — general resume / tool help.
# ---------------------------------------------------------------------
GENERAL_INTENTS = [
    {
        "intent": "greeting",
        "keywords": [],
        "examples": ["hi", "hello", "hey there", "good morning", "hey", "yo", "greetings"],
        "answer": (
            "Hi! I'm your resume assistant. Run an analysis above, then ask me about "
            "your score, matched/missing skills, or how to improve. You can also ask "
            "me general resume questions any time, e.g. 'how long should my resume be?'"
        ),
    },
    {
        "intent": "thanks",
        "keywords": ["thank"],
        "examples": ["thank you", "thanks a lot", "appreciate it", "thanks", "thx", "cheers"],
        "answer": "You're welcome! Good luck with your application.",
    },
    {
        "intent": "goodbye",
        "keywords": ["bye", "goodbye", "see you"],
        "examples": ["bye", "goodbye", "see you later", "that's all for now", "gotta go"],
        "answer": "Good luck with your job search! Come back any time you want another resume checked.",
    },
    {
        "intent": "ask_capabilities",
        "keywords": ["what can you do", "help me with", "capabilities", "features"],
        "examples": [
            "what can you do", "what can you help with", "what are your features",
            "how can you help me", "what questions can I ask",
        ],
        "answer": (
            "I can tell you your match score, category, matched/missing skills, and "
            "the semantic/TF-IDF/skill breakdown from your last analysis. I can also "
            "answer general resume questions — resume length, formatting, ATS, cover "
            "letters, keywords, and how the scoring works."
        ),
    },
    {
        "intent": "ask_explain_scoring",
        "keywords": ["how is the score calculated", "how does scoring work", "how do you calculate"],
        "examples": [
            "how is the score calculated", "how does this tool work", "how do you calculate the match",
            "explain the scoring method", "how does the matching algorithm work",
        ],
        "answer": (
            "Your overall score combines three signals: semantic similarity (40%, "
            "meaning-based comparison via sentence embeddings), TF-IDF similarity "
            "(30%, keyword/phrase overlap), and skill match (30%, overlap between "
            "detected skills in your resume and the job description)."
        ),
    },
    {
        "intent": "ask_supported_files",
        "keywords": ["file type", "upload", "pdf", "docx", "format supported"],
        "examples": [
            "what file types can I upload", "can I upload a pdf", "does it support docx",
            "what formats are supported",
        ],
        "answer": "You can upload a resume as PDF or DOCX, or just paste the plain text directly.",
    },
    {
        "intent": "ask_resume_length",
        "keywords": ["how long", "pages", "length"],
        "examples": [
            "how long should my resume be", "how many pages should a resume have",
            "is a two page resume okay", "resume length",
        ],
        "answer": (
            "For most roles, one page is ideal (up to two pages if you have 10+ years "
            "of experience). Focus on your most relevant and recent achievements rather "
            "than trying to list everything."
        ),
    },
    {
        "intent": "ask_ats_tips",
        "keywords": ["ats", "applicant tracking", "screening system"],
        "examples": [
            "what is ats", "how do I pass ats", "what is an applicant tracking system",
            "how do resumes get screened",
        ],
        "answer": (
            "ATS (Applicant Tracking System) software scans resumes for relevant "
            "keywords before a human sees them. Use standard section headings (Experience, "
            "Education, Skills), avoid text inside images/tables, and mirror the exact "
            "terms used in the job description wherever truthfully applicable."
        ),
    },
    {
        "intent": "ask_keyword_tips",
        "keywords": ["keyword", "terminology", "wording"],
        "examples": [
            "how do I use keywords in my resume", "should I match the job description wording",
            "keyword tips",
        ],
        "answer": (
            "Mirror the exact terminology from the job description where it truthfully "
            "applies — e.g. if they say 'CI/CD pipelines' and you have that experience, "
            "use that phrase rather than a rough synonym. This helps both ATS keyword "
            "scans and human skimmers."
        ),
    },
    {
        "intent": "ask_summary_tips",
        "keywords": ["summary", "objective", "profile"],
        "examples": [
            "how do I write a resume summary", "what should my resume objective say",
            "tips for a professional summary",
        ],
        "answer": (
            "Keep your summary to 2-3 sentences: your role/title, years of experience, "
            "and 1-2 standout achievements or specialties. Write it last, after you know "
            "what the rest of the resume emphasizes."
        ),
    },
    {
        "intent": "ask_format_tips",
        "keywords": ["format", "layout", "design", "template", "bullet"],
        "examples": [
            "how should I format my resume", "what layout works best",
            "any formatting tips", "how do I make my bullet points better",
        ],
        "answer": (
            "Use concise bullet points starting with strong action verbs, and quantify "
            "results where possible (e.g. 'reduced processing time by 30%'). Keep "
            "consistent fonts/spacing, and avoid dense paragraphs — recruiters skim, "
            "they don't read line by line."
        ),
    },
    {
        "intent": "ask_soft_skills_tips",
        "keywords": ["soft skill", "communication", "leadership", "teamwork"],
        "examples": [
            "should I include soft skills", "how do I show leadership on my resume",
            "how do I demonstrate teamwork",
        ],
        "answer": (
            "Show soft skills through concrete examples rather than just listing them — "
            "e.g. 'led a team of 5 to deliver X' demonstrates leadership better than "
            "writing 'leadership skills' as a bullet point."
        ),
    },
    {
        "intent": "ask_cover_letter",
        "keywords": ["cover letter"],
        "examples": [
            "do I need a cover letter", "how do I write a cover letter",
            "should I include a cover letter",
        ],
        "answer": (
            "A cover letter is worth including when it's optional-but-requested or when "
            "you have a specific reason for wanting the role — it's your chance to add "
            "context a resume can't (career change, referral, specific motivation)."
        ),
    },
]

ALL_INTENTS = CONTEXT_INTENTS + GENERAL_INTENTS


def _normalize(text):
    return re.sub(r"[^a-z0-9\s]", " ", text.lower())


class ResumeChatbot:
    """Retrieval-based conversational assistant for candidate Q&A.

    Usage:
        chatbot = ResumeChatbot()
        reply = chatbot.respond(user_message, context=analysis_result)

    `context` is the dict returned by ResumeJobMatcher.analyze_resume().
    Context-dependent questions (score, skills, etc.) will ask you to
    run an analysis first if `context` is None; general questions
    (resume tips, ATS, tool capabilities) work with or without it.
    """

    def __init__(self):
        self.examples = []
        self.intent_of_example = []
        for item in ALL_INTENTS:
            for ex in item["examples"]:
                self.examples.append(ex)
                self.intent_of_example.append(item["intent"])

        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.example_matrix = self.vectorizer.fit_transform(self.examples)

        self.keywords_by_intent = {item["intent"]: item.get("keywords", []) for item in ALL_INTENTS}
        self.general_answer_by_intent = {
            item["intent"]: item["answer"] for item in GENERAL_INTENTS
        }

    def _keyword_bonus(self, message_norm, intent):
        bonus = 0.0
        for kw in self.keywords_by_intent.get(intent, []):
            if kw in message_norm:
                bonus += 0.25
        return bonus

    def classify_intent(self, message):
        message_norm = _normalize(message)
        vec = self.vectorizer.transform([message])
        sims = cosine_similarity(vec, self.example_matrix)[0]

        # Aggregate similarity per intent: take the best-matching example
        # for each intent, then add a keyword-overlap bonus.
        best_score_by_intent = {}
        for idx, sim in enumerate(sims):
            intent = self.intent_of_example[idx]
            if sim > best_score_by_intent.get(intent, -1):
                best_score_by_intent[intent] = sim

        combined = {
            intent: score + self._keyword_bonus(message_norm, intent)
            for intent, score in best_score_by_intent.items()
        }

        best_intent = max(combined, key=combined.get)
        best_score = combined[best_intent]

        if best_score < SIMILARITY_THRESHOLD:
            return "unknown", best_score
        return best_intent, best_score

    def respond(self, message, context=None):
        """context: the dict returned by ResumeJobMatcher.analyze_resume()
        (keys: match_category, overall_score, detailed_scores,
        matched_skills, missing_skills, recommendations)."""

        message = (message or "").strip()
        if not message:
            return "Please type a question — you can ask about your score, skills, resume tips, or how this tool works."

        intent, _ = self.classify_intent(message)

        # General intents work with or without an analysis.
        if intent in self.general_answer_by_intent:
            return self.general_answer_by_intent[intent]

        # Everything below needs a prior analysis result.
        if not context:
            return (
                "I don't have an analysis to reference yet for that question. Please run "
                "'Analyze Resume Match' first, then ask me again — or ask me a general "
                "resume question (e.g. 'how long should my resume be?') any time."
            )

        if intent == "ask_score":
            pct = context.get("overall_score", 0) * 100
            category = context.get("match_category", "")
            return f"Your overall match score is {pct:.1f}% ({category})."

        if intent == "ask_category":
            return f"Your resume was rated as: {context.get('match_category', 'unknown')}."

        if intent == "ask_missing_skills":
            missing = context.get("missing_skills", [])
            if missing:
                return f"These skills from the job description weren't detected in your resume: {', '.join(missing)}."
            return "No missing skills were detected — nice work!"

        if intent == "ask_matched_skills":
            matched = context.get("matched_skills", [])
            if matched:
                return f"Your resume matched these required skills: {', '.join(matched)}."
            return "I couldn't find any directly matching skills between your resume and the job description."

        if intent == "ask_semantic_score":
            score = context.get("detailed_scores", {}).get("semantic_similarity", 0) * 100
            return f"Your semantic similarity score is {score:.1f}%, measuring how closely the overall meaning of your resume aligns with the job description."

        if intent == "ask_tfidf_score":
            score = context.get("detailed_scores", {}).get("tfidf_similarity", 0) * 100
            return f"Your TF-IDF similarity score is {score:.1f}%, measuring keyword/phrase overlap with the job description."

        if intent == "ask_skill_score":
            score = context.get("detailed_scores", {}).get("skill_match", 0) * 100
            return f"Your skill match score is {score:.1f}%, based on the overlap between skills detected in your resume and in the job description."

        if intent == "ask_recommendations":
            recs = context.get("recommendations", [])
            if recs:
                return " ".join(f"({i}) {r}" for i, r in enumerate(recs, 1))
            return "Great job! Your resume is already well-matched to this position — no specific recommendations."

        # Shouldn't normally reach here, but keep a helpful fallback.
        return (
            "I'm not fully sure I understood that. You can ask me about your match "
            "score, matched/missing skills, the semantic/TF-IDF/skill breakdown, "
            "recommendations, or general resume questions like resume length, ATS, "
            "or formatting tips."
        )


if __name__ == "__main__":
    # Quick manual test covering both context and general questions
    bot = ResumeChatbot()
    fake_context = {
        "match_category": "Good Match",
        "overall_score": 0.72,
        "detailed_scores": {
            "semantic_similarity": 0.68,
            "tfidf_similarity": 0.55,
            "skill_match": 0.75,
        },
        "matched_skills": ["python", "sql", "react"],
        "missing_skills": ["docker", "aws"],
        "recommendations": ["Add missing key skills mentioned in job requirements"],
    }
    test_questions = [
        "hi", "what's my score", "score?", "what skills am I missing",
        "how can I improve", "how long should my resume be", "what is ats",
        "do I need a cover letter", "what can you do", "thanks", "bye",
    ]
    for q in test_questions:
        print(f"> {q}\n{bot.respond(q, context=fake_context)}\n")
