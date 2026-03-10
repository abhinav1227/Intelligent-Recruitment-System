# config.py

# File Paths
TFIDF_MODEL_PATH = "ResumeFraserModelEncoding.pkl"
RESUME_MODEL_PATH = "ResumePhrasingModel.pkl"
MCQ_JSON_PATH = "mcq_questiosn.json"
INTERVIEW_TEXT_PATH = "interview.txt"
BERT_MODEL_NAME = "bert-large-uncased-whole-word-masking-finetuned-squad"
SPACY_MODEL_NAME = "en_core_web_md"

# Job Category Mapping (O(1) Lookup)
JOB_MAPPING = {
    0: "Advocate", 1: "Arts Professional", 2: "Automation Testing", 3: "Block Chain",
    4: "Business/Data Analyst", 5: "Civil Engineer", 6: "Data Science", 7: "Database Engineer",
    8: "DevOps Engineer", 9: "DotNet Developer", 10: "ETL Developer", 11: "Electrical Engineer",
    12: "Human Resource", 13: "Hadoop", 14: "Health and fitness", 15: "Java Developer",
    16: "Mechanical Engineer", 17: "Network Security Engineer", 18: "Operations Manager", 
    19: "PMO", 20: "Python Developer", 21: "SAP Developer", 22: "Sales", 23: "Selenium Testing", 
    24: "Web Designing"
}

# Question Templates
TEMPLATES = [
    "What is {subject} and what are the benefits of {subject}?",
    "Can you explain {subject}?",
    "How does {subject} work and given an application of {subject}?",
    "What are the applications of {subject}?"
]

# Dynamic Skill Benchmarks for Resume Scoring
ROLE_SKILLS = {
    "Data Science": ["python", "machine learning", "nlp", "sql", "tableau", "statistics", "pandas", "deep learning"],
    "Java Developer": ["java", "spring boot", "hibernate", "sql", "rest api", "git", "maven"],
    "Python Developer": ["python", "django", "flask", "docker", "aws", "postgresql", "rest"],
    # Add other roles from your JOB_MAPPING here as needed...
}

# Fallback skills if a role isn't explicitly defined in ROLE_SKILLS
DEFAULT_SKILLS = ["communication", "problem solving", "teamwork", "analytical", "agile"]