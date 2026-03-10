import joblib
from config import TFIDF_MODEL_PATH, RESUME_MODEL_PATH, JOB_MAPPING, ROLE_SKILLS, DEFAULT_SKILLS

class ResumeParser:
    def __init__(self):
        # Logic: Load models upon initialization to save memory overhead later
        try:
            self.tfidf = joblib.load(TFIDF_MODEL_PATH)
            self.model = joblib.load(RESUME_MODEL_PATH)
        except Exception as e:
            print(f"Error loading models: {e}. Ensure .pkl files exist.")
            self.tfidf, self.model = None, None

    def predict_role(self, resume_text, target_position):
        if not self.model or not self.tfidf:
            return False, "Models not loaded."
            
        prediction_array = self.model.predict(self.tfidf.transform([resume_text]))
        predicted_code = prediction_array[0]
        
        predicted_role = JOB_MAPPING.get(predicted_code)
        
        # Logic: Case-insensitive comparison ensures "data science" matches "Data Science"
        if predicted_role and predicted_role.lower() == target_position.lower():
            return True, predicted_role
        return False, predicted_role
    
    def calculate_resume_score(self, resume_text, predicted_role):
        # Retrieve the required skills for the role, or use defaults
        required_skills = ROLE_SKILLS.get(predicted_role, DEFAULT_SKILLS)
        
        # Normalize text for accurate matching
        resume_lower = resume_text.lower()
        
        # Count how many required skills exist in the parsed resume text
        skills_found = 0
        for skill in required_skills:
            if skill.lower() in resume_lower:
                skills_found += 1
                
        # Calculate percentage match
        match_percentage = skills_found / len(required_skills)
        
        # Scale to a maximum of 20 points (the fuzzy logic universe constraint)
        raw_score = match_percentage * 20
        
        # Round to nearest integer and ensure it doesn't exceed bounds
        final_score = min(int(round(raw_score)), 20)
        
        return final_score