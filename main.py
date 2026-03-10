from config import MCQ_JSON_PATH, INTERVIEW_TEXT_PATH
from resume_parser import ResumeParser
from interview_engine import InterviewEngine
from evaluator import Evaluator

def main():
    print("Initializing Intelligent Recruitment Pipeline...")
    parser = ResumeParser()
    engine = InterviewEngine()
    evaluator = Evaluator()
    
    # 1. Resume Parsing (Max 20 Points)
    resume_text = input("Paste your resume here: ")
    target_role = input("Enter target position: ")
    
    is_match, predicted = parser.predict_role(resume_text, target_role)
    if not is_match:
        print(f"Rejected. Applied for {target_role}, but model predicted {predicted}.")
        return
    
    print(f"\nPassed Stage 1! Proceeding as {predicted}.")
    resume_score = parser.calculate_resume_score(resume_text, predicted)
    print(f"[System] Dynamic Resume Score: {resume_score}/20")
    
    # 2. MCQ Interview (Tech Max 10, HR Max 10)
    mcq_tech, mcq_hr = engine.conduct_mcq(MCQ_JSON_PATH)
    
    # 3. Tech Descriptive (Max 40 points)
    with open(INTERVIEW_TEXT_PATH, 'r', encoding='utf-8') as f:
        context_text = f.read()
    desc_tech_results = engine.conduct_descriptive(context_text)
    
    desc_tech_score = 0
    for res in desc_tech_results:
        sim = evaluator.score_text_similarity(res['user_answer'], res['model_answer'])
        # Logic: 5 questions * 8 multiplier = max 40 points
        desc_tech_score += (sim * 8) 
        
    # 4. HR Descriptive (Max 20 points)
    hr_desc_results = engine.conduct_hr_descriptive(MCQ_JSON_PATH)
    desc_hr_score = 0
    
    # Ideal benchmark string for HR answers to measure cosine similarity against
    ideal_hr_answer = "I am a dedicated professional looking to grow. My strengths are communication and problem solving. In five years, I want to be a technical leader contributing to business success."
    
    for res in hr_desc_results:
        sim = evaluator.score_text_similarity(res['user_answer'], ideal_hr_answer)
        # Logic: 3 questions * 6.66 multiplier = max ~20 points
        desc_hr_score += (sim * 6.66)
        
    # 5. Final Fuzzy Logic Evaluation
    total_tech = (mcq_tech * 2) + desc_tech_score 
    total_hr = (mcq_hr * 2) + desc_hr_score
    
    final_score = evaluator.calculate_final_score(resume_score, total_tech, total_hr)
    
    print("\n" + "="*40)
    print(f"--- FINAL CANDIDATE SCORE: {final_score:.2f}/100 ---")
    print(f"Breakdown -> Resume: {resume_score}/20 | Tech: {total_tech:.1f}/50 | HR: {total_hr:.1f}/30")
    print("="*40)

if __name__ == "__main__":
    main()