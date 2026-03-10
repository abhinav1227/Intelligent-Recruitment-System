import json
import random
import re
import spacy
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from config import BERT_MODEL_NAME, SPACY_MODEL_NAME, TEMPLATES

class InterviewEngine:
    def __init__(self):
        # Logic: Initialize heavy NLP models once
        self.nlp = spacy.load(SPACY_MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
        self.qa_model = AutoModelForQuestionAnswering.from_pretrained(BERT_MODEL_NAME)

    def conduct_mcq(self, mcq_file_path):
        with open(mcq_file_path, 'r') as f:
            questions = json.load(f)
            
        tech_score = 0
        hr_score = 0
        
        print("\n--- Phase 1: MCQ Round ---")
        for category, section_score in [("Tech_MCQ", tech_score), ("HR_MCQ", hr_score)]:
            if category not in questions: continue
            
            for q_text, options in questions[category].items():
                print(f"\n{q_text}")
                for opt_key, opt_val in options.items():
                    if opt_key != "Answer":
                        print(f"{opt_key}: {opt_val}")
                
                ans = input("Your answer (A/B/C/D): ").strip().upper()
                if ans == options.get("Answer"):
                    if category == "Tech_MCQ": tech_score += 1
                    else: hr_score += 1
                    
        return tech_score, hr_score
    
    def conduct_hr_descriptive(self, mcq_file_path):
        # Logic: We extract the HR descriptive questions from the JSON just like the MCQ round.
        with open(mcq_file_path, 'r') as f:
            data = json.load(f)
            
        print("\n--- Phase 3: HR Descriptive Round ---")
        # Fallback list just in case the JSON key is missing
        hr_questions = list(data.get("HR_Desc", {}).keys()) if "HR_Desc" in data else [
            "Tell me about yourself?", 
            "What are your weakness and strength?", 
            "Where do you see yourself in 5 years?"
        ]

        results = []
        for q in hr_questions:
            user_answer = input(f"\nQ: {q}\nAns: ")
            results.append({"question": q, "user_answer": user_answer})
            
        return results

    def extract_keywords(self, doc_text, n=5):
        doc = self.nlp(doc_text)
        keywords = []
        # Logic: Look for Noun chunks to form meaningful question subjects
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) in [2, 3] and not chunk.root.is_stop:
                keywords.append(chunk.text.lower())
        
        unique_keys = list(set(keywords))
        return random.sample(unique_keys, min(n, len(unique_keys)))

    def conduct_descriptive(self, context_text):
        print("\n--- Phase 2: Descriptive Round ---")
        keywords = self.extract_keywords(context_text, 5)
        results = []
        
        max_chunk_size = 250
        doc = self.nlp(context_text)
        # Logic: Chunking prevents token-limit crashes in BERT
        chunked_text = [doc[i:i+max_chunk_size].text for i in range(0, len(doc), max_chunk_size)]

        for key in keywords:
            question = random.choice(TEMPLATES).format(subject=key)
            user_answer = input(f"\nQ: {question}\nAns: ")
            
            model_answer = ""
            for text_chunk in chunked_text:
                inputs = self.tokenizer.encode_plus(question, text_chunk, add_special_tokens=True, return_tensors="pt")
                answer_start_scores, answer_end_scores = self.qa_model(**inputs).values()
                
                answer_start = torch.argmax(answer_start_scores)
                answer_end = torch.argmax(answer_end_scores) + 1
                answer_chunk = self.tokenizer.convert_tokens_to_string(
                    self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end])
                )
                model_answer += answer_chunk.strip() + " "

            # Clean up special BERT tokens
            model_answer = re.sub(r'\[CLS\].*?\[SEP\]', '', model_answer).replace('[CLS]', '').strip()
            results.append({"question": question, "user_answer": user_answer, "model_answer": model_answer})
            
        return results