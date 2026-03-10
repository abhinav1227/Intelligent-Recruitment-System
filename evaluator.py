import spacy
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sklearn.metrics.pairwise import cosine_similarity
from config import SPACY_MODEL_NAME

class Evaluator:
    def __init__(self):
        self.nlp = spacy.load(SPACY_MODEL_NAME)
        self._setup_fuzzy_system()

    def score_text_similarity(self, user_text, reference_text):
        user_doc = self.nlp(user_text)
        ref_doc = self.nlp(reference_text)
        
        # Guard against zero-vectors
        if np.linalg.norm(user_doc.vector) == 0 or np.linalg.norm(ref_doc.vector) == 0:
            return 0.0
            
        sim = cosine_similarity(user_doc.vector.reshape(1, -1), ref_doc.vector.reshape(1, -1))
        return float(sim[0][0])

    def _setup_fuzzy_system(self):
        # Define the fuzzy logic variables and rules
        self.resume = ctrl.Antecedent(np.arange(0, 21, 1), 'resume')
        self.technical = ctrl.Antecedent(np.arange(0, 51, 1), 'technical')
        self.hr = ctrl.Antecedent(np.arange(0, 31, 1), 'hr')
        self.score = ctrl.Consequent(np.arange(0, 101, 1), 'score')

        self.resume.automf(3, names=['low', 'medium', 'high'])
        self.technical.automf(3, names=['low', 'medium', 'high'])
        self.hr.automf(3, names=['low', 'medium', 'high'])
        self.score.automf(3, names=['low', 'medium', 'high'])

        # rule mapping for scoring
        rules = [
            ctrl.Rule(self.resume['low'] & self.technical['low'] & self.hr['low'], self.score['low']),
            ctrl.Rule(self.resume['low'] & self.technical['low'] & self.hr['medium'], self.score['low']),
            ctrl.Rule(self.resume['low'] & self.technical['low'] & self.hr['high'], self.score['low']),
            ctrl.Rule(self.resume['low'] & self.technical['medium'] & self.hr['low'], self.score['medium']),
            ctrl.Rule(self.resume['low'] & self.technical['medium'] & self.hr['medium'], self.score['medium']),
            ctrl.Rule(self.resume['low'] & self.technical['medium'] & self.hr['high'], self.score['medium']),
            ctrl.Rule(self.resume['medium'] & self.technical['high'] & self.hr['low'], self.score['medium']),
            ctrl.Rule(self.resume['medium'] & self.technical['high'] & self.hr['medium'], self.score['high']),
            ctrl.Rule(self.resume['medium'] & self.technical['high'] & self.hr['high'], self.score['high']),
            ctrl.Rule(self.resume['medium'] & self.technical['low'] & self.hr['low'], self.score['low']),
            ctrl.Rule(self.resume['medium'] & self.technical['low'] & self.hr['medium'], self.score['low']),
            ctrl.Rule(self.resume['medium'] & self.technical['low'] & self.hr['high'], self.score['medium']),
            ctrl.Rule(self.resume['high'] & self.technical['medium'] & self.hr['low'], self.score['medium']),
            ctrl.Rule(self.resume['high'] & self.technical['medium'] & self.hr['medium'], self.score['medium']),
            ctrl.Rule(self.resume['high'] & self.technical['medium'] & self.hr['high'], self.score['medium']),
            ctrl.Rule(self.resume['high'] & self.technical['high'] & self.hr['low'], self.score['medium']),
            ctrl.Rule(self.resume['high'] & self.technical['high'] & self.hr['medium'], self.score['high']),
            ctrl.Rule(self.resume['high'] & self.technical['high'] & self.hr['high'], self.score['high'])
        ]
        
        self.score_ctrl = ctrl.ControlSystem(rules)
        self.score_sim = ctrl.ControlSystemSimulation(self.score_ctrl)

    def calculate_final_score(self, resume_score, tech_score, hr_score):
        # Safely cap scores to universe constraints
        self.score_sim.input['resume'] = min(max(resume_score, 0), 20)
        self.score_sim.input['technical'] = min(max(tech_score, 0), 50)
        self.score_sim.input['hr'] = min(max(hr_score, 0), 30)
        
        self.score_sim.compute()
        return self.score_sim.output['score']