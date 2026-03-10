# Intelligent Recruitment System 

## Overview
An end-to-end, modular machine learning pipeline designed to automate and optimize the technical recruitment process. This system reduces manual HR screening time and provides an objective, data-driven approach to evaluating candidates by synthesizing resume strength, technical knowledge, and HR cultural fit.

## System Architecture
The project has been refactored from a procedural script into a highly scalable, Object-Oriented pipeline consisting of distinct modules:

* **`resume_parser.py`**: Utilizes TF-IDF encoding and an ML classifier to categorize unstructured resumes into specific job profiles and dynamically scores the resume against role-specific skill benchmarks.
* **`interview_engine.py`**: Manages the automated interview process. It administers multiple-choice questions and uses a fine-tuned BERT Large model (`bert-large-uncased-whole-word-masking-finetuned-squad`) to dynamically generate context-aware descriptive technical and HR questions.
* **`evaluator.py`**: Computes semantic similarity between candidate answers and ideal benchmarks using Spacy (`en_core_web_md`) word vectors. It then feeds these metrics into a Fuzzy Logic Control System (`scikit-fuzzy`) to calculate a holistic, unbiased final candidate score out of 100.
* **`main.py`**: The pipeline orchestrator that connects data ingestion, evaluation, and terminal output.
* **`config.py`**: Centralizes file paths, job mappings, and skill benchmarks for easy maintenance.

## Tech Stack
- **Language:** Python
- **Machine Learning & NLP:** Scikit-Learn, Transformers (Hugging Face / PyTorch), Spacy
- **Logic Engine:** SciKit-Fuzzy
- **Data Manipulation:** Pandas, NumPy
- **Model Serialization:** Joblib

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/abhinav1227/intelligent-recruitment-system.git](https://github.com/abhinav1227/intelligent-recruitment-system.git)
   cd intelligent-recruitment-system

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt

3. **Download the required Spacy NLP model:**
    ```bash
    python -m spacy download en_core_web_md

4. **Run the pipeline:**
    ``` bash
    python main.py

