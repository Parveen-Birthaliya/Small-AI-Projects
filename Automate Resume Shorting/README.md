# Resume Matcher

A simple tool to rank resumes against a job description using [Sentence Transformers](https://www.sbert.net/).

## Usage
```bash
python src/resume_matcher.py --job "Looking for a Django developer" \
    --resumes "Flask dev with Python" "Django backend expert" "React frontend dev"
