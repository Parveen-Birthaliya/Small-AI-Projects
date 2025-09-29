# --- Step 1: Install dependencies ---
!pip install -q sentence-transformers torch PyPDF2

# --- Step 2: Import libraries ---
from sentence_transformers import SentenceTransformer, util
from PyPDF2 import PdfReader
from google.colab import files

# --- Step 3: Upload PDF resumes (optional) ---
print("Upload your PDF resumes. If you skip, you can provide your resume text instead.")
uploaded_files = files.upload()
pdf_paths = list(uploaded_files.keys())
resumes_text = []

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text.strip()

if pdf_paths:
    for path in pdf_paths:
        text = extract_text_from_pdf(path)
        if not text:
            print(f"Warning: No text found in {path}")
            text = "Empty resume content."
        resumes_text.append(text)
else:
    print("No PDF uploaded. You can provide your resume in plain text.")
    resume_text = input("Enter your resume text (or leave empty to use default placeholder): ").strip()
    if resume_text:
        pdf_paths = ["Text_Resume"]
        resumes_text = [resume_text]
    else:
        pdf_paths = ["Default_AI_Internship_Resume"]
        resumes_text = ["This is a placeholder resume for AI internship."]

# --- Step 4: Load model ---
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded.")

# --- Step 5: Input Job Description ---
job_desc = input("Enter the job description (press Enter to use default AI internship JD): ").strip()
if not job_desc:
    job_desc = "Looking for an AI/ML internship candidate skilled in Python, PyTorch, TensorFlow, and data analysis."
    print("Using default AI internship job description.")

# --- Step 6: Compute similarity ---
def compute_similarity(job_desc, resumes_text):
    job_emb = model.encode(job_desc, convert_to_tensor=True)
    resume_embs = model.encode(resumes_text, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(job_emb, resume_embs)
    return [(resumes_text[i], float(scores[0][i])) for i in range(len(resumes_text))]

def rank_resumes(results):
    return sorted(results, key=lambda x: x[1], reverse=True)

results = compute_similarity(job_desc, resumes_text)
ranked = rank_resumes(results)

# --- Step 7: Print ranked results ---
print("\n--- Ranked Resumes ---")
for i, (text, score) in enumerate(ranked, 1):
    print(f"{i}. {pdf_paths[i-1]} -> Score: {score:.4f}")


