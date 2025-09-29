#!/usr/bin/env python3
"""
resume_matcher.py
-----------------
A simple script to rank resumes against a job description using sentence-transformers.
"""

import argparse
import logging
from typing import List
from sentence_transformers import SentenceTransformer, util

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def load_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """Load the transformer model for embeddings."""
    logging.info(f"Loading model: {model_name}")
    return SentenceTransformer(model_name)

def compute_similarity(model: SentenceTransformer, job_desc: str, resumes: List[str]) -> List[tuple]:
    """Compute cosine similarity between job description and resumes."""
    job_embedding = model.encode(job_desc, convert_to_tensor=True)
    resume_embeddings = model.encode(resumes, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(job_embedding, resume_embeddings)
    return [(resumes[i], float(score)) for i, score in enumerate(cosine_scores[0])]

def rank_resumes(results: List[tuple]) -> List[tuple]:
    """Sort resumes by similarity score."""
    return sorted(results, key=lambda x: x[1], reverse=True)

def main():
    parser = argparse.ArgumentParser(description="Rank resumes by similarity to a job description.")
    parser.add_argument("--job", type=str, required=True, help="Job description text")
    parser.add_argument("--resumes", type=str, nargs="+", required=True, help="List of resumes")
    args = parser.parse_args()

    model = load_model()
    results = compute_similarity(model, args.job, args.resumes)
    ranked = rank_resumes(results)

    print("\nRanking Results:")
    for resume, score in ranked:
        print(f"{resume} -> {score:.4f}")

if __name__ == "__main__":
    main()
