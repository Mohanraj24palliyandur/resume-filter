#!/usr/bin/env python3
"""
AI Resume Screening System - Demo Script
This script demonstrates the core functionality of the resume screening system.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.resume_parser import ResumeParser
from src.text_preprocessing import TextPreprocessor
from src.similarity_engine import SimilarityEngine
from src.ranking import CandidateRanker

def demo_resume_screening():
    """Demonstrate the resume screening functionality."""

    print("🎯 AI Resume Screening System Demo")
    print("=" * 50)

    # Initialize components
    parser = ResumeParser()
    preprocessor = TextPreprocessor()
    engine = SimilarityEngine()
    ranker = CandidateRanker()

    # Load sample job description
    job_desc_path = "data/job_descriptions/senior_python_developer.txt"
    if os.path.exists(job_desc_path):
        with open(job_desc_path, 'r') as f:
            job_description = f.read()
        print("✅ Loaded job description")
    else:
        # Fallback job description
        job_description = """
        Senior Python Developer with experience in machine learning, data science,
        web development using Django/Flask, and cloud platforms like AWS.
        """
        print("⚠️  Using default job description")

    # Load sample resumes
    resume_dir = "data/resumes/"
    resumes = {}

    if os.path.exists(resume_dir):
        for filename in os.listdir(resume_dir):
            if filename.endswith('.txt'):  # For demo, using text files
                filepath = os.path.join(resume_dir, filename)
                with open(filepath, 'r') as f:
                    resumes[filename] = f.read()

    if not resumes:
        # Fallback resumes
        resumes = {
            "john_doe_resume.txt": """
            Senior Python Developer with 6 years experience in machine learning,
            TensorFlow, PyTorch, Django, Flask, AWS, Docker, PostgreSQL.
            """,
            "jane_smith_resume.txt": """
            Full Stack Developer with 4 years experience in JavaScript, React,
            Node.js, Python, SQL databases, Git.
            """
        }
        print("⚠️  Using default resume samples")

    print(f"✅ Loaded {len(resumes)} resumes")

    # Step 1: Preprocess job description
    print("\n🔄 Step 1: Preprocessing job description...")
    processed_job = preprocessor.preprocess_text(job_description)
    print(f"Original length: {len(job_description)} characters")
    print(f"Processed length: {len(processed_job)} characters")

    # Step 2: Preprocess resumes
    print("\n🔄 Step 2: Preprocessing resumes...")
    processed_resumes = {}
    for name, text in resumes.items():
        processed_resumes[name] = preprocessor.preprocess_text(text)
        print(f"✅ Processed {name}")

    # Step 3: Calculate similarities
    print("\n🔄 Step 3: Calculating similarities...")
    all_texts = [processed_job] + list(processed_resumes.values())
    tfidf_matrix = engine.fit_transform(all_texts)

    job_vector = tfidf_matrix[0:1]
    resume_vectors = tfidf_matrix[1:]

    similarities = engine.compute_similarity(job_vector, resume_vectors)
    resume_names = list(processed_resumes.keys())

    # Step 4: Rank candidates
    print("\n🔄 Step 4: Ranking candidates...")
    similarity_tuples = list(zip(resume_names, similarities))
    ranked_df = ranker.rank_candidates(similarity_tuples)

    # Display results
    print("\n📊 RESULTS")
    print("=" * 50)
    print(ranked_df.to_string(index=False))

    # Summary statistics
    print("\n📈 SUMMARY STATISTICS")
    print("=" * 50)
    stats = ranker.generate_summary_stats(ranked_df)
    for key, value in stats.items():
        print(f"{key.replace('_', ' ').title()}: {value}")

    # Save results
    output_path = "output/demo_results.csv"
    ranker.save_results(ranked_df, output_path)

    print(f"\n💾 Results saved to {output_path}")

    # Detailed analysis for top candidate
    if len(ranked_df) > 0:
        top_candidate = ranked_df.iloc[0]['candidate_name']
        print(f"\n🔍 ANALYSIS FOR TOP CANDIDATE: {top_candidate}")
        print("=" * 50)

        explanation = engine.get_similarity_explanation(
            processed_job,
            processed_resumes[top_candidate]
        )

        print("Common important terms:")
        for term, score in explanation['common_terms'][:10]:
            print(f"  • {term} (importance: {score:.3f})")

        print("\nMatch statistics:")
        print(f"  • Total common terms: {explanation['total_common_terms']}")
        print(f"  • Job-specific terms: {explanation['job_unique_terms']}")
        print(f"  • Resume-specific terms: {explanation['resume_unique_terms']}")

    print("\n🎉 Demo completed successfully!")
    print("\n💡 To run the interactive web app:")
    print("   streamlit run app.py")

if __name__ == "__main__":
    demo_resume_screening()