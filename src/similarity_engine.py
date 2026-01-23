from sentence_transformers import SentenceTransformer, util
import numpy as np
from typing import List, Dict, Tuple
import torch


SKILLS_LIST = [
    "python", "java", "nlp", "machine learning", "deep learning",
    "pandas", "numpy", "sql", "flask", "django", "streamlit",
    "javascript", "react", "html", "css", "tensorflow", "pytorch"
]


class SimilarityEngine:
    """
    A class to compute similarity between job descriptions and resumes using Sentence Transformers.
    """

    def __init__(self):
        """
        Initialize the SimilarityEngine with a pre-trained Sentence Transformer model.
        """
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def fit_transform(self, texts: List[str]):
        """
        Encode the documents (resumes + job description) using the Sentence Transformer model.

        Args:
            texts (List[str]): List of documents (resumes + job description).

        Returns:
            Tensor: Encoded document vectors.
        """
        return self.model.encode(texts, convert_to_tensor=True)

    def transform(self, texts: List[str]):
        """
        Transform new documents using the fitted model.

        Args:
            texts (List[str]): List of documents to transform.

        Returns:
            Tensor: Encoded document vectors for the new documents.
        """
        return self.model.encode(texts, convert_to_tensor=True)

    def compute_similarity(self, job_vector, resume_vectors):
        """
        Compute cosine similarity between job description and resumes.

        Args:
            job_vector: Encoded vector for job description.
            resume_vectors: Encoded vectors for resumes.

        Returns:
            List[float]: Similarity scores.
        """
        similarities = []
        job_vec = job_vector.squeeze(0)  # Remove batch dim
        for resume_vec in resume_vectors:
            sim = torch.nn.functional.cosine_similarity(job_vec, resume_vec, dim=0)
            similarities.append(sim.item())  # Return 0-1, remove *100
        return similarities

    def compute_keyword_similarity(self, job_text, resume_text):
        """
        Compute keyword-based similarity between job description and resume.

        Args:
            job_text: Job description text.
            resume_text: Resume text.

        Returns:
            float: Keyword similarity score.
        """
        jd_words = set(job_text.lower().split())
        resume_words = set(resume_text.lower().split())
        if not jd_words:
            return 0.0
        return len(jd_words & resume_words) / len(jd_words)  # Return 0-1, remove *100

    def get_top_similar_documents(self, similarities, document_names: List[str], top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Get top N most similar documents.

        Args:
            similarities (List[float]): Similarity scores.
            document_names (List[str]): Names of the documents.
            top_n (int): Number of top documents to return.

        Returns:
            List[Tuple[str, float]]: List of (document_name, similarity_score) tuples.
        """
        # Sort by similarity score (descending)
        sorted_indices = np.argsort(similarities)[::-1]
        top_indices = sorted_indices[:top_n]

        top_documents = []
        for idx in top_indices:
            top_documents.append((document_names[idx], similarities[idx]))

        return top_documents

    def get_similarity_explanation(self, job_text: str, resume_text: str, top_features: int = 10) -> Dict:
        """
        Provide a simplified explanation for similarity (semantic matching doesn't have TF-IDF features).
        """
        return {
            'common_terms': [],
            'total_common_terms': 0,
            'job_unique_terms': 0,
            'resume_unique_terms': 0
        }

    def extract_skills(self, text: str) -> list:
        text_lower = text.lower()
        found = [skill for skill in SKILLS_LIST if skill in text_lower]
        return found

    def rejection_reasons(self, jd_skills, candidate_skills, score):
        reasons = []
        missing = set(jd_skills) - set(candidate_skills)
        if missing:
            reasons.append(f"Missing key skills: {', '.join(missing)}")
        if score < 0.5:
            reasons.append("Overall similarity score is below threshold.")
        return reasons

    def improvement_suggestions(self, missing_skills):
        suggestions = []
        for skill in list(missing_skills)[:3]:
            suggestions.append(f"Learn and add projects in {skill}")
        suggestions.append("Add measurable achievements (numbers, impact)")
        suggestions.append("Tailor resume keywords to job description")
        return suggestions


if __name__ == "__main__":
    # Example usage
    engine = SimilarityEngine()

    # Sample documents
    job_desc = "We are looking for a Python developer with experience in machine learning and data science."
    resumes = [
        "I am a Python developer with 3 years of experience in ML and data analysis.",
        "I have experience in Java development and web technologies.",
        "Python, machine learning, data science, TensorFlow, scikit-learn experience."
    ]

    # Fit and transform
    all_docs = [job_desc] + resumes
    document_vectors = engine.fit_transform(all_docs)

    # Compute similarities
    job_vector = document_vectors[0:1]  # First document is job description
    resume_vectors = document_vectors[1:]  # Rest are resumes

    similarities = engine.compute_similarity(job_vector, resume_vectors)

    print("Similarities:", similarities)

    # Get top similar
    resume_names = [f"Resume {i+1}" for i in range(len(resumes))]
    top_similar = engine.get_top_similar_documents(similarities, resume_names, top_n=3)
    print("Top similar:", top_similar)