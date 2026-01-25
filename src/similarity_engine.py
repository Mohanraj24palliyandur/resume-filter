from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Dict, Tuple


SKILLS_LIST = [
    "python", "java", "nlp", "machine learning", "deep learning",
    "pandas", "numpy", "sql", "flask", "django", "streamlit",
    "javascript", "react", "html", "css", "tensorflow", "pytorch"
]


class SimilarityEngine:
    """
    A class to compute similarity between job descriptions and resumes using TF-IDF and cosine similarity.
    """

    def __init__(self):
        """
        Initialize the SimilarityEngine with TF-IDF vectorizer.
        """
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.is_fitted = False

    def fit_transform(self, texts: List[str]):
        """
        Fit the TF-IDF vectorizer and transform the documents.

        Args:
            texts (List[str]): List of documents (resumes + job description).

        Returns:
            np.ndarray: TF-IDF matrix.
        """
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.is_fitted = True
        return tfidf_matrix

    def transform(self, texts: List[str]):
        """
        Transform new documents using the fitted vectorizer.

        Args:
            texts (List[str]): List of documents to transform.

        Returns:
            np.ndarray: TF-IDF matrix for the new documents.
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted before transform. Call fit_transform first.")
        return self.vectorizer.transform(texts)

    def compute_similarity(self, job_vector, resume_vectors):
        """
        Compute cosine similarity between job description and resumes.

        Args:
            job_vector: TF-IDF vector for job description.
            resume_vectors: TF-IDF vectors for resumes.

        Returns:
            List[float]: Similarity scores (0-1).
        """
        similarities = cosine_similarity(job_vector, resume_vectors)[0]
        return similarities.tolist()

    def compute_keyword_similarity(self, job_text, resume_text):
        """
        Compute keyword-based similarity between job description and resume.

        Args:
            job_text: Job description text.
            resume_text: Resume text.

        Returns:
            float: Keyword similarity score (0-1).
        """
        jd_words = set(job_text.lower().split())
        resume_words = set(resume_text.lower().split())
        if not jd_words:
            return 0.0
        return len(jd_words & resume_words) / len(jd_words)

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
        Provide explanation for TF-IDF similarity based on important terms and their scores.
        """
        try:
            if not hasattr(self, 'is_fitted') or not self.is_fitted:
                print("DEBUG: Vectorizer not fitted")
                return {
                    'common_terms': [],
                    'total_common_terms': 0,
                    'job_unique_terms': 0,
                    'resume_unique_terms': 0
                }

            if not hasattr(self, 'vectorizer') or self.vectorizer is None:
                print("DEBUG: Vectorizer is None")
                return {
                    'common_terms': [],
                    'total_common_terms': 0,
                    'job_unique_terms': 0,
                    'resume_unique_terms': 0
                }

            # Transform both texts
            texts = [job_text, resume_text]
            tfidf_matrix = self.vectorizer.transform(texts)

            # Get feature names
            feature_names = self.vectorizer.get_feature_names_out()

            # Get TF-IDF scores for job and resume
            job_scores = tfidf_matrix[0].toarray().flatten()
            resume_scores = tfidf_matrix[1].toarray().flatten()

            # Find common terms (terms that appear in both with some score)
            common_terms_with_scores = []
            for i, term in enumerate(feature_names):
                job_score = job_scores[i]
                resume_score = resume_scores[i]
                if job_score > 0 and resume_score > 0:
                    # Use the minimum score as the importance (how well they match)
                    importance = float(min(job_score, resume_score))
                    common_terms_with_scores.append((term, importance))

            # Sort by importance (descending)
            common_terms_with_scores.sort(key=lambda x: x[1], reverse=True)

            # Get unique terms counts
            job_unique_count = sum(1 for score in job_scores if score > 0) - len(common_terms_with_scores)
            resume_unique_count = sum(1 for score in resume_scores if score > 0) - len(common_terms_with_scores)

            # Ensure common_terms is always a list of tuples
            safe_common_terms = []
            for item in common_terms_with_scores[:top_features]:
                if isinstance(item, tuple) and len(item) == 2:
                    safe_common_terms.append((str(item[0]), float(item[1])))
                else:
                    # Skip invalid items
                    continue

            return {
                'common_terms': safe_common_terms,
                'total_common_terms': len(common_terms_with_scores),
                'job_unique_terms': max(0, job_unique_count),
                'resume_unique_terms': max(0, resume_unique_count)
            }
        except Exception as e:
            # If anything goes wrong, return safe defaults
            print(f"Error in get_similarity_explanation: {e}")
            return {
                'common_terms': [],  # Always return empty list
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
    tfidf_matrix = engine.fit_transform(all_docs)

    # Compute similarities
    job_vector = tfidf_matrix[0:1]  # First document is job description
    resume_vectors = tfidf_matrix[1:]  # Rest are resumes

    similarities = engine.compute_similarity(job_vector, resume_vectors)

    print("Similarities:", similarities)

    # Get top similar
    resume_names = [f"Resume {i+1}" for i in range(len(resumes))]
    top_similar = engine.get_top_similar_documents(similarities, resume_names, top_n=3)
    print("Top similar:", top_similar)