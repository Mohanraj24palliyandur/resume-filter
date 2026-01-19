from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Dict, Tuple


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
            ngram_range=(1, 2)  # Include unigrams and bigrams
        )
        self.tfidf_matrix = None
        self.feature_names = None

    def fit_transform(self, documents: List[str]) -> np.ndarray:
        """
        Fit the TF-IDF vectorizer on the documents and transform them.

        Args:
            documents (List[str]): List of documents (resumes + job description).

        Returns:
            np.ndarray: TF-IDF matrix.
        """
        self.tfidf_matrix = self.vectorizer.fit_transform(documents)
        self.feature_names = self.vectorizer.get_feature_names_out()
        return self.tfidf_matrix

    def transform(self, documents: List[str]) -> np.ndarray:
        """
        Transform new documents using the fitted vectorizer.

        Args:
            documents (List[str]): List of documents to transform.

        Returns:
            np.ndarray: TF-IDF matrix for the new documents.
        """
        return self.vectorizer.transform(documents)

    def compute_similarity(self, job_vector: np.ndarray, resume_vectors: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between job description and resumes.

        Args:
            job_vector (np.ndarray): TF-IDF vector for job description.
            resume_vectors (np.ndarray): TF-IDF vectors for resumes.

        Returns:
            np.ndarray: Similarity scores.
        """
        similarities = cosine_similarity(job_vector, resume_vectors)
        return similarities.flatten()

    def get_top_similar_documents(self, similarities: np.ndarray, document_names: List[str], top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Get top N most similar documents.

        Args:
            similarities (np.ndarray): Similarity scores.
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
        Provide explanation for similarity by showing common important terms.

        Args:
            job_text (str): Job description text.
            resume_text (str): Resume text.
            top_features (int): Number of top features to show.

        Returns:
            Dict: Explanation with common terms and their importance.
        """
        # Transform individual texts
        job_vector = self.transform([job_text]).toarray()[0]
        resume_vector = self.transform([resume_text]).toarray()[0]

        # Find common non-zero features
        common_indices = np.where((job_vector > 0) & (resume_vector > 0))[0]

        # Get feature importance (TF-IDF scores)
        common_features = []
        for idx in common_indices:
            feature_name = self.feature_names[idx]
            job_score = job_vector[idx]
            resume_score = resume_vector[idx]
            avg_score = (job_score + resume_score) / 2
            common_features.append((feature_name, avg_score))

        # Sort by importance
        common_features.sort(key=lambda x: x[1], reverse=True)

        return {
            'common_terms': common_features[:top_features],
            'total_common_terms': len(common_features),
            'job_unique_terms': len(np.where((job_vector > 0) & (resume_vector == 0))[0]),
            'resume_unique_terms': len(np.where((job_vector == 0) & (resume_vector > 0))[0])
        }


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