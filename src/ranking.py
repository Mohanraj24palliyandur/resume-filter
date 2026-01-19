import pandas as pd
from typing import List, Dict, Tuple
import os


class CandidateRanker:
    """
    A class to rank candidates based on similarity scores and additional criteria.
    """

    def __init__(self):
        """
        Initialize the CandidateRanker.
        """
        self.ranked_candidates = []

    def rank_candidates(self, similarities: List[Tuple[str, float]], additional_criteria: Dict = None) -> pd.DataFrame:
        """
        Rank candidates based on similarity scores and additional criteria.

        Args:
            similarities (List[Tuple[str, float]]): List of (candidate_name, similarity_score) tuples.
            additional_criteria (Dict): Additional criteria for ranking (optional).

        Returns:
            pd.DataFrame: Ranked candidates DataFrame.
        """
        # Create DataFrame from similarities
        df = pd.DataFrame(similarities, columns=['candidate_name', 'similarity_score'])

        # Convert similarity to percentage
        df['similarity_percentage'] = (df['similarity_score'] * 100).round(2)

        # Add ranking
        df = df.sort_values('similarity_score', ascending=False).reset_index(drop=True)
        df['rank'] = df.index + 1

        # Add additional criteria if provided
        if additional_criteria:
            for criterion, values in additional_criteria.items():
                if criterion in df.columns:
                    continue  # Skip if already exists
                df[criterion] = values

        self.ranked_candidates = df
        return df

    def apply_skill_weighting(self, skills_required: List[str], candidate_skills: Dict[str, List[str]]) -> Dict[str, float]:
        """
        Apply skill-based weighting to similarity scores.

        Args:
            skills_required (List[str]): List of required skills from job description.
            candidate_skills (Dict[str, List[str]]): Dictionary of candidate skills.

        Returns:
            Dict[str, float]: Skill match percentages for each candidate.
        """
        skill_weights = {}

        for candidate, skills in candidate_skills.items():
            if not skills_required:
                skill_weights[candidate] = 0.0
                continue

            matched_skills = set(skill.lower() for skill in skills) & set(skill.lower() for skill in skills_required)
            skill_weights[candidate] = (len(matched_skills) / len(skills_required)) * 100

        return skill_weights

    def apply_experience_weighting(self, required_experience: int, candidate_experience: Dict[str, int]) -> Dict[str, float]:
        """
        Apply experience-based weighting.

        Args:
            required_experience (int): Required years of experience.
            candidate_experience (Dict[str, int]): Dictionary of candidate experience years.

        Returns:
            Dict[str, float]: Experience match scores.
        """
        experience_scores = {}

        for candidate, exp in candidate_experience.items():
            if required_experience == 0:
                experience_scores[candidate] = 100.0
            elif exp >= required_experience:
                experience_scores[candidate] = 100.0
            else:
                experience_scores[candidate] = (exp / required_experience) * 100

        return experience_scores

    def get_weighted_ranking(self, base_similarities: List[Tuple[str, float]],
                           skill_weights: Dict[str, float] = None,
                           experience_weights: Dict[str, float] = None,
                           skill_weight: float = 0.3,
                           experience_weight: float = 0.2) -> pd.DataFrame:
        """
        Calculate weighted ranking combining similarity, skills, and experience.

        Args:
            base_similarities (List[Tuple[str, float]]): Base similarity scores.
            skill_weights (Dict[str, float]): Skill match weights.
            experience_weights (Dict[str, float]): Experience match weights.
            skill_weight (float): Weight for skills (0-1).
            experience_weight (float): Weight for experience (0-1).

        Returns:
            pd.DataFrame: Weighted ranking DataFrame.
        """
        df = pd.DataFrame(base_similarities, columns=['candidate_name', 'base_similarity'])

        # Add skill and experience weights
        if skill_weights:
            df['skill_weight'] = df['candidate_name'].map(skill_weights).fillna(0)
        else:
            df['skill_weight'] = 0

        if experience_weights:
            df['experience_weight'] = df['candidate_name'].map(experience_weights).fillna(0)
        else:
            df['experience_weight'] = 0

        # Calculate weighted score
        similarity_weight = 1 - skill_weight - experience_weight
        df['weighted_score'] = (
            df['base_similarity'] * similarity_weight +
            (df['skill_weight'] / 100) * skill_weight +
            (df['experience_weight'] / 100) * experience_weight
        )

        # Sort by weighted score
        df = df.sort_values('weighted_score', ascending=False).reset_index(drop=True)
        df['rank'] = df.index + 1

        # Convert to percentages
        df['weighted_percentage'] = (df['weighted_score'] * 100).round(2)
        df['base_similarity_percentage'] = (df['base_similarity'] * 100).round(2)

        return df

    def save_results(self, df: pd.DataFrame, output_path: str = "output/results.csv"):
        """
        Save ranking results to CSV file.

        Args:
            df (pd.DataFrame): Results DataFrame.
            output_path (str): Path to save the results.
        """
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")

    def generate_summary_stats(self, df: pd.DataFrame) -> Dict:
        """
        Generate summary statistics from the ranking results.

        Args:
            df (pd.DataFrame): Results DataFrame.

        Returns:
            Dict: Summary statistics.
        """
        stats = {
            'total_candidates': len(df),
            'average_similarity': df['similarity_percentage'].mean() if 'similarity_percentage' in df.columns else 0,
            'top_candidate_score': df['similarity_percentage'].max() if 'similarity_percentage' in df.columns else 0,
            'candidates_above_70_percent': len(df[df['similarity_percentage'] > 70]) if 'similarity_percentage' in df.columns else 0,
        }

        return stats


if __name__ == "__main__":
    # Example usage
    ranker = CandidateRanker()

    # Sample similarities
    similarities = [
        ("candidate1.pdf", 0.85),
        ("candidate2.pdf", 0.72),
        ("candidate3.pdf", 0.91),
        ("candidate4.pdf", 0.68)
    ]

    # Rank candidates
    ranked_df = ranker.rank_candidates(similarities)
    print(ranked_df)

    # Save results
    ranker.save_results(ranked_df)