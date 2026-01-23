import re
from typing import List
import spacy

# Lazy-load SpaCy model to avoid OSError during Streamlit startup
_nlp = None

def get_nlp():
    """
    Lazy-load SpaCy model. This ensures the model is loaded only once
    and prevents OSErrors on Streamlit Cloud deployment.
    
    Returns:
        spacy.Language: The loaded SpaCy model
    """
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            # Model not found, try to download it
            try:
                from spacy.cli import download as spacy_download
                spacy_download("en_core_web_sm")
                _nlp = spacy.load("en_core_web_sm")
            except Exception as e:
                # If all else fails, print error and raise
                print(f"Error loading SpaCy model: {e}")
                raise RuntimeError(f"Failed to load SpaCy model: {e}. Make sure 'en_core_web_sm' is in requirements.txt")
    return _nlp


class TextPreprocessor:
    """
    A class to preprocess text data for NLP tasks.
    """

    def __init__(self):
        """
        Initialize the TextPreprocessor with necessary NLP tools.
        Uses SpaCy for all NLP tasks to avoid NLTK download issues on cloud platforms.
        """
        # Use SpaCy for all NLP operations (more reliable on cloud platforms)
        self.nlp = get_nlp()
        self.stop_words = self.nlp.Defaults.stop_words  # SpaCy's built-in stop words

    def clean_text(self, text: str) -> str:
        """
        Clean the text by removing special characters, extra spaces, etc.

        Args:
            text (str): Input text to clean.

        Returns:
            str: Cleaned text.
        """
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize the text into words using SpaCy (more reliable than NLTK on cloud platforms).

        Args:
            text (str): Input text to tokenize.

        Returns:
            List[str]: List of tokens.
        """
        doc = self.nlp(text)
        return [token.text for token in doc]

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Remove stopwords from the list of tokens.

        Args:
            tokens (List[str]): List of tokens.

        Returns:
            List[str]: List of tokens without stopwords.
        """
        return [token for token in tokens if token.lower() not in self.stop_words]

    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """
        Lemmatize the tokens.

        Args:
            tokens (List[str]): List of tokens.

        Returns:
            List[str]: List of lemmatized tokens.
        """
        return [self.lemmatizer.lemmatize(token) for token in tokens]

    def clean_names(self, text):
        """
        Remove common names from the text to reduce noise in NLP tasks.

        Args:
            text (str): Input text.

        Returns:
            str: Text with names removed.
        """
        names = ['mohan', 'ravi', 'kumar', 'john', 'jane', 'doe', 'smith', 'alex', 'emma', 'liam']  # Expand as needed
        text = text.lower()
        for name in names:
            text = re.sub(r'\b' + re.escape(name) + r'\b', '', text, flags=re.IGNORECASE)
        return text

    def preprocess_text(self, text: str) -> str:
        """
        Complete preprocessing pipeline: clean, tokenize, remove stopwords, lemmatize.

        Args:
            text (str): Input text to preprocess.

        Returns:
            str: Preprocessed text.
        """
        # Clean text
        cleaned_text = self.clean_text(text)
        # Convert to lowercase
        cleaned_text = cleaned_text.lower()
        # Tokenize
        tokens = self.tokenize_text(cleaned_text)
        # Remove stopwords
        tokens = self.remove_stopwords(tokens)
        # Lemmatize
        tokens = self.lemmatize_tokens(tokens)
        # Join back to string
        return ' '.join(tokens)

    def extract_skills(self, text: str) -> List[str]:
        """
        Extract potential skills from text using spaCy NER.

        Args:
            text (str): Input text.

        Returns:
            List[str]: List of extracted skills.
        """
        doc = self.nlp(text)
        skills = []

        # Simple skill extraction based on noun phrases and proper nouns
        for token in doc:
            if token.pos_ in ['NOUN', 'PROPN'] and len(token.text) > 2:
                skills.append(token.text.lower())

    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """
        Lemmatize the tokens using SpaCy.

        Args:
            tokens (List[str]): List of tokens.

        Returns:
            List[str]: List of lemmatized tokens.
        """
        # Process the text with SpaCy to get lemmas
        text = ' '.join(tokens)
        doc = self.nlp(text)
        return [token.lemma_ for token in doc]
        """
        Extract years of experience from text.

        Args:
            text (str): Input text.

        Returns:
            int: Estimated years of experience.
        """
        # Simple regex to find patterns like "5 years", "3+ years", etc.
        patterns = [
            r'(\d+)\+?\s*years?\s*(?:of\s*)?experience',
            r'experience\s*(?:of\s*)?(\d+)\+?\s*years?',
            r'(\d+)\+?\s*years?\s*(?:of\s*)?work',
        ]

        max_years = 0
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    years = int(match)
                    max_years = max(max_years, years)
                except ValueError:
                    pass

        return max_years


if __name__ == "__main__":
    # Example usage
    preprocessor = TextPreprocessor()

    sample_text = """
    I am a Python developer with 5 years of experience in machine learning,
    data science, and web development. I have worked with technologies like
    TensorFlow, scikit-learn, Django, and React.
    """

    preprocessed = preprocessor.preprocess_text(sample_text)
    skills = preprocessor.extract_skills(sample_text)
    experience = preprocessor.extract_experience_years(sample_text)

    print(f"Original: {sample_text}")
    print(f"Preprocessed: {preprocessed}")
    print(f"Skills: {skills}")
    print(f"Experience: {experience} years")