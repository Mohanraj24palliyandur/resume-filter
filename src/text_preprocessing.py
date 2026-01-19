import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from typing import List
import spacy

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


class TextPreprocessor:
    """
    A class to preprocess text data for NLP tasks.
    """

    def __init__(self):
        """
        Initialize the TextPreprocessor with necessary NLP tools.
        """
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.nlp = spacy.load('en_core_web_sm')

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
        Tokenize the text into words.

        Args:
            text (str): Input text to tokenize.

        Returns:
            List[str]: List of tokens.
        """
        return word_tokenize(text)

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

        return list(set(skills))  # Remove duplicates

    def extract_experience_years(self, text: str) -> int:
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