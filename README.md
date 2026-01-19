# AI Resume Screening System

An intelligent resume screening system that uses Natural Language Processing (NLP) to automatically analyze resumes and match them with job descriptions. Built with Python, Streamlit, and modern NLP libraries.

## 🚀 Features

- **Multi-format Support**: Parse PDF and DOCX resume files
- **Advanced NLP Processing**: Text cleaning, tokenization, lemmatization
- **Smart Matching**: TF-IDF vectorization with cosine similarity
- **Intelligent Ranking**: Score and rank candidates based on job fit
- **Interactive UI**: Streamlit-based web interface
- **Detailed Analysis**: Get explanations for why candidates match
- **Export Results**: Download ranking results as CSV

## 🏗️ Architecture

```
AI_Resume_Screener/
│
├── data/
│   ├── resumes/          # Place resume files here
│   └── job_descriptions/ # Place job description files here
│
├── src/
│   ├── resume_parser.py      # PDF/DOCX text extraction
│   ├── text_preprocessing.py # NLP text preprocessing
│   ├── similarity_engine.py  # TF-IDF and similarity calculation
│   └── ranking.py           # Candidate ranking and scoring
│
├── app.py                 # Streamlit web application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
└── output/
    └── results.csv       # Generated results
```

## 📋 Requirements

- Python 3.8+
- Virtual environment (recommended)

## 🛠️ Installation

1. **Clone or download the project**

   ```bash
   cd AI_Resume_Screener
   ```

2. **Create virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLP models**
   ```bash
   python -m spacy download en_core_web_sm
   ```

## 🚀 Usage

### Web Interface (Recommended)

1. **Run the Streamlit app**

   ```bash
   streamlit run app.py
   ```

2. **Open your browser** to `http://localhost:8501`

3. **Upload resumes and enter job description**

4. **Click "Analyze Resumes"** to get results

### Command Line Usage

You can also use individual components programmatically:

```python
from src.resume_parser import ResumeParser
from src.text_preprocessing import TextPreprocessor
from src.similarity_engine import SimilarityEngine
from src.ranking import CandidateRanker

# Initialize components
parser = ResumeParser()
preprocessor = TextPreprocessor()
engine = SimilarityEngine()
ranker = CandidateRanker()

# Process data
resumes = parser.get_all_resumes()
# ... (see individual module docstrings for usage)
```

## 📊 How It Works

1. **Text Extraction**: Extract text from PDF/DOCX resume files
2. **Preprocessing**: Clean text, remove stopwords, lemmatize tokens
3. **Feature Extraction**: Convert text to TF-IDF vectors
4. **Similarity Calculation**: Compute cosine similarity between job description and resumes
5. **Ranking**: Sort candidates by similarity score
6. **Analysis**: Provide detailed explanations and statistics

## 🎯 Key Components

### ResumeParser

- Extracts text from PDF files using PyMuPDF
- Extracts text from DOCX files using python-docx
- Handles multiple files in batch

### TextPreprocessor

- Cleans and normalizes text
- Removes stopwords and special characters
- Performs lemmatization using NLTK and spaCy

### SimilarityEngine

- Uses TF-IDF vectorization for feature extraction
- Calculates cosine similarity scores
- Provides detailed similarity explanations

### CandidateRanker

- Ranks candidates based on similarity scores
- Supports weighted ranking with skills and experience
- Generates summary statistics

## 📈 Premium Enhancements

- **Skill Gap Analysis**: Identify missing skills in resumes
- **Experience Detection**: Extract years of experience automatically
- **Weighted Scoring**: Combine multiple factors for better ranking
- **Batch Processing**: Handle large volumes of resumes
- **Export Reports**: Generate detailed PDF reports

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- NLP powered by [spaCy](https://spacy.io/) and [NLTK](https://www.nltk.org/)
- ML capabilities from [scikit-learn](https://scikit-learn.org/)

## 📞 Support

For questions or issues, please open an issue on GitHub or contact the development team.

---

**Made with ❤️ for HR Tech innovation**
