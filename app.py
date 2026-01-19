import streamlit as st
import pandas as pd
from src.resume_parser import ResumeParser
from src.text_preprocessing import TextPreprocessor
from src.similarity_engine import SimilarityEngine
from src.ranking import CandidateRanker
import sqlite3
import os

# Set page configuration
st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="📄",
    layout="wide"
)

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_email' not in st.session_state:
    st.session_state.user_email = None

# Custom CSS for professional styling (theme-aware)
st.markdown("""
<style>
    /* Import Google Fonts for better typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styles - theme aware */
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    /* App background gradient */
    .stApp {
        background-color: #0a192f;
        background-attachment: fixed;
    }
    
    /* Dark theme text colors */
    h1, h2, h3, p, label {
        color: #e6f1ff;
    }
    
    /* Dark theme button styling */
    .stButton>button {
        background-color: #112240;
        color: #e6f1ff;
        border-radius: 8px;
        border: none;
    }
    
    /* Dark theme input fields */
    .stTextInput>div>div>input,
    .stTextArea textarea {
        background-color: #112240;
        color: #e6f1ff;
        border-radius: 6px;
    }
    
    /* Title styling - theme aware */
    .title {
        text-align: center;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: none;
    }
    
    /* Dark mode title override */
    [data-theme="dark"] .title {
        color: #ffffff;
        -webkit-text-fill-color: initial;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    /* Subtitle styling - theme aware */
    .subtitle {
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        opacity: 0.8;
        font-weight: 400;
    }
    
    /* Card-like containers - theme aware */
    .card {
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: all 0.3s ease;
    }
    
    /* Light mode cards */
    [data-theme="light"] .card {
        background: rgba(255, 255, 255, 0.95);
        border: 1px solid rgba(0, 0, 0, 0.1);
    }
    
    /* Dark mode cards */
    [data-theme="dark"] .card {
        background: rgba(30, 30, 30, 0.95);
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    /* Button styling - theme aware */
    .stButton>button {
        border-radius: 25px;
        padding: 12px 30px;
        font-size: 1.1rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        border: none;
        font-family: 'Inter', sans-serif;
    }
    
    /* Light mode button */
    [data-theme="light"] .stButton>button {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
    }
    
    /* Dark mode button */
    [data-theme="dark"] .stButton>button {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
    }
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 12px 30px;
        font-size: 1.1rem;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
        background: linear-gradient(45deg, #5a6fd8, #6a4190);
    }
    
    /* Text area styling - theme aware */
    .stTextArea>div>div>textarea {
        border-radius: 10px;
        border: 2px solid #667eea;
        padding: 10px;
        font-size: 1rem;
        font-family: 'Inter', sans-serif;
        transition: border-color 0.3s ease;
    }
    
    .stTextArea>div>div>textarea:focus {
        border-color: #764ba2;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* File uploader styling - theme aware */
    .stFileUploader>div>div {
        border-radius: 10px;
        border: 2px dashed #667eea;
        background: transparent;
        transition: all 0.3s ease;
    }
    
    .stFileUploader>div>div:hover {
        border-color: #764ba2;
        background: rgba(102, 126, 234, 0.05);
    }
    
    /* Metrics styling - theme aware */
    .metric-card {
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        margin: 0.5rem;
        transition: all 0.3s ease;
    }
    
    /* Light mode metrics */
    [data-theme="light"] .metric-card {
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid rgba(0, 0, 0, 0.1);
    }
    
    /* Dark mode metrics */
    [data-theme="dark"] .metric-card {
        background: rgba(45, 45, 45, 0.9);
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        color: inherit;
        font-size: 0.9rem;
        opacity: 0.8;
        font-weight: 500;
    }
    
    /* Dataframe styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    /* Progress bar styling */
    .stProgress>div>div>div {
        background: linear-gradient(45deg, #667eea, #764ba2);
    }
    
    /* Expander styling - theme aware */
    .streamlit-expanderHeader {
        border-radius: 10px;
        transition: all 0.3s ease;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
    }
    
    /* Light mode expander */
    [data-theme="light"] .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.8);
        border: 1px solid rgba(0, 0, 0, 0.1);
    }
    
    /* Dark mode expander */
    [data-theme="dark"] .streamlit-expanderHeader {
        background: rgba(45, 45, 45, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .streamlit-expanderHeader:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    /* Sidebar styling - theme aware */
    .sidebar .sidebar-content {
        border-radius: 15px;
        padding: 1rem;
        margin: 1rem 0;
        font-family: 'Inter', sans-serif;
    }
    
    /* Light mode sidebar */
    [data-theme="light"] .sidebar .sidebar-content {
        background: rgba(255, 255, 255, 0.95);
        border: 1px solid rgba(0, 0, 0, 0.1);
    }
    
    /* Dark mode sidebar */
    [data-theme="dark"] .sidebar .sidebar-content {
        background: rgba(45, 45, 45, 0.95);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Success message styling - theme aware */
    .success-message {
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: 600;
        margin: 1rem 0;
        font-family: 'Inter', sans-serif;
    }
    
    /* Light mode success */
    [data-theme="light"] .success-message {
        background: linear-gradient(45deg, #4CAF50, #45a049);
        color: white;
    }
    
    /* Dark mode success */
    [data-theme="dark"] .success-message {
        background: linear-gradient(45deg, #66BB6A, #4CAF50);
        color: white;
    }
    
    /* Warning message styling - theme aware */
    .warning-message {
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: 600;
        margin: 1rem 0;
        font-family: 'Inter', sans-serif;
    }
    
    /* Light mode warning */
    [data-theme="light"] .warning-message {
        background: linear-gradient(45deg, #ff9800, #e68900);
        color: white;
    }
    
    /* Dark mode warning */
    [data-theme="dark"] .warning-message {
        background: linear-gradient(45deg, #FFB74D, #FF9800);
        color: black;
    }
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        margin: 1rem 0;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(45deg, #667eea, #764ba2);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(45deg, #5a6fd8, #6a4190);
    }
</style>
""", unsafe_allow_html=True)

# Database functions
def login_user(email, password):
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM users WHERE email=? AND password=?",
        (email, password)
    )
    user = cursor.fetchone()
    conn.close()
    return user

def register_user(email, password):
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO users (email, password) VALUES (?, ?)",
            (email, password)
        )
        conn.commit()
        return True
    except:
        return False
    finally:
        conn.close()

def save_result(email, resume_name, score):
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO results (email, resume_name, score) VALUES (?, ?, ?)",
        (email, resume_name, score)
    )
    conn.commit()
    conn.close()

def get_user_results(email):
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute(
        "SELECT resume_name, score FROM results WHERE email=?",
        (email,)
    )
    data = cursor.fetchall()
    conn.close()
    return data

def clear_user_results(email):
    """Clear all results for a user"""
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM results WHERE email = ?", (email,))
        conn.commit()
        return True, "Results cleared successfully!"
    except Exception as e:
        return False, f"Error clearing results: {str(e)}"
    finally:
        conn.close()

# Initialize components
@st.cache_resource
def load_components():
    """Load and cache the main components."""
    parser = ResumeParser()
    preprocessor = TextPreprocessor()
    engine = SimilarityEngine()
    ranker = CandidateRanker()
    return parser, preprocessor, engine, ranker

parser, preprocessor, engine, ranker = load_components()

def login_page():
    """Display login/signup page"""
    st.markdown('<h1 class="title">🔐 AI Resume Screening System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Login to access your personalized resume screening</p>', unsafe_allow_html=True)

    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🚀 Login", type="primary"):
            user = login_user(email, password)
            if user:
                st.session_state.logged_in = True
                st.session_state.user_email = email
                st.success("Login successful! Redirecting...")
                st.rerun()
            else:
                st.error("Invalid credentials")

    with col2:
        if st.button("📝 Sign Up"):
            if register_user(email, password):
                st.success("Account created! Please login with your credentials.")
            else:
                st.error("User already exists or error occurred")

def logout():
    """Logout user"""
    st.session_state.logged_in = False
    st.session_state.user_email = None
    st.rerun()

def main():
    # Check authentication
    if not st.session_state.logged_in:
        login_page()
        return

    # Add logout button in sidebar
    with st.sidebar:
        st.header(f"👤 {st.session_state.user_email}")
        if st.button("🚪 Logout", type="secondary"):
            logout()
            return

        st.header("⚙️ Configuration")

    # Custom styled title
    st.markdown('<h1 class="title">🎯 AI Resume Screening System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Automatically analyze resumes and match them with job descriptions using NLP</p>', unsafe_allow_html=True)

    # Job description input in a card
    # st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("📝 Job Description")
    job_description = st.text_area(
        "Enter the job description:",
        height=200,
        placeholder="Paste your job description here...",
        key="job_desc"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # File upload section in a card
    # st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("📤 Resume Upload")
    uploaded_files = st.file_uploader(
        "Upload resume files (PDF/DOCX)",
        type=['pdf', 'docx'],
        accept_multiple_files=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # Analysis button
    if st.button("🚀 Analyze Resumes", type="primary"):
        if not job_description.strip():
            st.error("Please enter a job description.")
            return

        if not uploaded_files:
            st.error("Please upload at least one resume.")
            return

        with st.spinner("Processing resumes..."):
            # Process job description
            processed_job = preprocessor.preprocess_text(job_description)

            # Process resumes
            resume_texts = {}
            resume_names = []

            for uploaded_file in uploaded_files:
                # Save uploaded file temporarily
                file_path = f"temp_{uploaded_file.name}"
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

                try:
                    # Extract text based on file type
                    if uploaded_file.name.lower().endswith('.pdf'):
                        text = parser.extract_text_from_pdf(file_path)
                    elif uploaded_file.name.lower().endswith('.docx'):
                        text = parser.extract_text_from_docx(file_path)
                    else:
                        continue

                    processed_text = preprocessor.preprocess_text(text)
                    resume_texts[uploaded_file.name] = processed_text
                    resume_names.append(uploaded_file.name)

                except Exception as e:
                    st.warning(f"Error processing {uploaded_file.name}: {e}")
                finally:
                    # Clean up temp file
                    if os.path.exists(file_path):
                        os.remove(file_path)

            if not resume_texts:
                st.error("No resumes could be processed.")
                return

            # Compute similarities
            all_texts = [processed_job] + list(resume_texts.values())
            tfidf_matrix = engine.fit_transform(all_texts)

            job_vector = tfidf_matrix[0:1]
            resume_vectors = tfidf_matrix[1:]

            similarities = engine.compute_similarity(job_vector, resume_vectors)

            # Create similarity tuples
            similarity_tuples = list(zip(resume_names, similarities))

            # Rank candidates
            ranked_df = ranker.rank_candidates(similarity_tuples)

            # Save results to database
            for _, row in ranked_df.iterrows():
                save_result(
                    st.session_state.user_email,
                    row['candidate_name'],
                    row['similarity_percentage']
                )

            # Success message
            st.markdown("""
            <div class="success-message">
                ✅ Analysis Complete! AI has successfully processed and ranked all resumes.
            </div>
            """, unsafe_allow_html=True)

            # Display results in a card
            # st.markdown('<div class="card">', unsafe_allow_html=True)
            st.header("📊 Results")

            # Summary statistics with custom styling
            st.markdown("### 📈 Summary Statistics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{len(ranked_df)}</div>
                    <div class="metric-label">Total Candidates</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{ranked_df['similarity_percentage'].mean():.1f}%</div>
                    <div class="metric-label">Average Score</div>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{ranked_df['similarity_percentage'].max():.1f}%</div>
                    <div class="metric-label">Top Score</div>
                </div>
                """, unsafe_allow_html=True)
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{len(ranked_df[ranked_df['similarity_percentage'] > 70])}</div>
                    <div class="metric-label">Above 70%</div>
                </div>
                """, unsafe_allow_html=True)

            # Results table
            st.subheader("🏆 Ranked Candidates")
            st.dataframe(
                ranked_df[['rank', 'candidate_name', 'similarity_percentage']],
                width='stretch'
            )

            # st.markdown('</div>', unsafe_allow_html=True)  # Close the results card

            # Detailed analysis in a card
            # st.markdown('<div class="card">', unsafe_allow_html=True)
            st.header("🔍 Detailed Analysis")

            top_n = st.slider("Number of top candidates to analyze:", 1, min(5, len(ranked_df)), 3)

            for i in range(min(top_n, len(ranked_df))):
                candidate_name = ranked_df.iloc[i]['candidate_name']
                similarity_score = ranked_df.iloc[i]['similarity_percentage']

                with st.expander(f"📄 {candidate_name} - {similarity_score:.1f}% match"):
                    # Get explanation
                    explanation = engine.get_similarity_explanation(
                        processed_job,
                        resume_texts[candidate_name]
                    )

                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("Common Skills/Terms")
                        if explanation['common_terms']:
                            for term, score in explanation['common_terms'][:10]:
                                st.write(f"• {term} (importance: {score:.3f})")
                        else:
                            st.write("No common terms found.")

                    with col2:
                        st.subheader("Match Statistics")
                        st.write(f"Total common terms: {explanation['total_common_terms']}")
                        st.write(f"Job-specific terms: {explanation['job_unique_terms']}")
                        st.write(f"Resume-specific terms: {explanation['resume_unique_terms']}")

            st.markdown('</div>', unsafe_allow_html=True)  # Close detailed analysis card

            # Download results in a card
            # st.markdown('<div class="card">', unsafe_allow_html=True)
            st.header("💾 Download Results")
            csv = ranked_df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name="resume_screening_results.csv",
                mime="text/csv"
            )
            st.markdown('</div>', unsafe_allow_html=True)  # Close download card

    # # Previous results section
    # st.header("📚 Your Previous Results")
    user_results = get_user_results(st.session_state.user_email)

    if user_results:
        # st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📊 Your Resume Analysis History")

        # Convert to DataFrame for display
        results_df = pd.DataFrame(user_results, columns=['Resume Name', 'Score'])
        results_df['Rank'] = range(1, len(results_df) + 1)
        results_df = results_df[['Rank', 'Resume Name', 'Score']]

        st.dataframe(results_df, width='stretch')

        # Clear results button
        if st.button("🗑️ Clear All Results", type="secondary"):
            success, message = clear_user_results(st.session_state.user_email)
            if success:
                st.success(message)
                st.rerun()
            else:
                st.error(message)

        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("No previous results found. Upload and analyze some resumes to see your history here!")

if __name__ == "__main__":
    main()

