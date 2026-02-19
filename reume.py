import streamlit as st
import PyPDF2
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Smart Resume Analyzer", page_icon="💼", layout="wide")

st.title("💼 AI-Powered Smart Resume Screening & Job Matching System")
st.markdown("### Match your Resume with Job Description using AI 🤖")

# ----------- PDF TEXT EXTRACTION -----------
def extract_text_from_pdf(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

# ----------- TEXT CLEANING -----------
def clean_text(text):
    text = re.sub(r'[^a-zA-Z ]', '', text)
    text = text.lower()
    return text

# ----------- SIDEBAR INFO -----------
st.sidebar.header("📌 About Project")
st.sidebar.write("""
This system analyzes resumes using NLP techniques 
and calculates similarity score with job description 
using TF-IDF and Cosine Similarity.
""")

# ----------- FILE UPLOAD -----------
uploaded_file = st.file_uploader("📄 Upload Resume (PDF Only)", type=["pdf"])

job_description = st.text_area("📝 Paste Job Description Here")

if uploaded_file and job_description:

    resume_text = extract_text_from_pdf(uploaded_file)

    resume_clean = clean_text(resume_text)
    jd_clean = clean_text(job_description)

    documents = [resume_clean, jd_clean]

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)

    similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

    match_percentage = round(float(similarity_score[0][0]) * 100, 2)

    st.subheader("📊 Match Result")

    st.progress(int(match_percentage))
    st.metric("Matching Score", f"{match_percentage} %")

    # Result Analysis
    if match_percentage >= 75:
        st.success("✅ Excellent Match! High probability of selection.")
    elif match_percentage >= 50:
        st.warning("⚠ Moderate Match. Improve some skills.")
    else:
        st.error("❌ Low Match. Resume needs significant improvement.")

    # Missing Keywords
    resume_words = set(resume_clean.split())
    jd_words = set(jd_clean.split())

    missing_words = list(jd_words - resume_words)

    st.subheader("🔍 Missing Keywords (Top 20)")
    st.write(missing_words[:20])

    # Skill Strength
    common_words = list(jd_words.intersection(resume_words))

    st.subheader("💪 Matching Keywords (Top 20)")
    st.write(common_words[:20])

else:
    st.info("👆 Please upload resume and paste job description to analyze.")
