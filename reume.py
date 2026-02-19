import streamlit as st
import PyPDF2
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import HRFlowable
from io import BytesIO

st.set_page_config(page_title="Smart Resume Analyzer", page_icon="💼", layout="wide")

st.title("💼 AI-Powered Resume Screening & Enhancement System")

# -------- PDF TEXT EXTRACTION --------
def extract_text_from_pdf(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

# -------- TEXT CLEANING --------
def clean_text(text):
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text.lower()

# -------- IMPROVED SUMMARY GENERATOR --------
def generate_summary(skills):
    return f"Results-driven professional skilled in {', '.join(skills[:5])} with strong analytical and problem-solving abilities."

# -------- PDF GENERATION --------
def create_pdf(content):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    elements = []

    styles = getSampleStyleSheet()
    style = styles["Normal"]

    elements.append(Paragraph("<b>Improved Resume Version</b>", styles["Heading1"]))
    elements.append(Spacer(1, 0.3 * inch))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.black))
    elements.append(Spacer(1, 0.3 * inch))

    for line in content.split("\n"):
        elements.append(Paragraph(line, style))
        elements.append(Spacer(1, 0.2 * inch))

    doc.build(elements)
    buffer.seek(0)
    return buffer

# -------- UI --------
uploaded_file = st.file_uploader("📄 Upload Resume (PDF)", type=["pdf"])
job_description = st.text_area("📝 Paste Job Description Here")

if uploaded_file and job_description:

    resume_text = extract_text_from_pdf(uploaded_file)

    resume_clean = clean_text(resume_text)
    jd_clean = clean_text(job_description)

    documents = [resume_clean, jd_clean]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)

    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    match_percentage = round(float(similarity[0][0]) * 100, 2)

    st.subheader("📊 Match Result")
    st.progress(int(match_percentage))
    st.metric("Matching Score", f"{match_percentage}%")

    # Missing Skills
    resume_words = set(resume_clean.split())
    jd_words = set(jd_clean.split())
    missing_skills = list(jd_words - resume_words)

    st.subheader("🔍 Missing Keywords")
    st.write(missing_skills[:15])

    # Matching Skills
    common_skills = list(jd_words.intersection(resume_words))
    st.subheader("💪 Matching Keywords")
    st.write(common_skills[:15])

    # -------- ENHANCEMENT --------
    st.subheader("✨ AI Resume Enhancement")

    improved_summary = generate_summary(common_skills if common_skills else ["technology", "software", "development"])

    improved_resume = f"""
Professional Summary:
{improved_summary}

Key Skills:
{', '.join(common_skills[:10])}

Suggested Skills to Add:
{', '.join(missing_skills[:10])}
"""

    st.text_area("📌 Improved Resume Preview", improved_resume, height=250)

    if st.button("📥 Download Improved Resume as PDF"):
        pdf_file = create_pdf(improved_resume)
        st.download_button(
            label="Download PDF",
            data=pdf_file,
            file_name="Improved_Resume.pdf",
            mime="application/pdf"
        )

else:
    st.info("Upload resume and paste job description to continue.")
