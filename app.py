import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import fitz  # PyMuPDF
import pandas as pd

def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

st.title("📄 Resume Matcher")

job_desc = st.text_area("Paste Job Description Here", height=200)
uploaded_resumes = st.file_uploader("Upload Resumes (PDFs)", accept_multiple_files=True, type=["pdf"])

if st.button("Find Similarity"):
    if job_desc and uploaded_resumes:
        texts = [job_desc]
        resume_names = []

        for file in uploaded_resumes:
            resume_text = extract_text_from_pdf(file)
            texts.append(resume_text)
            resume_names.append(file.name)

        vectorizer = TfidfVectorizer(stop_words='english')
        vectors = vectorizer.fit_transform(texts)

        cosine_sim = cosine_similarity(vectors[0:1], vectors[1:]).flatten()

        df = pd.DataFrame({
            "Resume": resume_names,
            "Similarity Score": [round(score * 100, 2) for score in cosine_sim]
        }).sort_values(by="Similarity Score", ascending=False)

        st.dataframe(df)
    else:
        st.warning("Please provide both job description and resumes.")
