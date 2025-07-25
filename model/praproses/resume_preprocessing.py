# Coding by Samitha Randika | https://www.linkedin.com/in/samitha-randika-edirisinghe-b3a68a2b6 #
import pandas as pd
import re
from utils.text_processing import clean_text, tokenize_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

def extract_sections(resume_text):
    sections = {
        'experience': '',
        'education': '',
        'skills': '',
        'summary': ''
    }
    
    # Simple regex-based section extraction
    exp_match = re.search(r'experience(.+?)(?=education|skills|summary|$)', resume_text, re.IGNORECASE | re.DOTALL)
    edu_match = re.search(r'education(.+?)(?=experience|skills|summary|$)', resume_text, re.IGNORECASE | re.DOTALL)
    skills_match = re.search(r'skills(.+?)(?=experience|education|summary|$)', resume_text, re.IGNORECASE | re.DOTALL)
    summary_match = re.search(r'summary(.+?)(?=experience|education|skills|$)', resume_text, re.IGNORECASE | re.DOTALL)
    
    if exp_match: sections['experience'] = clean_text(exp_match.group(1))
    if edu_match: sections['education'] = clean_text(edu_match.group(1))
    if skills_match: sections['skills'] = clean_text(skills_match.group(1))
    if summary_match: sections['summary'] = clean_text(summary_match.group(1))
    
    return sections

def reduce_dimensionality(texts, n_components=100):
    vectorizer = TfidfVectorizer(max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    svd = TruncatedSVD(n_components=n_components)
    reduced_matrix = svd.fit_transform(tfidf_matrix)
    
    return reduced_matrix, svd, vectorizer