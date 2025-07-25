# Coding by Samitha Randika | https://www.linkedin.com/in/samitha-randika-edirisinghe-b3a68a2b6 #
import re
import os
from PyPDF2 import PdfReader
from docx import Document
from utils.text_processing import clean_text

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PdfReader(file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"Error processing PDF: {e}")
    return text

def extract_text_from_docx(docx_path):
    text = ""
    try:
        doc = Document(docx_path)
        for para in doc.paragraphs:
            if para.text.strip():
                text += para.text + "\n"
    except Exception as e:
        print(f"Error processing DOCX: {e}")
    return text

def process_resume(file_path):
    if file_path.endswith('.pdf'):
        text = extract_text_from_pdf(file_path)
    elif file_path.endswith('.docx'):
        text = extract_text_from_docx(file_path)
    else:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
        except:
            with open(file_path, 'r', encoding='latin-1') as file:
                text = file.read()
    return clean_text(text)

def process_jd(file_path):
    return process_resume(file_path)