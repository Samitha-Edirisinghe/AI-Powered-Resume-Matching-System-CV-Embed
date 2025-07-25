import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove special characters
    text = re.sub(r'[^\w\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_text(text, keep_numbers=False):
    """Tokenize text with optional number preservation"""
    text = clean_text(text)
    
    # Handle special cases for job titles
    special_cases = {
        "system administrator": "system_administrator",
        "database administrator": "database_administrator",
        "web developer": "web_developer",
        "security analyst": "security_analyst",
        "data scientist": "data_scientist",
        "devops engineer": "devops_engineer",
        "cloud engineer": "cloud_engineer",
        "machine learning engineer": "machine_learning_engineer",
        "software engineer": "software_engineer"
    }
    
    for phrase, replacement in special_cases.items():
        text = text.replace(phrase, replacement)
    
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    
    # Filter tokens
    filtered = []
    for word in tokens:
        if word in stop_words:
            continue
        if not keep_numbers and word.isdigit():
            continue
        if len(word) < 2:
            continue
        filtered.append(word)
    
    return filtered