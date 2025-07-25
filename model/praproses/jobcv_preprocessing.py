# Coding by Samitha Randika | https://www.linkedin.com/in/samitha-randika-edirisinghe-b3a68a2b6 #
import pandas as pd
from utils.text_processing import clean_text, tokenize_text
from sklearn.model_selection import train_test_split

def preprocess_jobcv_data(file_path):
    # Load data
    df = pd.read_csv(file_path)
    
    # Clean and tokenize
    df['cleaned_resume'] = df['resume'].apply(clean_text)
    df['cleaned_jd'] = df['job_description'].apply(clean_text)
    df['resume_tokens'] = df['cleaned_resume'].apply(tokenize_text)
    df['jd_tokens'] = df['cleaned_jd'].apply(tokenize_text)
    
    # Create labels (simplified example)
    df['match_score'] = df['match_label'].apply(lambda x: 1 if x == 'match' else 0)
    
    # Split data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    return train_df, test_df