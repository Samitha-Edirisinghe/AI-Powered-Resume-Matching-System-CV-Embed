# Coding by Samitha Randika | https://www.linkedin.com/in/samitha-randika-edirisinghe-b3a68a2b6 #
import pandas as pd
import numpy as np
import re
from utils.text_processing import clean_text

def check_data_quality(df):
    # Check for missing values
    missing_values = df.isnull().sum()
    
    # Check text length
    df['text_length'] = df['text'].apply(len)
    min_length = df['text_length'].min()
    max_length = df['text_length'].max()
    avg_length = df['text_length'].mean()
    
    # Check special characters
    def count_special_chars(text):
        return len(re.findall(r'[^\w\s]', text))
    
    df['special_chars'] = df['text'].apply(count_special_chars)
    avg_special_chars = df['special_chars'].mean()
    
    return {
        "missing_values": missing_values.to_dict(),
        "text_length_stats": {
            "min": min_length,
            "max": max_length,
            "average": avg_length
        },
        "avg_special_chars": avg_special_chars
    }

def clean_dataset(df):
    # Remove rows with missing text
    df = df.dropna(subset=['text'])
    
    # Clean text
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['cleaned_text'])
    
    return df