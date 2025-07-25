# Coding by Samitha Randika | https://www.linkedin.com/in/samitha-randika-edirisinghe-b3a68a2b6 #
from datasets import load_dataset
import pandas as pd
import os
import joblib
import logging
from config import DATASETS
from utils.text_processing import clean_text

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_all_datasets():
    all_data = pd.DataFrame()
    
    # Load Hugging Face datasets
    for ds_name in DATASETS:
        if "/" in ds_name:  # HF dataset
            try:
                logger.info(f"Loading Hugging Face dataset: {ds_name}")
                dataset = load_dataset(ds_name)
                
                if 'train' in dataset:
                    df = pd.DataFrame(dataset['train'])
                else:
                    df = pd.DataFrame(dataset)
                
                # Standardize column names
                if 'resume' in df.columns and 'job_description' in df.columns:
                    df = df.rename(columns={'resume': 'text', 'job_description': 'jd_text'})
                    df['type'] = 'resume_jd_pair'
                elif 'resume' in df.columns:
                    df = df.rename(columns={'resume': 'text'})
                    df['type'] = 'resume'
                elif 'job_description' in df.columns:
                    df = df.rename(columns={'job_description': 'text'})
                    df['type'] = 'job_description'
                
                all_data = pd.concat([all_data, df], ignore_index=True)
                logger.info(f"Loaded {len(df)} records from {ds_name}")
            except Exception as e:
                logger.error(f"Error loading {ds_name}: {e}")
    
    # Load Kaggle datasets
    kaggle_path = "data/kaggle"
    if os.path.exists(kaggle_path):
        for file in os.listdir(kaggle_path):
            if file.endswith('.csv'):
                try:
                    file_path = os.path.join(kaggle_path, file)
                    logger.info(f"Loading Kaggle dataset: {file}")
                    df = pd.read_csv(file_path)
                    
                    # Standardize columns
                    if 'Resume' in df.columns:
                        df = df.rename(columns={'Resume': 'text'})
                        df['type'] = 'resume'
                    elif 'Description' in df.columns:
                        df = df.rename(columns={'Description': 'text'})
                        df['type'] = 'job_description'
                    elif 'job_desc' in df.columns:
                        df = df.rename(columns={'job_desc': 'text'})
                        df['type'] = 'job_description'
                    
                    all_data = pd.concat([all_data, df], ignore_index=True)
                    logger.info(f"Loaded {len(df)} records from {file}")
                except Exception as e:
                    logger.error(f"Error loading Kaggle dataset {file}: {e}")
    
    # Clean text
    if not all_data.empty and 'text' in all_data.columns:
        all_data['cleaned_text'] = all_data['text'].apply(clean_text)
    
    return all_data