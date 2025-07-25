import pandas as pd
import os
import random

def create_sbert_training_data(output_path):
    """Create sample training data for SBERT model"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Sample training data
    data = {
        'resume': [
            'Experienced software engineer with 5 years in Python and Java development',
            'Web developer specializing in JavaScript and React',
            'Data scientist with machine learning expertise',
            'DevOps engineer with cloud infrastructure experience',
            'System administrator with Linux and networking skills'
        ],
        'job_description': [
            'Seeking senior Python developer with Django experience',
            'Frontend developer needed for React applications',
            'Machine learning engineer for AI projects',
            'Cloud infrastructure specialist with AWS knowledge',
            'IT administrator with network security background'
        ],
        'match_score': [85, 90, 75, 80, 95]  # Match scores (0-100)
    }
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Created SBERT training data at {output_path} with {len(df)} records")

if __name__ == "__main__":
    create_sbert_training_data("../data/training/train.csv")