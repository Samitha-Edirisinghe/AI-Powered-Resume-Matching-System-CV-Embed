# Coding by Samitha Randika | https://www.linkedin.com/in/samitha-randika-edirisinghe-b3a68a2b6 #
import argparse
import logging
import os
import sys
import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(message)s', 
    datefmt='%Y-%m-%d %H:%M:%S', 
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('sbert_training.log')
    ]
)

def create_default_training_data(output_path):
    """Create minimal training data if none exists"""
    data = {
        'resume': [
            'Experienced software engineer with Python and Java',
            'Data scientist with machine learning expertise',
            'Web developer specializing in React'
        ],
        'job_description': [
            'Seeking software engineer with Python experience',
            'Looking for data scientist with ML background',
            'Hiring frontend developer with React skills'
        ],
        'match_score': [85, 90, 95]
    }
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logging.warning(f"Created minimal training data at {output_path}")
    return df

def train_sbert_model(train_data_path, output_dir, 
                     model_name='all-MiniLM-L6-v2', 
                     num_epochs=10, batch_size=32):
    """Train an SBERT model for semantic similarity"""
    
    # Handle missing training data
    if not os.path.exists(train_data_path):
        logging.warning(f"No training data found at {train_data_path}")
        df = create_default_training_data(train_data_path)
    else:
        try:
            df = pd.read_csv(train_data_path)
            logging.info(f"Loaded training data with {len(df)} records")
        except Exception as e:
            logging.error(f"Error loading training data: {str(e)}")
            df = create_default_training_data(train_data_path)
    
    # Prepare training examples
    train_examples = []
    for _, row in df.iterrows():
        try:
            # Normalize score to 0-1 range
            score = max(0, min(100, float(row['match_score']))) / 100
            train_examples.append(InputExample(
                texts=[str(row['resume']), str(row['job_description'])],
                label=score
            ))
        except Exception as e:
            logging.warning(f"Skipping invalid row: {str(e)}")
    
    if not train_examples:
        logging.error("No valid training examples found!")
        return False
    
    # Initialize model
    model = SentenceTransformer(model_name)
    logging.info(f"Initialized model: {model_name}")
    
    # Data loader
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    train_loss = losses.CosineSimilarityLoss(model)
    
    # Training
    logging.info(f"Starting training for {num_epochs} epochs with {len(train_examples)} examples")
    
    try:
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=num_epochs,
            show_progress_bar=True,
            output_path=output_dir
        )
        logging.info(f"Model successfully saved to {output_dir}")
        return True
    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SBERT model for resume-JD matching')
    parser.add_argument('--train_data', type=str, 
                        default='data/training/train.csv',
                        help='Path to training data CSV')
    parser.add_argument('--output_dir', type=str, 
                        default='trained_models/sbert',
                        help='Output directory for trained model')
    parser.add_argument('--model_name', type=str, 
                        default='all-MiniLM-L6-v2',
                        help='Base SBERT model name')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Training batch size')
    
    args = parser.parse_args()
    
    # Create output directory if not exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run training
    success = train_sbert_model(
        train_data_path=args.train_data,
        output_dir=args.output_dir,
        model_name=args.model_name,
        num_epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    sys.exit(0 if success else 1)