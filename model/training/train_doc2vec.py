# Coding by Samitha Randika | https://www.linkedin.com/in/samitha-randika-edirisinghe-b3a68a2b6 #
import argparse
import logging
import os
import sys
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import numpy as np

# Configure logging
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', 
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('doc2vec_training.log')
    ]
)

def train_doc2vec_model(documents_path, output_path, vector_size=200, window=5, min_count=5, epochs=20):
    """Train a Doc2Vec model for document embeddings"""
    
    # Read documents
    try:
        with open(documents_path, 'r', encoding='utf-8') as f:
            documents = [line.strip() for line in f if line.strip()]
    except Exception as e:
        logging.error(f"Error reading documents: {str(e)}")
        return False
    
    # Prepare tagged documents
    tagged_data = []
    for i, doc in enumerate(documents):
        try:
            tokens = tokenize_text(doc)
            tagged_data.append(TaggedDocument(words=tokens, tags=[str(i)]))
        except Exception as e:
            logging.warning(f"Skipping document {i}: {str(e)}")
    
    if not tagged_data:
        logging.error("No valid documents for training")
        return False
    
    # Train model
    try:
        model = Doc2Vec(
            documents=tagged_data,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=os.cpu_count(),
            epochs=epochs,
            dm=1,  # Use PV-DM (Distributed Memory) mode
            dbow_words=0  # Skip training word vectors in DBOW mode
        )
        
        # Save model
        model.save(output_path)
        logging.info(f"Doc2Vec model saved to {output_path}")
        
        # Test the model
        test_phrases = [
            "software engineer python java",
            "data scientist machine learning",
            "cloud engineer aws docker"
        ]
        
        for phrase in test_phrases:
            tokens = tokenize_text(phrase)
            vec = model.infer_vector(tokens)
            similar = model.dv.most_similar([vec], topn=3)
            logging.info(f"Similar to '{phrase}': {similar}")
        
        return True
    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        return False