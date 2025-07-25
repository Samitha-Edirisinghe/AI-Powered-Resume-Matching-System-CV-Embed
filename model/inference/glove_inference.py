# Coding by Samitha Randika | https://www.linkedin.com/in/samitha-randika-edirisinghe-b3a68a2b6 #
import numpy as np
from gensim.models import KeyedVectors
from utils.text_processing import tokenize_text
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model(model_path):
    try:
        model = KeyedVectors.load(model_path)
        logger.info(f"Loaded GloVe model from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading GloVe model: {e}")
        return None

def average_embeddings(model, tokens):
    valid_embeddings = [model[word] for word in tokens if word in model]
    if valid_embeddings:
        return np.mean(valid_embeddings, axis=0)
    return None

def calculate_similarity(model, text1, text2):
    try:
        tokens1 = tokenize_text(text1)
        tokens2 = tokenize_text(text2)
        
        vec1 = average_embeddings(model, tokens1)
        vec2 = average_embeddings(model, tokens2)
        
        if vec1 is not None and vec2 is not None:
            # Handle zero vectors to avoid division by zero
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
                
            similarity = np.dot(vec1, vec2) / (norm1 * norm2)
            return max(0, min(1, similarity)) * 100  # Convert to percentage
        return 0.0
    except Exception as e:
        logger.error(f"Similarity calculation error: {e}")
        return 0.0

# ADD THIS FUNCTION TO FIX THE ERROR
def calculate_text_similarity(model, text1, text2):
    """Calculate similarity between two text strings"""
    return calculate_similarity(model, text1, text2)

def get_job_embeddings(model, job_taxonomy):
    embeddings = []
    vector_size = model.vector_size
    for job in job_taxonomy:
        tokens = tokenize_text(job)
        emb = average_embeddings(model, tokens)
        if emb is None:
            # Use zero vector if embedding can't be computed
            emb = np.zeros(vector_size)
        embeddings.append(emb)
    return np.array(embeddings)