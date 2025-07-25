# Coding by Samitha Randika | https://www.linkedin.com/in/samitha-randika-edirisinghe-b3a68a2b6 #
from gensim.models import Doc2Vec
import numpy as np
import logging
from utils.text_processing import tokenize_text
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default inference parameters
DEFAULT_EPOCHS = 50  # Changed from steps to epochs
DEFAULT_ALPHA = 0.025
DEFAULT_MIN_ALPHA = 0.001

def load_model(model_path):
    try:
        model = Doc2Vec.load(model_path)
        logger.info(f"Loaded Doc2Vec model from {model_path}")
        logger.info(f"Vector size: {model.vector_size}")
        logger.info(f"Vocabulary size: {len(model.wv)}")
        return model
    except Exception as e:
        logger.error(f"Error loading Doc2Vec model: {e}")
        return None

def infer_vector(model, text, epochs=DEFAULT_EPOCHS, alpha=DEFAULT_ALPHA, min_alpha=DEFAULT_MIN_ALPHA):
    """Infer vector with proper parameters"""
    try:
        tokens = tokenize_text(text)
        if not tokens:
            logger.warning("No tokens after tokenization")
            return np.zeros(model.vector_size)
            
        # Ensure we have at least 5 tokens
        if len(tokens) < 5:
            tokens = tokens * (5 // len(tokens) + 1)
            
        return model.infer_vector(
            tokens,
            epochs=epochs,  # Only accepts 'epochs'
            alpha=alpha,
            min_alpha=min_alpha
        )
    except Exception as e:
        logger.error(f"Vector inference error: {e}")
        return np.zeros(model.vector_size) if model else np.zeros(100)

def calculate_similarity(model, text1, text2):
    try:
        # Infer vectors with proper parameters
        vec1 = infer_vector(model, text1)
        vec2 = infer_vector(model, text2)
        
        if vec1 is None or vec2 is None:
            return 0.0
            
        # Calculate cosine similarity
        similarity = cosine_similarity([vec1], [vec2])[0][0]
        
        # Ensure valid similarity score
        score = max(-1.0, min(1.0, similarity))
        return max(0, min(100, (score + 1) * 50))  # Convert to 0-100 range
    except Exception as e:
        logger.error(f"Doc2Vec similarity calculation error: {e}")
        return 0.0

def get_job_embeddings(model, job_taxonomy):
    embeddings = []
    for job in job_taxonomy:
        embeddings.append(infer_vector(model, job))
    return np.array(embeddings)