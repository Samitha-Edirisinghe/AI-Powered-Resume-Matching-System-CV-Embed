# Coding by Samitha Randika | https://www.linkedin.com/in/samitha-randika-edirisinghe-b3a68a2b6 #
import joblib
import os
import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from config import MODEL_CONFIG
from model.inference import sbert_inference, glove_inference, doc2vec_inference
from utils.text_processing import tokenize_text

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize models
MODELS = {
    "sbert": sbert_inference.load_model(MODEL_CONFIG['sbert_path']),
    "glove": glove_inference.load_model(MODEL_CONFIG['glove_path']),
    "doc2vec": doc2vec_inference.load_model(MODEL_CONFIG['doc2vec_path'])
}

# Job embeddings cache
JOB_EMBEDDINGS = {}

def get_model_embedding(model_type, text):
    """Get embedding based on model type"""
    if model_type not in MODELS or MODELS[model_type] is None:
        return None
    
    try:
        if model_type == "sbert":
            return MODELS[model_type].encode([text])[0]
        elif model_type == "glove":
            tokens = tokenize_text(text)
            return glove_inference.average_embeddings(MODELS[model_type], tokens)
        elif model_type == "doc2vec":
            # CORRECTED: Changed 'steps' to 'epochs'
            return doc2vec_inference.infer_vector(
                MODELS[model_type], 
                text,
                epochs=50,  # Fixed parameter name
                alpha=0.025
            )
    except Exception as e:
        logger.error(f"Error getting embedding for {model_type}: {e}")
        return None

def calculate_similarity(resume_text, jd_text, model_type=None):
    """Calculate similarity between resume and JD"""
    if not model_type:
        model_type = MODEL_CONFIG['active_model']
    
    if model_type not in MODELS or MODELS[model_type] is None:
        logger.error(f"Model {model_type} not available")
        return 0.0
    
    try:
        if model_type == "sbert":
            return sbert_inference.calculate_similarity(MODELS[model_type], resume_text, jd_text)
        elif model_type == "glove":
            return glove_inference.calculate_text_similarity(MODELS[model_type], resume_text, jd_text)
        elif model_type == "doc2vec":
            # Use the improved Doc2Vec similarity function
            return doc2vec_inference.calculate_similarity(MODELS[model_type], resume_text, jd_text)
        else:
            logger.error(f"Unknown model type: {model_type}")
            return 0.0
    except Exception as e:
        logger.error(f"Similarity calculation error: {e}")
        return 0.0
    
def get_top_job_matches(resume_text, model_type=None, top_n=5):
    """Get top job matches for resume"""
    if not model_type:
        model_type = MODEL_CONFIG['active_model']
    
    if model_type not in MODELS or MODELS[model_type] is None:
        logger.error(f"Model {model_type} not available")
        return []
    
    try:
        # Generate resume embedding
        resume_embedding = get_model_embedding(model_type, resume_text)
        if resume_embedding is None:
            return []
        
        # Get job embeddings (cached)
        if model_type not in JOB_EMBEDDINGS:
            if model_type == "sbert":
                JOB_EMBEDDINGS[model_type] = sbert_inference.get_job_embeddings(
                    MODELS[model_type], MODEL_CONFIG['job_taxonomy'])
            elif model_type == "glove":
                JOB_EMBEDDINGS[model_type] = glove_inference.get_job_embeddings(
                    MODELS[model_type], MODEL_CONFIG['job_taxonomy'])
            elif model_type == "doc2vec":
                JOB_EMBEDDINGS[model_type] = doc2vec_inference.get_job_embeddings(
                    MODELS[model_type], MODEL_CONFIG['job_taxonomy'])
        
        # Calculate similarities
        similarities = cosine_similarity([resume_embedding], JOB_EMBEDDINGS[model_type])[0]
        
        # Get top matches
        top_indices = np.argsort(similarities)[-top_n:][::-1]
        return [(MODEL_CONFIG['job_taxonomy'][i], round(similarities[i] * 100, 2)) 
                for i in top_indices]

    
    except Exception as e:
        logger.error(f"Top job matches error: {e}")
        return []