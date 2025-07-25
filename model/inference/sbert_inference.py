# Coding by Samitha Randika | https://www.linkedin.com/in/samitha-randika-edirisinghe-b3a68a2b6 #
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model(model_path):
    try:
        model = SentenceTransformer(model_path)
        logger.info(f"Loaded SBERT model from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading SBERT model: {e}")
        return None

def calculate_similarity(model, text1, text2):
    try:
        embeddings = model.encode([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return max(0, min(1, similarity)) * 100  # Convert to percentage
    except Exception as e:
        logger.error(f"SBERT similarity calculation error: {e}")
        return 0.0

def get_job_embeddings(model, job_taxonomy):
    return model.encode(job_taxonomy)