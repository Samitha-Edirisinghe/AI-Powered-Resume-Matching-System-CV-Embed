import os
import random
from utils.text_processing import clean_text

def create_glove_corpus(output_path, num_sentences=10000):
    """Create sample corpus data for GloVe training"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Sample job-related phrases
    phrases = [
        "software engineer", "web development", "python programming", 
        "machine learning", "data analysis", "cloud computing",
        "devops practices", "system administration", "network security",
        "frontend development", "backend systems", "database management",
        "artificial intelligence", "deep learning", "natural language processing",
        "cloud infrastructure", "containerization", "continuous integration",
        "agile methodology", "scrum framework", "version control systems",
        "restful apis", "microservices architecture", "test driven development"
    ]
    
    # Generate sample sentences
    with open(output_path, 'w', encoding='utf-8') as f:
        for _ in range(num_sentences):
            # Create a sentence with 3-8 random phrases
            sentence = ' '.join(random.sample(phrases, random.randint(3, 8)))
            f.write(clean_text(sentence) + "\n")
    
    print(f"Created GloVe corpus at {output_path} with {num_sentences} sentences")

if __name__ == "__main__":
    create_glove_corpus("../data/corpus/corpus.txt")
