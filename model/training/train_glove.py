# Coding by Samitha Randika | https://www.linkedin.com/in/samitha-randika-edirisinghe-b3a68a2b6 #
import argparse
import logging
import os
import sys
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

# Configure logging
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', 
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('glove_training.log')
    ]
)

def create_default_corpus(output_path, num_sentences=1000):
    """Create minimal corpus if none exists"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for _ in range(num_sentences):
            f.write("software engineering machine learning data science cloud computing devops ")
            f.write("python java javascript react aws docker kubernetes\n")
    logging.warning(f"Created minimal corpus at {output_path}")

def train_glove_model(corpus_path, output_path, vector_size=100, window=5, min_count=5, epochs=10):
    """Train a Word2Vec model (GloVe-like embeddings)"""
    
    # Handle missing corpus
    if not os.path.exists(corpus_path):
        logging.warning(f"No corpus found at {corpus_path}")
        create_default_corpus(corpus_path)
    
    try:
        # Read corpus
        sentences = LineSentence(corpus_path)
        
        # Train model
        model = Word2Vec(
            sentences=sentences,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=os.cpu_count(),
            epochs=epochs
        )
        
        # Save model
        model.wv.save(output_path)
        logging.info(f"GloVe model saved to {output_path}")
        return True
    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train GloVe-like word embeddings')
    parser.add_argument('--corpus_path', type=str, 
                        default='data/corpus.txt',
                        help='Path to corpus text file')
    parser.add_argument('--output_path', type=str, 
                        default='trained_models/glove.model',
                        help='Output path for trained model')
    parser.add_argument('--vector_size', type=int, default=100,
                        help='Embedding dimension size')
    parser.add_argument('--window', type=int, default=5,
                        help='Context window size')
    parser.add_argument('--min_count', type=int, default=5,
                        help='Minimum word frequency')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    
    args = parser.parse_args()
    
    # Create output directory if not exists
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Run training
    success = train_glove_model(
        corpus_path=args.corpus_path,
        output_path=args.output_path,
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count,
        epochs=args.epochs
    )
    
    sys.exit(0 if success else 1)