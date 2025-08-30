# config.py
import sys
import logging
import qbreader
import spacy
from sentence_transformers import SentenceTransformer

# ----- CONFIGURATION FOR CLUSTERING -----
DEFAULT_THRESHOLD = 1.3           # Default clustering threshold (from dendrogram)
MAX_PLACEMENT = 100               # Placement is a percentage (for display only)

# New parameters for quality formula:
GOOD_DISTANCE = 0.40              # avg_distance <= this is considered ideal (dq = 1.0)
BAD_DISTANCE = 0.50               # avg_distance >= this is considered useless (dq = 0.0)
ALPHA_DISTANCE = 0.6              # Weight for the distance factor in quality
BETA_SIZE = 0.4                 # Weight for the normalized cluster size in quality

# ----- LOGGING SETUP -----
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    handlers=[
                        logging.FileHandler("errors.log"),
                        logging.StreamHandler(sys.stdout)
                    ])

# ----- LOAD MODELS & INITIALIZE API -----
try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    logging.error(f"Failed to load spaCy model: {e}")
    sys.exit(1)

try:
    api = qbreader.Sync()
except Exception as e:
    logging.error(f"Failed to initialize qbreader API: {e}")
    sys.exit(1)

try:
    sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
except Exception as e:
    logging.error(f"Failed to load SentenceTransformer model: {e}")
    sys.exit(1)
