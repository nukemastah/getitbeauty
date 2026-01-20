"""
Configuration file for Skincare Hybrid Recommendation System
"""
import os

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Dataset paths
SEPHORA_PRODUCTS_PATH = os.path.join(DATASET_DIR, 'sephora', 'product_info.csv')
SKINCARE_PRODUCTS_PATH = os.path.join(DATASET_DIR, 'skincare', 'skincare_products_clean.csv')
REVIEWS_PATHS = [
    os.path.join(DATASET_DIR, 'sephora', 'reviews_0-250_masked.csv'),
    os.path.join(DATASET_DIR, 'sephora', 'reviews_250-500_masked.csv'),
    os.path.join(DATASET_DIR, 'sephora', 'reviews_500-750_masked.csv'),
    os.path.join(DATASET_DIR, 'sephora', 'reviews_750-1250_masked.csv'),
    os.path.join(DATASET_DIR, 'sephora', 'reviews_1250-end_masked.csv'),
]

# Model save paths
TFIDF_VECTORIZER_PATH = os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl')
TFIDF_MATRIX_PATH = os.path.join(MODELS_DIR, 'tfidf_matrix.pkl')
SVD_MODEL_PATH = os.path.join(MODELS_DIR, 'svd_model.pkl')
USER_ITEM_MATRIX_PATH = os.path.join(MODELS_DIR, 'user_item_matrix.pkl')
PRODUCTS_DATA_PATH = os.path.join(MODELS_DIR, 'products_data.pkl')
SIMILARITY_MATRIX_PATH = os.path.join(MODELS_DIR, 'similarity_matrix.pkl')

# Model parameters
TFIDF_MAX_FEATURES = 500
TFIDF_NGRAM_RANGE = (1, 2)
SVD_N_COMPONENTS = 50
ALPHA_DEFAULT = 0.5  # Weight for content-based vs collaborative filtering

# Skin types
SKIN_TYPES = ['Combination', 'Dry', 'Normal', 'Oily', 'Sensitive']

# Recommendation settings
DEFAULT_N_RECOMMENDATIONS = 10
FUZZY_SEARCH_THRESHOLD = 70  # Minimum score for fuzzy matching
