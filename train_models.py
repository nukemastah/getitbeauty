"""
Model Training Script
Trains and saves Content-Based and Collaborative Filtering models
"""
import sys
import os
import pickle

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from utils.data_preprocessing import preprocess_all_data
from utils.content_based import ContentBasedRecommender
from utils.collaborative_filtering import CollaborativeFilteringRecommender


def main():
    """
    Main training pipeline.
    """
    print("=" * 60)
    print("Skincare Hybrid Recommender - Model Training")
    print("=" * 60)
    
    # Create models directory if not exists
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    
    # Step 1: Load and preprocess data
    print("\n[Step 1/4] Loading and preprocessing data...")
    print("-" * 60)
    
    products_df, reviews_df, user_item_matrix = preprocess_all_data(
        sephora_path=config.SEPHORA_PRODUCTS_PATH,
        skincare_path=config.SKINCARE_PRODUCTS_PATH,
        reviews_paths=config.REVIEWS_PATHS,
        sample_reviews=50000  # Sample for faster training, remove for full data
    )
    
    # Save products data
    print("\nSaving products data...")
    with open(config.PRODUCTS_DATA_PATH, 'wb') as f:
        pickle.dump(products_df, f)
    print(f"Products data saved to {config.PRODUCTS_DATA_PATH}")
    
    # Step 2: Train Content-Based model
    print("\n[Step 2/4] Training Content-Based Filtering model...")
    print("-" * 60)
    
    cb_recommender = ContentBasedRecommender(
        max_features=config.TFIDF_MAX_FEATURES,
        ngram_range=config.TFIDF_NGRAM_RANGE
    )
    
    cb_recommender.fit(products_df, ingredients_column='ingredients_clean')
    
    # Save CB model
    cb_recommender.save_model(
        vectorizer_path=config.TFIDF_VECTORIZER_PATH,
        matrix_path=config.TFIDF_MATRIX_PATH,
        similarity_path=config.SIMILARITY_MATRIX_PATH
    )
    
    # Step 3: Train Collaborative Filtering model
    print("\n[Step 3/4] Training Collaborative Filtering model...")
    print("-" * 60)
    
    if len(user_item_matrix) > 0:
        cf_recommender = CollaborativeFilteringRecommender(
            n_components=config.SVD_N_COMPONENTS
        )
        
        cf_recommender.fit(user_item_matrix)
        
        # Save CF model
        cf_recommender.save_model(
            model_path=config.SVD_MODEL_PATH,
            matrix_path=config.USER_ITEM_MATRIX_PATH
        )
    else:
        print("Warning: No reviews data available. Skipping CF training.")
        print("System will use Content-Based filtering only.")
    
    # Step 4: Test recommendations
    print("\n[Step 4/4] Testing recommendations...")
    print("-" * 60)
    
    # Test CB recommendations
    test_product_id = products_df.iloc[0]['product_id']
    test_product_name = products_df.iloc[0]['product_name']
    
    print(f"\nTest Product: {test_product_name}")
    print(f"Product ID: {test_product_id}")
    
    cb_recs = cb_recommender.get_similar_products(test_product_id, n_recommendations=5)
    print("\nContent-Based Recommendations:")
    print(cb_recs[['product_name', 'brand_name', 'cb_score']].to_string(index=False))
    
    # Test CF recommendations if available
    if len(user_item_matrix) > 0:
        cf_recs = cf_recommender.get_collaborative_scores(test_product_id, n_recommendations=5)
        if len(cf_recs) > 0:
            cf_recs = cf_recs.merge(products_df[['product_id', 'product_name', 'brand_name']], on='product_id')
            print("\nCollaborative Filtering Recommendations:")
            print(cf_recs[['product_name', 'brand_name', 'cf_score']].to_string(index=False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nModels saved to: {config.MODELS_DIR}")
    print(f"Total products: {len(products_df)}")
    print(f"Total reviews: {len(reviews_df)}")
    print(f"User-item matrix shape: {user_item_matrix.shape if len(user_item_matrix) > 0 else 'N/A'}")
    
    print("\nSaved files:")
    print(f"  - {os.path.basename(config.PRODUCTS_DATA_PATH)}")
    print(f"  - {os.path.basename(config.TFIDF_VECTORIZER_PATH)}")
    print(f"  - {os.path.basename(config.TFIDF_MATRIX_PATH)}")
    print(f"  - {os.path.basename(config.SIMILARITY_MATRIX_PATH)}")
    if len(user_item_matrix) > 0:
        print(f"  - {os.path.basename(config.SVD_MODEL_PATH)}")
        print(f"  - {os.path.basename(config.USER_ITEM_MATRIX_PATH)}")
    
    print("\n✓ You can now run the Streamlit app with: streamlit run app.py")
    

if __name__ == "__main__":
    main()
