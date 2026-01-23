"""
Content-Based Filtering Module
Implements TF-IDF vectorization and cosine similarity on product ingredients
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Optional
import pickle


class ContentBasedRecommender:
    """
    Content-based recommender using TF-IDF on product ingredients.
    """
    
    def __init__(self, max_features: int = 500, ngram_range: Tuple[int, int] = (1, 2)):
        """
        Initialize the content-based recommender.
        
        Args:
            max_features: Maximum number of features for TF-IDF
            ngram_range: N-gram range for TF-IDF
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english'
        )
        self.tfidf_matrix = None
        self.similarity_matrix = None
        self.products_df = None
        
    def fit(self, products_df: pd.DataFrame, ingredients_column: str = 'ingredients_clean'):
        """
        Fit the TF-IDF vectorizer on product ingredients.
        
        Args:
            products_df: DataFrame with product information
            ingredients_column: Name of the column containing ingredients
        """
        print("Training content-based model...")
        self.products_df = products_df.copy()
        
        # Create TF-IDF matrix
        self.tfidf_matrix = self.vectorizer.fit_transform(
            products_df[ingredients_column].fillna('')
        )
        
        print(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")
        
        # Compute similarity matrix
        print("Computing cosine similarity matrix...")
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        
        print("Content-based model training complete!")
        
    def get_similar_products(self, product_id: str, n_recommendations: int = 10) -> pd.DataFrame:
        """
        Get similar products based on content similarity.
        
        Args:
            product_id: ID of the reference product
            n_recommendations: Number of recommendations to return
            
        Returns:
            DataFrame with recommended products and similarity scores
        """
        if self.similarity_matrix is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Find product index
        try:
            idx = self.products_df[self.products_df['product_id'] == product_id].index[0]
        except IndexError:
            print(f"Product {product_id} not found")
            return pd.DataFrame()
        
        # Get similarity scores
        sim_scores = list(enumerate(self.similarity_matrix[idx]))
        
        # Sort by similarity (excluding the product itself)
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n_recommendations+1]
        
        # Get product indices
        product_indices = [i[0] for i in sim_scores]
        similarity_scores = [i[1] for i in sim_scores]
        
        # Create result DataFrame
        recommendations = self.products_df.iloc[product_indices].copy()
        recommendations['cb_score'] = similarity_scores
        
        return recommendations
    
    def get_recommendations_by_skin_type(self, skin_type: str, n_recommendations: int = 10, 
                                          category: Optional[str] = None) -> pd.DataFrame:
        """
        Get product recommendations based on skin type (cold start scenario).
        
        Args:
            skin_type: Target skin type
            n_recommendations: Number of recommendations
            category: Optional product category to filter by (tertiary_category)
            
        Returns:
            DataFrame with recommended products
        """
        if self.products_df is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Start with all products
        filtered = self.products_df.copy()
        
        # Filter by category if specified
        if category and category != "All":
            filtered = filtered[filtered['tertiary_category'] == category]
        
        # Filter products by rating and reviews
        quality_filtered = filtered[
            (filtered['rating'] >= 4.0) & 
            (filtered['reviews'] >= 10)
        ].copy()
        
        # If no products match quality filter, use all filtered products
        if len(quality_filtered) == 0:
            quality_filtered = filtered.copy()
        
        # If still no products (category has none), return empty
        if len(quality_filtered) == 0:
            return pd.DataFrame()
        
        # Sort by popularity (rating * log(reviews + 1))
        quality_filtered['popularity_score'] = (
            quality_filtered['rating'] * np.log1p(quality_filtered['reviews'])
        )
        
        # Get top products
        recommendations = quality_filtered.nlargest(n_recommendations, 'popularity_score')
        recommendations['cb_score'] = recommendations['popularity_score'] / recommendations['popularity_score'].max()
        
        return recommendations
    
    def save_model(self, vectorizer_path: str, matrix_path: str, similarity_path: str):
        """
        Save the trained model components.
        
        Args:
            vectorizer_path: Path to save the vectorizer
            matrix_path: Path to save the TF-IDF matrix
            similarity_path: Path to save the similarity matrix
        """
        print("Saving content-based model...")
        
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        with open(matrix_path, 'wb') as f:
            pickle.dump(self.tfidf_matrix, f)
        
        with open(similarity_path, 'wb') as f:
            pickle.dump(self.similarity_matrix, f)
        
        print("Content-based model saved!")
    
    def load_model(self, vectorizer_path: str, matrix_path: str, 
                   similarity_path: str, products_df: pd.DataFrame):
        """
        Load a trained model.
        
        Args:
            vectorizer_path: Path to the saved vectorizer
            matrix_path: Path to the saved TF-IDF matrix
            similarity_path: Path to the saved similarity matrix
            products_df: Products DataFrame
        """
        print("Loading content-based model...")
        
        with open(vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)
        
        with open(matrix_path, 'rb') as f:
            self.tfidf_matrix = pickle.load(f)
        
        with open(similarity_path, 'rb') as f:
            self.similarity_matrix = pickle.load(f)
        
        self.products_df = products_df
        
        print("Content-based model loaded!")
