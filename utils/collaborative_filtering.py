"""
Collaborative Filtering Module
Implements TruncatedSVD for matrix factorization on user-item ratings
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
from typing import List
import pickle


class CollaborativeFilteringRecommender:
    """
    Collaborative filtering recommender using TruncatedSVD for matrix factorization.
    """
    
    def __init__(self, n_components: int = 50):
        """
        Initialize the collaborative filtering recommender.
        
        Args:
            n_components: Number of latent factors for SVD
        """
        self.svd_model = TruncatedSVD(n_components=n_components, random_state=42)
        self.user_item_matrix = None
        self.user_factors = None
        self.item_factors = None
        self.product_ids = None
        self.user_ids = None
        self.mean_ratings = None
        
    def fit(self, user_item_matrix: pd.DataFrame):
        """
        Fit the SVD model on user-item matrix.
        
        Args:
            user_item_matrix: DataFrame with users as rows, products as columns
        """
        print("Training collaborative filtering model...")
        
        self.user_item_matrix = user_item_matrix
        self.product_ids = user_item_matrix.columns
        self.user_ids = user_item_matrix.index
        
        # Fill NaN with 0 (no rating)
        matrix_filled = user_item_matrix.fillna(0)
        
        # Calculate mean ratings for normalization
        self.mean_ratings = user_item_matrix.mean(axis=0)
        
        # Normalize by subtracting mean
        matrix_normalized = matrix_filled.subtract(self.mean_ratings, axis=1)
        matrix_normalized = matrix_normalized.fillna(0)
        
        # Convert to sparse matrix for efficiency
        sparse_matrix = csr_matrix(matrix_normalized.values)
        
        # Fit SVD
        print(f"Fitting SVD with {self.svd_model.n_components} components...")
        self.user_factors = self.svd_model.fit_transform(sparse_matrix)
        self.item_factors = self.svd_model.components_.T
        
        print(f"User factors shape: {self.user_factors.shape}")
        print(f"Item factors shape: {self.item_factors.shape}")
        print(f"Explained variance ratio: {self.svd_model.explained_variance_ratio_.sum():.4f}")
        print("Collaborative filtering model training complete!")
        
    def predict_rating(self, user_id: int, product_id: str) -> float:
        """
        Predict rating for a user-product pair.
        
        Args:
            user_id: User ID
            product_id: Product ID
            
        Returns:
            Predicted rating
        """
        try:
            user_idx = self.user_ids.get_loc(user_id)
            product_idx = self.product_ids.get_loc(product_id)
            
            # Predict using matrix factorization
            prediction = np.dot(self.user_factors[user_idx], self.item_factors[product_idx])
            
            # Add back the mean
            prediction += self.mean_ratings[product_id]
            
            # Clip to valid rating range
            return np.clip(prediction, 1, 5)
        except:
            # Return average rating if user or product not found
            return self.mean_ratings.mean()
    
    def get_collaborative_scores(self, product_id: str, n_recommendations: int = 10) -> pd.DataFrame:
        """
        Get products with high collaborative filtering scores.
        Uses item-item similarity based on latent factors.
        
        Args:
            product_id: Reference product ID
            n_recommendations: Number of recommendations
            
        Returns:
            DataFrame with product IDs and CF scores
        """
        try:
            product_idx = self.product_ids.get_loc(product_id)
            
            # Calculate item-item similarity using latent factors
            product_vector = self.item_factors[product_idx].reshape(1, -1)
            similarities = np.dot(self.item_factors, product_vector.T).flatten()
            
            # Get top similar products
            top_indices = np.argsort(similarities)[::-1][1:n_recommendations+1]
            
            # Create result DataFrame
            result = pd.DataFrame({
                'product_id': [self.product_ids[i] for i in top_indices],
                'cf_score': similarities[top_indices]
            })
            
            # Normalize scores to [0, 1]
            if result['cf_score'].max() > 0:
                result['cf_score'] = result['cf_score'] / result['cf_score'].max()
            
            return result
        except:
            # Return empty DataFrame if product not found
            return pd.DataFrame(columns=['product_id', 'cf_score'])
    
    def get_user_recommendations(self, user_id: int, n_recommendations: int = 10) -> pd.DataFrame:
        """
        Get personalized recommendations for a user.
        
        Args:
            user_id: User ID
            n_recommendations: Number of recommendations
            
        Returns:
            DataFrame with recommended products and predicted ratings
        """
        try:
            user_idx = self.user_ids.get_loc(user_id)
            
            # Get user's ratings
            user_ratings = self.user_item_matrix.iloc[user_idx]
            
            # Get unrated products
            unrated_products = user_ratings[user_ratings.isna()].index
            
            if len(unrated_products) == 0:
                return pd.DataFrame()
            
            # Predict ratings for unrated products
            predictions = []
            for product_id in unrated_products:
                pred_rating = self.predict_rating(user_id, product_id)
                predictions.append({
                    'product_id': product_id,
                    'predicted_rating': pred_rating,
                    'cf_score': pred_rating / 5.0  # Normalize to [0, 1]
                })
            
            result = pd.DataFrame(predictions)
            result = result.nlargest(n_recommendations, 'predicted_rating')
            
            return result
        except:
            return pd.DataFrame(columns=['product_id', 'cf_score'])
    
    def save_model(self, model_path: str, matrix_path: str):
        """
        Save the trained model.
        
        Args:
            model_path: Path to save the SVD model
            matrix_path: Path to save the user-item matrix
        """
        print("Saving collaborative filtering model...")
        
        model_data = {
            'svd_model': self.svd_model,
            'user_factors': self.user_factors,
            'item_factors': self.item_factors,
            'product_ids': self.product_ids,
            'user_ids': self.user_ids,
            'mean_ratings': self.mean_ratings
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        with open(matrix_path, 'wb') as f:
            pickle.dump(self.user_item_matrix, f)
        
        print("Collaborative filtering model saved!")
    
    def load_model(self, model_path: str, matrix_path: str):
        """
        Load a trained model.
        
        Args:
            model_path: Path to the saved model
            matrix_path: Path to the saved user-item matrix
        """
        print("Loading collaborative filtering model...")
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.svd_model = model_data['svd_model']
        self.user_factors = model_data['user_factors']
        self.item_factors = model_data['item_factors']
        self.product_ids = model_data['product_ids']
        self.user_ids = model_data['user_ids']
        self.mean_ratings = model_data['mean_ratings']
        
        with open(matrix_path, 'rb') as f:
            self.user_item_matrix = pickle.load(f)
        
        print("Collaborative filtering model loaded!")
