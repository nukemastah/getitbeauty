"""
Hybrid Recommender System
Combines Content-Based and Collaborative Filtering with weighted scoring
"""
import pandas as pd
import numpy as np
from typing import Optional
from .content_based import ContentBasedRecommender
from .collaborative_filtering import CollaborativeFilteringRecommender


class HybridRecommender:
    """
    Hybrid recommender that combines content-based and collaborative filtering.
    
    Final Score = (alpha × CB_Score) + ((1 - alpha) × CF_Score)
    """
    
    def __init__(self, 
                 cb_recommender: ContentBasedRecommender,
                 cf_recommender: CollaborativeFilteringRecommender,
                 alpha: float = 0.5):
        """
        Initialize hybrid recommender.
        
        Args:
            cb_recommender: Trained content-based recommender
            cf_recommender: Trained collaborative filtering recommender
            alpha: Weight for content-based (0-1). Higher = more content-based
        """
        self.cb_recommender = cb_recommender
        self.cf_recommender = cf_recommender
        self.alpha = alpha
        
    def get_hybrid_recommendations(self,
                                   product_id: str,
                                   user_id: Optional[int] = None,
                                   skin_type: Optional[str] = None,
                                   category: Optional[str] = None,
                                   n_recommendations: int = 10,
                                   alpha: Optional[float] = None) -> pd.DataFrame:
        """
        Get hybrid recommendations combining CB and CF.
        
        Args:
            product_id: Reference product ID (for CB similarity)
            user_id: User ID (for CF personalization). If None, uses CB only
            skin_type: Skin type (for cold start). Used if no product_id or user_id
            category: Product category to filter recommendations (tertiary_category)
            n_recommendations: Number of recommendations to return
            alpha: Custom alpha value (overrides default)
            
        Returns:
            DataFrame with recommended products and hybrid scores
        """
        if alpha is None:
            alpha = self.alpha
        
        # Cold start: No user history and no product selected
        if product_id is None and user_id is None:
            if skin_type:
                print(f"Cold start: Recommending based on skin type ({skin_type}) and category ({category})...")
                recommendations = self.cb_recommender.get_recommendations_by_skin_type(
                    skin_type, n_recommendations, category=category
                )
                recommendations['hybrid_score'] = recommendations['cb_score']
                recommendations['recommendation_type'] = 'cold_start'
                return recommendations
            else:
                print("No input provided. Cannot generate recommendations.")
                return pd.DataFrame()
        
        # Get content-based recommendations
        if product_id:
            cb_recs = self.cb_recommender.get_similar_products(
                product_id, n_recommendations * 3  # Get more to merge with CF
            )
        else:
            cb_recs = pd.DataFrame()
        
        # Get collaborative filtering recommendations
        if user_id is not None:
            # Try to get CF scores
            if product_id:
                cf_recs = self.cf_recommender.get_collaborative_scores(
                    product_id, n_recommendations * 3
                )
            else:
                cf_recs = self.cf_recommender.get_user_recommendations(
                    user_id, n_recommendations * 3
                )
        else:
            cf_recs = pd.DataFrame()
        
        # Merge recommendations
        if len(cb_recs) > 0 and len(cf_recs) > 0:
            # Both CB and CF available - true hybrid
            merged = cb_recs.merge(
                cf_recs[['product_id', 'cf_score']], 
                on='product_id', 
                how='outer'
            )
            
            # Fill missing scores with 0
            merged['cb_score'] = merged['cb_score'].fillna(0)
            merged['cf_score'] = merged['cf_score'].fillna(0)
            
            # Calculate hybrid score
            merged['hybrid_score'] = (alpha * merged['cb_score']) + ((1 - alpha) * merged['cf_score'])
            merged['recommendation_type'] = 'hybrid'
            
        elif len(cb_recs) > 0:
            # Only CB available
            merged = cb_recs.copy()
            merged['cf_score'] = 0
            merged['hybrid_score'] = merged['cb_score']
            merged['recommendation_type'] = 'content_based'
            
        elif len(cf_recs) > 0:
            # Only CF available
            merged = cf_recs.copy()
            merged['cb_score'] = 0
            merged['hybrid_score'] = merged['cf_score']
            merged['recommendation_type'] = 'collaborative'
            
        else:
            # No recommendations found
            return pd.DataFrame()
        
        # Sort by hybrid score and return top N
        merged = merged.sort_values('hybrid_score', ascending=False)
        merged = merged.head(n_recommendations)
        
        return merged
    
    def search_product(self, query: str, products_df: pd.DataFrame, 
                      threshold: int = 70) -> pd.DataFrame:
        """
        Search for products using fuzzy matching.
        
        Args:
            query: Search query
            products_df: Products DataFrame
            threshold: Minimum fuzzy matching score (0-100)
            
        Returns:
            DataFrame with matching products and scores
        """
        from thefuzz import fuzz, process
        
        # Create searchable strings
        products_df['search_string'] = (
            products_df['product_name'].fillna('') + ' ' + 
            products_df['brand_name'].fillna('')
        )
        
        # Perform fuzzy search
        matches = process.extract(
            query, 
            products_df['search_string'].to_dict(), 
            limit=10,
            scorer=fuzz.token_sort_ratio
        )
        
        # Filter by threshold
        matches = [(match, score, idx) for match, score, idx in matches if score >= threshold]
        
        if not matches:
            return pd.DataFrame()
        
        # Get matching products
        indices = [idx for _, _, idx in matches]
        scores = [score for _, score, _ in matches]
        
        results = products_df.iloc[indices].copy()
        results['match_score'] = scores
        
        return results
    
    def get_recommendations_with_search(self,
                                       search_query: Optional[str] = None,
                                       user_id: Optional[int] = None,
                                       skin_type: Optional[str] = None,
                                       n_recommendations: int = 10,
                                       alpha: Optional[float] = None,
                                       products_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Get recommendations with optional product search.
        
        Args:
            search_query: Product search query (fuzzy matching)
            user_id: User ID for personalization
            skin_type: Skin type for cold start
            n_recommendations: Number of recommendations
            alpha: Custom alpha value
            products_df: Products DataFrame for search
            
        Returns:
            DataFrame with recommendations
        """
        product_id = None
        
        # Search for product if query provided
        if search_query and products_df is not None:
            search_results = self.search_product(search_query, products_df)
            
            if len(search_results) > 0:
                product_id = search_results.iloc[0]['product_id']
                print(f"Found product: {search_results.iloc[0]['product_name']}")
            else:
                print(f"No products found matching '{search_query}'")
        
        # Get recommendations
        recommendations = self.get_hybrid_recommendations(
            product_id=product_id,
            user_id=user_id,
            skin_type=skin_type,
            n_recommendations=n_recommendations,
            alpha=alpha
        )
        
        return recommendations
