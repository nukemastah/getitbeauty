"""
GetItBeauty - Skincare Recommendation System
"""
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from utils.content_based import ContentBasedRecommender
from utils.collaborative_filtering import CollaborativeFilteringRecommender
from utils.hybrid_recommender import HybridRecommender


# Page configuration
st.set_page_config(
    page_title="GetItBeauty",
    page_icon="✨",
    layout="wide"
)

# Custom CSS to hide default top padding and footer
st.markdown("""
    <style>
        .block-container {
            padding-top: 2rem;
        }
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """Load all trained models and data."""
    try:
        # Load products data
        with open(config.PRODUCTS_DATA_PATH, 'rb') as f:
            products_df = pickle.load(f)
        
        # Load Content-Based model
        cb_recommender = ContentBasedRecommender()
        cb_recommender.load_model(
            vectorizer_path=config.TFIDF_VECTORIZER_PATH,
            matrix_path=config.TFIDF_MATRIX_PATH,
            similarity_path=config.SIMILARITY_MATRIX_PATH,
            products_df=products_df
        )
        
        # Load Collaborative Filtering model if available
        cf_recommender = None
        if os.path.exists(config.SVD_MODEL_PATH):
            cf_recommender = CollaborativeFilteringRecommender()
            cf_recommender.load_model(
                model_path=config.SVD_MODEL_PATH,
                matrix_path=config.USER_ITEM_MATRIX_PATH
            )
        
        return products_df, cb_recommender, cf_recommender
    
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.info("Wait for model training to complete...")
        st.stop()


def display_product_card(product, score=None, score_type="Similarity"):
    """Display a product card with information."""
    with st.container():
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.markdown("🧴")
        
        with col2:
            st.markdown(f"**{product['product_name']}**")
            st.markdown(f"*{product['brand_name']}*")
            
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                if product['rating'] > 0:
                    st.metric("Rating", f"⭐ {product['rating']:.1f}")
                else:
                    st.metric("Rating", "N/A")
            
            with col_b:
                if product['reviews'] > 0:
                    st.metric("Reviews", f"{int(product['reviews'])}")
                else:
                    st.metric("Reviews", "N/A")
            
            with col_c:
                if pd.notna(product['price_usd']) and product['price_usd'] > 0:
                    st.metric("Price", f"${product['price_usd']:.2f}")
                else:
                    st.metric("Price", "N/A")
            
            if score is not None:
                st.progress(float(score), text=f"{score_type} Score: {score:.3f}")
            
            if 'ingredients_clean' in product and pd.notna(product['ingredients_clean']):
                with st.expander("View Ingredients"):
                    st.caption(product['ingredients_clean'])
        
        st.divider()


def main():
    """Main application."""
    
    st.title("✨ GetItBeauty")
    st.markdown("### Find your perfect skincare match")
    
    # Load models
    with st.spinner("Loading AI models..."):
        products_df, cb_recommender, cf_recommender = load_models()
    
    # --- Main Search Form ---
    with st.container():
        col1, col2, col3 = st.columns(3)
        
        with col1:
            skin_type = st.selectbox(
                "Skin Type",
                options=[""] + config.SKIN_TYPES,
                help="Select your skin type for personalized results"
            )
            
        with col2:
            n_recs = st.number_input(
                "Number of recommendations",
                min_value=1,
                max_value=20,
                value=5
            )
            
        with col3:
             # Category selection
            categories = sorted(products_df['tertiary_category'].dropna().unique().tolist())
            selected_category = st.selectbox(
                "Product Category",
                options=["All"] + categories,
                index=0
            )

        # Product selection row
        product_options = products_df.copy()
        if selected_category != "All":
            product_options = product_options[
                product_options['tertiary_category'] == selected_category
            ]
        product_options = product_options.sort_values('product_name')
        
        selected_product_idx = st.selectbox(
            "Select a Product to match (Optional)",
            options=[None] + product_options.index.tolist(),
            format_func=lambda x: f"{product_options.loc[x, 'product_name']} ({product_options.loc[x, 'brand_name']})" if x is not None else "Choose a product...",
            help="Choose a product you like to find similar ones"
        )
        
        # Centered Button
        col_btn_1, col_btn_2, col_btn_3 = st.columns([1, 2, 1])
        with col_btn_2:
            search_clicked = st.button("🔍 Get Recommendations", type="primary", use_container_width=True)

    st.markdown("---")

    # --- Results Logic ---
    if search_clicked:
        hybrid = HybridRecommender(
            cb_recommender=cb_recommender,
            cf_recommender=cf_recommender if cf_recommender else cb_recommender,
            alpha=0.5  # Default balanced alpha
        )
        
        product_id = None
        if selected_product_idx is not None:
            product_id = product_options.loc[selected_product_idx, 'product_id']
            st.subheader(f"Because you liked: *{product_options.loc[selected_product_idx, 'product_name']}*")
        elif skin_type:
            st.subheader(f"Top picks for *{skin_type}* skin")
        else:
            st.warning("Please select a product or a skin type to get recommendations.")
            st.stop()

        with st.spinner("Analyzing ingredients and reviews..."):
            recommendations = hybrid.get_hybrid_recommendations(
                product_id=product_id,
                user_id=None,
                skin_type=skin_type if skin_type else None,
                n_recommendations=n_recs,
                alpha=0.5
            )
        
        if len(recommendations) > 0:
            # Merge with product details
            recs_with_details = recommendations.merge(
                products_df, 
                on='product_id', 
                how='left',
                suffixes=('', '_y')
            )
            
            # Display results
            for idx, rec in recs_with_details.iterrows():
                # Prefer original columns, fallback to _y if merge created duplicates
                display_cols = rec.to_dict()
                if 'product_name_y' in display_cols: 
                     # Handle potential merge column duplication if needed, 
                     # but left join usually keeps left keys. 
                     # Actually, merging on product_id with products_df (which is the source of recs) 
                     # might cause x/y columns if keys overlap.
                     # Let's clean up:
                     pass

                score_val = rec.get('hybrid_score', rec.get('cb_score', 0))
                display_product_card(rec, score=score_val, score_type="Match")
        else:
            st.warning("No recommendations found. Try adjusting your inputs.")


if __name__ == "__main__":
    main()
