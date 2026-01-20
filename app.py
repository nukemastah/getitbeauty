"""
Skincare Hybrid Recommendation System - Streamlit App
"""
import streamlit as st
import pandas as pd
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
    page_title="Skincare Recommender",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)


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
        st.info("Please run `python train_models.py` first to train the models.")
        st.stop()


def display_product_card(product, score=None, score_type="Similarity"):
    """Display a product card with information."""
    with st.container():
        col1, col2 = st.columns([1, 3])
        
        with col1:
            # Placeholder for product image
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
            
            # Show top ingredients
            if 'ingredients_clean' in product and pd.notna(product['ingredients_clean']):
                ingredients_preview = ' '.join(product['ingredients_clean'].split()[:10])
                with st.expander("View Ingredients"):
                    st.caption(ingredients_preview + "...")
        
        st.divider()


def main():
    """Main application."""
    
    # Header
    st.title("✨ Skincare Hybrid Recommendation System")
    st.markdown("""
    Get personalized skincare product recommendations based on:
    - **Content-Based Filtering**: Ingredient similarity
    - **Collaborative Filtering**: User preferences and ratings
    - **Hybrid Approach**: Best of both worlds!
    """)
    
    # Load models
    with st.spinner("Loading recommendation models..."):
        products_df, cb_recommender, cf_recommender = load_models()
    
    st.success(f"✓ Loaded {len(products_df)} products")
    
    # Sidebar - User Input
    st.sidebar.header("🎯 Your Preferences")
    
    # Skin type selection
    skin_type = st.sidebar.selectbox(
        "Select Your Skin Type",
        options=[""] + config.SKIN_TYPES,
        help="Choose your skin type for better recommendations"
    )
    
    # Alpha parameter for hybrid weighting
    st.sidebar.subheader("⚙️ Advanced Settings")
    alpha = st.sidebar.slider(
        "Content-Based Weight (α)",
        min_value=0.0,
        max_value=1.0,
        value=config.ALPHA_DEFAULT,
        step=0.1,
        help="Higher values prioritize ingredient similarity. Lower values prioritize user ratings."
    )
    
    st.sidebar.markdown(f"""
    **Current Mix:**
    - Content-Based: {alpha * 100:.0f}%
    - Collaborative: {(1 - alpha) * 100:.0f}%
    """)
    
    # Number of recommendations
    n_recs = st.sidebar.slider(
        "Number of Recommendations",
        min_value=5,
        max_value=20,
        value=10,
        step=1
    )
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["🔍 Product Search", "🌟 Popular Products", "📊 About"])
    
    with tab1:
        st.header("Search for Similar Products")
        
        # Product search
        search_query = st.text_input(
            "Search for a product",
            placeholder="e.g., moisturizer, hyaluronic acid serum...",
            help="Use fuzzy search to find products even with typos"
        )
        
        if search_query:
            # Create hybrid recommender
            hybrid = HybridRecommender(
                cb_recommender=cb_recommender,
                cf_recommender=cf_recommender if cf_recommender else cb_recommender,
                alpha=alpha
            )
            
            # Search products
            search_results = hybrid.search_product(
                search_query, 
                products_df, 
                threshold=config.FUZZY_SEARCH_THRESHOLD
            )
            
            if len(search_results) > 0:
                st.success(f"Found {len(search_results)} matching products")
                
                # Show search results
                selected_product = st.selectbox(
                    "Select a product to get recommendations",
                    options=search_results.index,
                    format_func=lambda x: f"{search_results.loc[x, 'product_name']} - {search_results.loc[x, 'brand_name']}"
                )
                
                product_id = search_results.loc[selected_product, 'product_id']
                product_name = search_results.loc[selected_product, 'product_name']
                
                # Show selected product
                st.subheader("Selected Product")
                display_product_card(search_results.loc[selected_product])
                
                # Get recommendations
                with st.spinner("Finding similar products..."):
                    recommendations = hybrid.get_hybrid_recommendations(
                        product_id=product_id,
                        user_id=None,
                        skin_type=skin_type if skin_type else None,
                        n_recommendations=n_recs,
                        alpha=alpha
                    )
                
                if len(recommendations) > 0:
                    st.subheader(f"✨ Recommended Products Similar to '{product_name}'")
                    
                    # Show recommendation type
                    rec_type = recommendations.iloc[0]['recommendation_type']
                    if rec_type == 'hybrid':
                        st.info("🎯 Using Hybrid Recommendations (Content + Collaborative)")
                    elif rec_type == 'content_based':
                        st.info("🧪 Using Content-Based Recommendations (Ingredient Similarity)")
                    else:
                        st.info("👥 Using Collaborative Recommendations (User Preferences)")
                    
                    # Merge with product details
                    recs_with_details = recommendations.merge(
                        products_df, 
                        on='product_id', 
                        how='left',
                        suffixes=('', '_y')
                    )
                    
                    # Display recommendations
                    for idx, rec in recs_with_details.iterrows():
                        display_product_card(rec, score=rec['hybrid_score'], score_type="Match")
                else:
                    st.warning("No recommendations found. Try a different product.")
            else:
                st.warning(f"No products found matching '{search_query}'. Try a different search term.")
    
    with tab2:
        st.header("Popular Products by Category")
        
        # Category filter
        categories = products_df['tertiary_category'].dropna().unique()
        selected_category = st.selectbox(
            "Filter by Category",
            options=["All"] + sorted(categories.tolist()),
            index=0
        )
        
        # Filter products
        if selected_category == "All":
            filtered_products = products_df.copy()
        else:
            filtered_products = products_df[
                products_df['tertiary_category'] == selected_category
            ].copy()
        
        # Sort by popularity
        filtered_products['popularity'] = (
            filtered_products['rating'] * pd.np.log1p(filtered_products['reviews'])
        )
        
        top_products = filtered_products.nlargest(n_recs, 'popularity')
        
        st.subheader(f"Top {len(top_products)} Products" + (f" in {selected_category}" if selected_category != "All" else ""))
        
        for idx, product in top_products.iterrows():
            display_product_card(product)
    
    with tab3:
        st.header("About This System")
        
        st.markdown("""
        ### 🎯 How It Works
        
        This hybrid recommendation system combines two powerful approaches:
        
        #### 1. Content-Based Filtering (CB)
        - Analyzes product **ingredients** using TF-IDF vectorization
        - Computes **cosine similarity** between products
        - Recommends products with similar ingredient profiles
        
        #### 2. Collaborative Filtering (CF)
        - Uses **TruncatedSVD** for matrix factorization
        - Learns from user ratings and preferences
        - Finds patterns in how users rate products
        
        #### 3. Hybrid Scoring
        The final recommendation score combines both approaches:
        
        $$Final\\_Score = (\\alpha \\times CB\\_Score) + ((1 - \\alpha) \\times CF\\_Score)$$
        
        Where $\\alpha$ is the weight you control in the sidebar.
        
        ### 🚀 Features
        - **Fuzzy Search**: Find products even with typos (powered by thefuzz)
        - **Cold Start Handling**: Get recommendations even without user history
        - **Skin Type Filtering**: Personalized recommendations based on your skin type
        - **Adjustable Hybrid Weight**: Control the balance between content and collaborative filtering
        
        ### 📊 Dataset
        - **Sephora Products**: ~8,000 skincare products
        - **User Reviews**: Hundreds of thousands of ratings
        - **Additional Skincare Data**: Extended ingredient information
        
        ### 🛠️ Technology Stack
        - **Framework**: Streamlit
        - **ML Libraries**: scikit-learn, pandas, numpy
        - **Recommendation Models**: TF-IDF, TruncatedSVD
        - **Search**: thefuzz (fuzzy string matching)
        
        ### 👨‍💻 Model Details
        - TF-IDF max features: {config.TFIDF_MAX_FEATURES}
        - SVD components: {config.SVD_N_COMPONENTS}
        - Default α: {config.ALPHA_DEFAULT}
        """)
        
        # System statistics
        st.subheader("📈 System Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Products", len(products_df))
        
        with col2:
            avg_rating = products_df[products_df['rating'] > 0]['rating'].mean()
            st.metric("Average Rating", f"{avg_rating:.2f}⭐")
        
        with col3:
            total_reviews = products_df['reviews'].sum()
            st.metric("Total Reviews", f"{int(total_reviews):,}")
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style='text-align: center'>
        <p style='font-size: 0.8em; color: #666;'>
            Made with ❤️ for skincare enthusiasts<br>
            Powered by Streamlit & scikit-learn
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
