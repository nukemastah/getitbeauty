"""
Data Preprocessing Module
Handles loading, cleaning, and merging of product and review datasets
"""
import pandas as pd
import numpy as np
import re
from typing import List, Tuple
import ast


def clean_ingredients(ingredients_str: str) -> str:
    """
    Clean and preprocess ingredient strings.
    
    Args:
        ingredients_str: Raw ingredient string from dataset
        
    Returns:
        Cleaned ingredient string (lowercase, no punctuation, tokenized)
    """
    if pd.isna(ingredients_str) or ingredients_str == '':
        return ''
    
    try:
        # Try to parse if it's a list string representation
        if ingredients_str.startswith('['):
            ingredient_list = ast.literal_eval(ingredients_str)
            if isinstance(ingredient_list, list):
                ingredients_str = ' '.join(str(item) for item in ingredient_list)
    except:
        pass
    
    # Convert to lowercase
    ingredients_str = str(ingredients_str).lower()
    
    # Remove special characters and numbers, keep only letters and spaces
    ingredients_str = re.sub(r'[^a-z\s]', ' ', ingredients_str)
    
    # Remove extra whitespace
    ingredients_str = re.sub(r'\s+', ' ', ingredients_str).strip()
    
    return ingredients_str


def load_sephora_products(products_path: str) -> pd.DataFrame:
    """
    Load and preprocess Sephora products dataset.
    
    Args:
        products_path: Path to Sephora products CSV file
        
    Returns:
        Cleaned products DataFrame
    """
    print("Loading Sephora products...")
    df = pd.read_csv(products_path)
    
    # Keep relevant columns
    columns_to_keep = [
        'product_id', 'product_name', 'brand_name', 'rating', 
        'reviews', 'ingredients', 'price_usd', 'primary_category',
        'secondary_category', 'tertiary_category'
    ]
    
    df = df[[col for col in columns_to_keep if col in df.columns]]
    
    # Clean ingredients
    if 'ingredients' in df.columns:
        df['ingredients_clean'] = df['ingredients'].apply(clean_ingredients)
    else:
        df['ingredients_clean'] = ''
    
    # Remove products without ingredients
    df = df[df['ingredients_clean'] != ''].copy()
    
    # Fill missing values
    df['rating'] = df['rating'].fillna(0)
    df['reviews'] = df['reviews'].fillna(0)
    df['price_usd'] = df['price_usd'].fillna(df['price_usd'].median())
    
    # Add source column
    df['source'] = 'sephora'
    
    print(f"Loaded {len(df)} Sephora products")
    return df


def load_skincare_products(skincare_path: str) -> pd.DataFrame:
    """
    Load and preprocess skincare products dataset.
    
    Args:
        skincare_path: Path to skincare products CSV file
        
    Returns:
        Cleaned products DataFrame
    """
    print("Loading additional skincare products...")
    df = pd.read_csv(skincare_path)
    
    # Rename columns to match Sephora format
    column_mapping = {
        'product_name': 'product_name',
        'clean_ingreds': 'ingredients',
        'product_type': 'tertiary_category',
        'price': 'price_usd'
    }
    
    df = df.rename(columns=column_mapping)
    
    # Create product_id if not exists
    if 'product_id' not in df.columns:
        df['product_id'] = 'SK' + df.index.astype(str)
    
    # Clean ingredients
    if 'ingredients' in df.columns:
        df['ingredients_clean'] = df['ingredients'].apply(clean_ingredients)
    else:
        df['ingredients_clean'] = ''
    
    # Remove products without ingredients
    df = df[df['ingredients_clean'] != ''].copy()
    
    # Extract price if it contains currency symbols
    if 'price_usd' in df.columns:
        df['price_usd'] = df['price_usd'].astype(str).str.extract(r'([\d.]+)')[0].astype(float)
    
    # Fill missing columns
    df['brand_name'] = df.get('brand_name', 'Unknown')
    df['rating'] = 0
    df['reviews'] = 0
    df['source'] = 'skincare'
    
    print(f"Loaded {len(df)} additional skincare products")
    return df


def load_reviews(reviews_paths: List[str], sample_size: int = None) -> pd.DataFrame:
    """
    Load and concatenate review files.
    
    Args:
        reviews_paths: List of paths to review CSV files
        sample_size: Optional sample size per file (for faster processing)
        
    Returns:
        Combined reviews DataFrame
    """
    print("Loading reviews...")
    dfs = []
    
    for path in reviews_paths:
        try:
            if sample_size:
                df = pd.read_csv(path, nrows=sample_size)
            else:
                # For large files, read in chunks
                df = pd.read_csv(path)
            dfs.append(df)
            print(f"  Loaded {len(df)} reviews from {path.split('/')[-1]}")
        except Exception as e:
            print(f"  Warning: Could not load {path}: {e}")
            continue
    
    if not dfs:
        print("No reviews loaded!")
        return pd.DataFrame()
    
    reviews_df = pd.concat(dfs, ignore_index=True)
    
    # Keep relevant columns
    columns_to_keep = ['product_id', 'rating', 'is_recommended', 'skin_type']
    reviews_df = reviews_df[[col for col in columns_to_keep if col in reviews_df.columns]]
    
    # Clean rating column
    reviews_df['rating'] = pd.to_numeric(reviews_df['rating'], errors='coerce')
    reviews_df = reviews_df.dropna(subset=['rating'])
    reviews_df = reviews_df[reviews_df['rating'] > 0]
    
    # Create synthetic user IDs (since original might be masked)
    reviews_df['user_id'] = reviews_df.index
    
    print(f"Loaded total {len(reviews_df)} reviews")
    return reviews_df


def merge_products_and_reviews(products_df: pd.DataFrame, 
                                reviews_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Merge products with reviews to ensure only reviewed products are included.
    
    Args:
        products_df: Products DataFrame
        reviews_df: Reviews DataFrame
        
    Returns:
        Tuple of (merged_products, filtered_reviews)
    """
    print("Merging products and reviews...")
    
    # Get products that have reviews
    products_with_reviews = products_df[
        products_df['product_id'].isin(reviews_df['product_id'])
    ].copy()
    
    # Filter reviews to only include products we have
    filtered_reviews = reviews_df[
        reviews_df['product_id'].isin(products_with_reviews['product_id'])
    ].copy()
    
    print(f"Products with reviews: {len(products_with_reviews)}")
    print(f"Filtered reviews: {len(filtered_reviews)}")
    
    return products_with_reviews, filtered_reviews


def create_user_item_matrix(reviews_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create user-item rating matrix for collaborative filtering.
    
    Args:
        reviews_df: Reviews DataFrame with user_id, product_id, rating
        
    Returns:
        User-item matrix (users as rows, products as columns)
    """
    print("Creating user-item matrix...")
    
    # Pivot to create matrix
    user_item_matrix = reviews_df.pivot_table(
        index='user_id',
        columns='product_id',
        values='rating',
        aggfunc='mean'
    )
    
    print(f"User-item matrix shape: {user_item_matrix.shape}")
    print(f"Sparsity: {(user_item_matrix.isna().sum().sum() / (user_item_matrix.shape[0] * user_item_matrix.shape[1]) * 100):.2f}%")
    
    return user_item_matrix


def preprocess_all_data(sephora_path: str, 
                        skincare_path: str, 
                        reviews_paths: List[str],
                        sample_reviews: int = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Main preprocessing pipeline.
    
    Args:
        sephora_path: Path to Sephora products
        skincare_path: Path to skincare products
        reviews_paths: List of review file paths
        sample_reviews: Optional number of reviews to sample per file
        
    Returns:
        Tuple of (products_df, reviews_df, user_item_matrix)
    """
    # Load products
    sephora_df = load_sephora_products(sephora_path)
    skincare_df = load_skincare_products(skincare_path)
    
    # Combine products
    products_df = pd.concat([sephora_df, skincare_df], ignore_index=True)
    products_df = products_df.drop_duplicates(subset=['product_id'])
    
    print(f"\nTotal products: {len(products_df)}")
    
    # Load reviews
    reviews_df = load_reviews(reviews_paths, sample_size=sample_reviews)
    
    if len(reviews_df) == 0:
        print("Warning: No reviews loaded. Using product data only.")
        return products_df, pd.DataFrame(), pd.DataFrame()
    
    # Merge and filter
    products_df, reviews_df = merge_products_and_reviews(products_df, reviews_df)
    
    # Create user-item matrix
    user_item_matrix = create_user_item_matrix(reviews_df)
    
    return products_df, reviews_df, user_item_matrix
