# ✨ Skincare Hybrid Recommendation System

A sophisticated recommendation system for skincare products that combines **Content-Based Filtering** (ingredient similarity) and **Collaborative Filtering** (user preferences) to provide personalized product recommendations.

![Python](https://img.shields.io/badge/python-3.14-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)

## 🎯 Features

- **Hybrid Recommendation**: Combines Content-Based and Collaborative Filtering with adjustable weighting
- **Fuzzy Search**: Find products even with typos using intelligent string matching
- **Cold Start Handling**: Get recommendations even without user history
- **Skin Type Personalization**: Filter recommendations based on skin type
- **Interactive UI**: Beautiful Streamlit interface with product cards and visualizations
- **Model Caching**: Pre-trained models are saved and loaded for instant recommendations

## 📊 How It Works

### Content-Based Filtering (CB)
- Uses **TF-IDF vectorization** on product ingredients
- Computes **cosine similarity** between product ingredient profiles
- Recommends products with similar chemical compositions

### Collaborative Filtering (CF)
- Implements **TruncatedSVD** (Singular Value Decomposition) for matrix factorization
- Learns latent factors from user-product rating matrix
- Finds patterns in user preferences and product popularity

### Hybrid Scoring Formula

$$Final\\_Score = (\\alpha \\times CB\\_Score) + ((1 - \\alpha) \\times CF\\_Score)$$

Where:
- $\\alpha$ = Content-Based weight (default: 0.5)
- $CB\\_Score$ = Content similarity score (0-1)
- $CF\\_Score$ = Collaborative filtering score (0-1)

## 🚀 Installation & Setup

### Prerequisites
- Python 3.10 or higher (tested on Python 3.14)
- Linux OS (tested on Arch Linux)

### Step 1: Clone or Navigate to Project Directory

```bash
cd /home/bayu/Documents/Archive/bayu/ML/getitbeuty
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Verify Dataset Structure

Ensure your dataset folder has this structure:

```
dataset/
├── sephora/
│   ├── product_info.csv
│   ├── reviews_0-250_masked.csv
│   ├── reviews_250-500_masked.csv
│   ├── reviews_500-750_masked.csv
│   ├── reviews_750-1250_masked.csv
│   └── reviews_1250-end_masked.csv
└── skincare/
    └── skincare_products_clean.csv
```

### Step 5: Train Models

```bash
python train_models.py
```

This will:
- Load and preprocess all datasets
- Train Content-Based model (TF-IDF + cosine similarity)
- Train Collaborative Filtering model (TruncatedSVD)
- Save all models to the `models/` directory
- Display sample recommendations

**Expected output:**
```
============================================================
Skincare Hybrid Recommender - Model Training
============================================================

[Step 1/4] Loading and preprocessing data...
------------------------------------------------------------
Loading Sephora products...
Loaded 8496 Sephora products
Loading additional skincare products...
Loaded 1141 additional skincare products

Total products: 9637
Loading reviews...
...

Training Complete!
============================================================
```

**Note:** Training may take 5-15 minutes depending on your system and data size.

### Step 6: Launch Streamlit App

```bash
streamlit run app.py
```

The app will automatically open in your browser at `http://localhost:8501`

## 📖 Usage Guide

### 1. Product Search Tab

1. Enter a product name or ingredient in the search box
2. Select a product from the fuzzy search results
3. Adjust the Content-Based weight (α) in the sidebar
4. View personalized recommendations based on ingredient similarity and user ratings

### 2. Popular Products Tab

- Browse top-rated products by category
- Filter by product type (Moisturizer, Serum, Cleanser, etc.)
- View product details, ratings, and ingredients

### 3. Advanced Settings (Sidebar)

- **Skin Type**: Select your skin type for better recommendations
- **Content-Based Weight (α)**: 
  - `1.0` = Pure ingredient-based recommendations
  - `0.5` = Balanced hybrid (default)
  - `0.0` = Pure user preference-based recommendations
- **Number of Recommendations**: 5-20 products

## 🏗️ Project Structure

```
getitbeuty/
├── app.py                      # Streamlit application
├── train_models.py             # Model training script
├── config.py                   # Configuration and paths
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── utils/                      # Utility modules
│   ├── __init__.py
│   ├── data_preprocessing.py   # Data loading and cleaning
│   ├── content_based.py        # Content-Based Filtering
│   ├── collaborative_filtering.py  # Collaborative Filtering
│   └── hybrid_recommender.py   # Hybrid recommendation engine
│
├── models/                     # Saved models (generated after training)
│   ├── tfidf_vectorizer.pkl
│   ├── tfidf_matrix.pkl
│   ├── similarity_matrix.pkl
│   ├── svd_model.pkl
│   ├── user_item_matrix.pkl
│   └── products_data.pkl
│
└── dataset/                    # Your datasets
    ├── sephora/
    └── skincare/
```

## 🔧 Configuration

Edit `config.py` to customize:

```python
# Model parameters
TFIDF_MAX_FEATURES = 500        # TF-IDF vocabulary size
TFIDF_NGRAM_RANGE = (1, 2)      # Unigrams and bigrams
SVD_N_COMPONENTS = 50            # Latent factors for CF
ALPHA_DEFAULT = 0.5              # Default hybrid weight

# Recommendation settings
DEFAULT_N_RECOMMENDATIONS = 10
FUZZY_SEARCH_THRESHOLD = 70      # Minimum fuzzy match score
```

## 🧪 Model Details

### Content-Based Model
- **Algorithm**: TF-IDF + Cosine Similarity
- **Features**: Product ingredients (preprocessed and tokenized)
- **Max Features**: 500 (configurable)
- **N-gram Range**: (1, 2) - unigrams and bigrams

### Collaborative Filtering Model
- **Algorithm**: TruncatedSVD (Matrix Factorization)
- **Components**: 50 latent factors (configurable)
- **Input**: User-item rating matrix (sparse)
- **Normalization**: Mean-centered ratings

### Performance Optimization
- **Model Caching**: Uses Streamlit's `@st.cache_resource`
- **Pre-computation**: Similarity matrices computed once during training
- **Sparse Matrices**: Efficient storage for large user-item matrices

## 🐛 Troubleshooting

### Issue: "No module named 'utils'"
**Solution**: Make sure you're running scripts from the project root directory.

### Issue: "Models not found" error in Streamlit
**Solution**: Run `python train_models.py` first to generate model files.

### Issue: Training takes too long
**Solution**: Reduce sample size in `train_models.py`:
```python
preprocess_all_data(..., sample_reviews=10000)  # Use fewer reviews
```

### Issue: Python 3.14 compatibility errors
**Solution**: 
- Most libraries should work with Python 3.14
- If issues persist, use Python 3.11 or 3.12
- Check for updated package versions

### Issue: Memory errors during training
**Solution**:
- Reduce `TFIDF_MAX_FEATURES` in config.py
- Reduce `SVD_N_COMPONENTS` in config.py
- Use sample_reviews parameter to limit data size

## 📈 Future Enhancements

- [ ] Add image-based product search
- [ ] Implement user authentication and history tracking
- [ ] Add product review sentiment analysis
- [ ] Include price-based filtering
- [ ] Deploy to cloud (Streamlit Cloud, Heroku, etc.)
- [ ] A/B testing for different alpha values
- [ ] Add ingredient allergen warnings

## 📚 Technical References

### Algorithms Used
- **TF-IDF**: Term Frequency-Inverse Document Frequency for text vectorization
- **Cosine Similarity**: Measure of similarity between ingredient vectors
- **TruncatedSVD**: Dimensionality reduction for collaborative filtering
- **Fuzzy String Matching**: Levenshtein distance for search

### Libraries
- `scikit-learn`: Machine learning algorithms
- `pandas`: Data manipulation
- `numpy`: Numerical computations
- `streamlit`: Web application framework
- `thefuzz`: Fuzzy string matching

## 🤝 Contributing

Feel free to enhance the system by:
1. Adding more data sources
2. Implementing additional recommendation algorithms
3. Improving the UI/UX
4. Optimizing model performance

## 📄 License

This project is for educational and personal use.

## 👨‍💻 Author

Built with ❤️ for skincare enthusiasts using Python, Streamlit, and scikit-learn.

---

**Happy Skincare Shopping! ✨**
