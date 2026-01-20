# 🚀 Quick Start Guide

## Setup & Run (3 Steps)

### 1️⃣ Setup Environment
```bash
# Run the setup script
./setup.sh

# Or manually:
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2️⃣ Train Models
```bash
python train_models.py
```
⏱️ Takes ~5-15 minutes depending on your system

### 3️⃣ Launch App
```bash
streamlit run app.py
```
🌐 Opens automatically at http://localhost:8501

---

## 📁 What You Get

```
✨ Hybrid Recommendation System with:
├── 🧪 Content-Based Filtering (Ingredients)
├── 👥 Collaborative Filtering (User Ratings)
├── 🔍 Fuzzy Product Search
├── 🎯 Skin Type Personalization
└── ⚙️ Adjustable Hybrid Weighting
```

---

## 🎮 How to Use

1. **Search Tab**: 
   - Search for a product (e.g., "moisturizer")
   - Select from fuzzy-matched results
   - Get personalized recommendations

2. **Popular Tab**:
   - Browse top-rated products
   - Filter by category

3. **Sidebar Settings**:
   - Select skin type
   - Adjust α (Content vs Collaborative weight)
   - Change number of recommendations

---

## 💡 Tips

- **α = 1.0**: Pure ingredient-based (best for specific formulations)
- **α = 0.5**: Balanced hybrid (recommended)
- **α = 0.0**: Pure popularity-based (best for trendy products)

- Use fuzzy search - typos are OK! ("hylaronic" finds "hyaluronic")
- Cold start works - get recs even without search history
- Models are cached - instant recommendations after first load

---

## ⚠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| "Models not found" | Run `python train_models.py` first |
| Import errors | Activate venv: `source venv/bin/activate` |
| Memory errors | Reduce sample size in train_models.py |
| Slow training | Use fewer reviews (edit sample_reviews param) |

---

## 📊 Model Info

- **Content-Based**: TF-IDF on 500 ingredient features
- **Collaborative**: SVD with 50 latent factors  
- **Dataset**: ~9,600 products + 100K+ reviews
- **Search**: Levenshtein distance fuzzy matching

---

**For full documentation, see [README.md](README.md)**
