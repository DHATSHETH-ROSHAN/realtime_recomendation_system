# Step 1: Load the data
# Step 2: Collaborative Filtering
# Step 3: Content-Based Filtering
# Step 4: Hybrid Recommendations
# Step 5: Test the system

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
# Step 1: Load the data
# Sample user-product interaction data
interactions = pd.DataFrame({
    'user_id': [1, 1, 2, 2, 3, 3],
    'product_id': [101, 102, 101, 103, 102, 104],
    'rating': [5, 4, 4, 5, 4, 5]  # Implicit or explicit feedback
})

# Sample product metadata
products = pd.DataFrame({
    'product_id': [101, 102, 103, 104],
    'title': ['Laptop', 'Smartphone', 'Tablet', 'Smartwatch'],
    'category': ['Electronics', 'Electronics', 'Electronics', 'Wearable'],
    'description': [
        'Powerful gaming laptop with high performance',
        'Latest smartphone with advanced features',
        'Portable tablet for everyday use',
        'Stylish smartwatch with fitness tracking'
    ]
})

# Step 2: Collaborative Filtering
# Create a user-item interaction matrix
interaction_matrix = interactions.pivot(index='user_id', columns='product_id', values='rating').fillna(0)

# Perform SVD
U, sigma, Vt = svds(interaction_matrix, k=2)  # k = Latent factors
sigma = np.diag(sigma)

# Predict ratings
predicted_ratings = np.dot(np.dot(U, sigma), Vt)
predicted_ratings_df = pd.DataFrame(predicted_ratings, columns=interaction_matrix.columns, index=interaction_matrix.index)

# Function to recommend products based on collaborative filtering
def recommend_products_cf(user_id, n_recommendations=2):
    user_row = predicted_ratings_df.loc[user_id]
    recommended_products = user_row.sort_values(ascending=False).head(n_recommendations).index.tolist()
    return recommended_products

# Step 3: Content-Based Filtering
# Create TF-IDF vectors for product descriptions
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(products['description'])

# Compute cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Map product IDs to indices
product_indices = pd.Series(products.index, index=products['product_id'])

# Function to recommend products based on content-based filtering
def recommend_products_cb(product_id, n_recommendations=2):
    idx = product_indices[product_id]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n_recommendations+1]
    recommended_indices = [i[0] for i in sim_scores]
    return products.iloc[recommended_indices]['product_id'].tolist()

# Step 4: Hybrid Recommendations
# Combine collaborative and content-based filtering
def hybrid_recommendations(user_id, product_id, n_recommendations=3):
    # Collaborative recommendations
    cf_recs = recommend_products_cf(user_id, n_recommendations)
    
    # Content-based recommendations
    cb_recs = recommend_products_cb(product_id, n_recommendations)
    
    # Combine and rank unique recommendations
    combined_recs = list(set(cf_recs + cb_recs))
    return combined_recs[:n_recommendations]

# Step 5: Test the system
# Collaborative Filtering Example
print("Collaborative Filtering Recommendations for User 1:", recommend_products_cf(user_id=1))

# Content-Based Filtering Example
print("Content-Based Recommendations for Product 101:", recommend_products_cb(product_id=101))

# Hybrid Recommendations Example
print("Hybrid Recommendations for User 1 and Product 101:", hybrid_recommendations(user_id=1, product_id=101))
