import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')

def train_and_classify():
    # 1. Load the data
    filename = 'Influencer_metrics_with_sales_percentage.csv'
    print(f"Loading data from {filename}...")
    df = pd.read_csv(filename)
    
    # 2. Define the features requested by the user
    features = [
        'INSTAGRAM_TOTAL_LIKES',
        'INSTAGRAM_TOTAL_COMMENTS',
        'INSTAGRAM_TOTAL_POSTS',
        'INSTAGRAM_TOTAL_ENGAGEMENT',
        'INSTAGRAM_FOLLOWER_COUNT',
        'INSTAGRAM_ENGAGEMENT_RATE'
    ]
    
    # Drop rows with missing values in the feature columns
    # Alternatively, fillna(0) could work, but let's drop them for clean training
    analysis_df = df.dropna(subset=features).copy()
    
    print(f"Number of records after dropping missing values: {len(analysis_df)}")
    
    if len(analysis_df) < 3:
        print("Not enough data to form 3 clusters.")
        return
        
    # 3. Scale the features
    # Standardizing the features so that distance metrics work correctly
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(analysis_df[features])
    
    # 4. Apply K-Means clustering (k=3 for high, moderate, low fit)
    print("Training KMeans model with 3 clusters...")
    kmeans = KMeans(n_clusters=3, random_state=42)
    analysis_df['Cluster_Label'] = kmeans.fit_predict(scaled_features)
    
    # 5. Analyze the clusters to assign 'High Fit', 'Moderate Fit', 'Low Fit'
    # We will rank them based on engagement rate and follower count.
    # We rank clusters by a score = avg normalized follower count + avg normalized engagement rate
    # Or just simply sort by mean values of engagement rate + normalized followers
    cluster_centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=features)
    cluster_centers['Cluster'] = range(3)
    
    # Let's normalize just the centers for ranking purposes (to combine followers and engagement)
    # Min-max scale the centers so they are on a 0-1 scale, then add them
    follower_scaled = (cluster_centers['INSTAGRAM_FOLLOWER_COUNT'] - cluster_centers['INSTAGRAM_FOLLOWER_COUNT'].min()) / (cluster_centers['INSTAGRAM_FOLLOWER_COUNT'].max() - cluster_centers['INSTAGRAM_FOLLOWER_COUNT'].min() + 1e-9)
    engagement_scaled = (cluster_centers['INSTAGRAM_ENGAGEMENT_RATE'] - cluster_centers['INSTAGRAM_ENGAGEMENT_RATE'].min()) / (cluster_centers['INSTAGRAM_ENGAGEMENT_RATE'].max() - cluster_centers['INSTAGRAM_ENGAGEMENT_RATE'].min() + 1e-9)
    
    cluster_score = follower_scaled + engagement_scaled
    
    # Rank clusters based on the score
    # Highest score = High Fit, Middle = Moderate Fit, Lowest = Low Fit
    sorted_cluster_indices = cluster_score.sort_values(ascending=True).index.tolist()
    
    cluster_mapping = {
        sorted_cluster_indices[0]: 'Low Fit',
        sorted_cluster_indices[1]: 'Moderate Fit',
        sorted_cluster_indices[2]: 'High Fit'
    }
    
    analysis_df['Fit_Classification'] = analysis_df['Cluster_Label'].map(cluster_mapping)
    
    print("\nCluster Characteristics:")
    for cluster_idx in cluster_mapping:
        fit = cluster_mapping[cluster_idx]
        subset = analysis_df[analysis_df['Cluster_Label'] == cluster_idx]
        avg_followers = subset['INSTAGRAM_FOLLOWER_COUNT'].mean()
        avg_engagement = subset['INSTAGRAM_ENGAGEMENT_RATE'].mean()
        count = len(subset)
        print(f"{fit} (Cluster {cluster_idx}): {count} influencers, Avg Followers: {avg_followers:.0f}, Avg Engagement Rate: {avg_engagement:.2f}%")
        
    # Join back to original dataset or save new
    # Merge using index or just save analysis_df
    output_filename = 'Influencer_Classified.csv'
    analysis_df.to_csv(output_filename, index=False)
    print(f"\nClassification complete. Output saved to {output_filename}")
    
    # Let's show a few examples
    print("\nSample Output:")
    cols_to_display = ['INFLUENCER_ID', 'NAME', 'INSTAGRAM_FOLLOWER_COUNT', 'INSTAGRAM_ENGAGEMENT_RATE', 'Fit_Classification']
    existing_cols = [c for c in cols_to_display if c in analysis_df.columns]
    print(analysis_df[existing_cols].sample(min(10, len(analysis_df)), random_state=42))

if __name__ == "__main__":
    train_and_classify()
