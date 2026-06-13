"""
SmartInfluence Benchmarking — Data Loader & Feature Engineering

Shared module providing data loading, feature engineering, target variable
creation, and data preparation for all 6 benchmark trials.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# DATA PATHS
# ─────────────────────────────────────────────────────────────────────────────
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CSV_PATH = os.path.join(_BASE_DIR, 'data', 'processed', 'Influencer_Classified_with_growth.csv')


def load_data():
    """Load the processed influencer dataset and clean text columns."""
    df = pd.read_csv(_CSV_PATH)
    df['niche']       = df['niche'].fillna('').str.strip().str.lower()
    df['description'] = df['description'].fillna('').str.strip().str.lower()
    df['ACCOUNTNAME'] = df['ACCOUNTNAME'].fillna('').str.strip()
    return df


def engineer_features(df):
    """
    Apply feature engineering matching the existing xgboost_model.py.
    Creates ratio features, quality signals, interaction features,
    and log transforms.
    """
    d = df.copy()

    # ── Activity ratios ──
    d['engagement_per_post'] = d['INSTAGRAM_TOTAL_ENGAGEMENT'] / (d['INSTAGRAM_TOTAL_POSTS'] + 1)
    d['posting_consistency'] = d['INSTAGRAM_POSTS_LAST_30_DAYS'] / (d['INSTAGRAM_POSTS_LAST_90_DAYS'] + 1)
    d['posting_momentum']    = d['INSTAGRAM_POSTS_LAST_90_DAYS'] / (d['INSTAGRAM_POSTS_LAST_180_DAYS'] + 1)
    d['likes_per_post']      = d['INSTAGRAM_TOTAL_LIKES'] / (d['INSTAGRAM_TOTAL_POSTS'] + 1)
    d['comments_per_post']   = d['INSTAGRAM_TOTAL_COMMENTS'] / (d['INSTAGRAM_TOTAL_POSTS'] + 1)
    d['clicks_per_post']     = d['REFERRAL_LINK_TOTAL_CLICKS'] / (d['INSTAGRAM_TOTAL_POSTS'] + 1)
    d['campaign_commitment'] = d['CAMPAIGN_OPT_INS'] / (d['INSTAGRAM_TOTAL_POSTS'] + 1)

    # ── Quality signals ──
    d['virality_score']      = d['INSTAGRAM_TOTAL_COMMENTS'] / (d['INSTAGRAM_TOTAL_LIKES'] + 1)
    d['reach_efficiency']    = d['REFERRAL_LINK_TOTAL_CLICKS'] / (d['INSTAGRAM_FOLLOWER_COUNT'] + 1)
    d['engagement_density']  = d['INSTAGRAM_TOTAL_ENGAGEMENT'] / (d['INSTAGRAM_FOLLOWER_COUNT'] + 1)

    # ── Interaction features ──
    d['engagement_x_growth'] = d['INSTAGRAM_ENGAGEMENT_RATE'] * d['Growth Rate']
    d['absolute_engagement'] = d['INSTAGRAM_FOLLOWER_COUNT'] * d['INSTAGRAM_ENGAGEMENT_RATE'] / 100
    d['follower_x_growth']   = d['INSTAGRAM_FOLLOWER_COUNT'] * d['Growth Rate']
    d['eng_x_clicks']        = d['INSTAGRAM_ENGAGEMENT_RATE'] * d['REFERRAL_LINK_TOTAL_CLICKS']
    d['eng_rate_sq']         = d['INSTAGRAM_ENGAGEMENT_RATE'] ** 2

    # ── Log transforms ──
    d['log_followers']       = np.log1p(d['INSTAGRAM_FOLLOWER_COUNT'])
    d['log_engagement']      = np.log1p(d['INSTAGRAM_TOTAL_ENGAGEMENT'])
    d['log_likes']           = np.log1p(d['INSTAGRAM_TOTAL_LIKES'])
    d['log_clicks']          = np.log1p(d['REFERRAL_LINK_TOTAL_CLICKS'])

    return d


def get_feature_columns(exclude_cols=None):
    """
    Return the list of feature columns for modeling.
    Excludes leakage columns by default:
    TOTAL_ORDERS, NEW_CUSTOMERS, AVG_ORDER_SIZE, TOTAL_COMMISSION, PERCENTAGE_OF_BRAND_SALES
    
    Parameters
    ----------
    exclude_cols : list, optional
        Additional columns to exclude (e.g., when target is a feature).
    """
    features = [
        # Raw Instagram metrics
        'INSTAGRAM_ENGAGEMENT_RATE', 'Growth Rate', 'INSTAGRAM_FOLLOWER_COUNT',
        'INSTAGRAM_TOTAL_POSTS', 'INSTAGRAM_POSTS_LAST_30_DAYS',
        'INSTAGRAM_POSTS_LAST_90_DAYS', 'INSTAGRAM_POSTS_LAST_180_DAYS',
        'INSTAGRAM_POSTS_IN_LAST_1_YEAR', 'INSTAGRAM_TOTAL_LIKES',
        'INSTAGRAM_TOTAL_COMMENTS', 'INSTAGRAM_TOTAL_ENGAGEMENT',
        'COMMISSION_PERCENTAGE', 'REFERRAL_LINK_TOTAL_CLICKS',
        'CAMPAIGN_OPT_INS', 'FLAT_FEE_COMMISSION',
        # Engineered features
        'engagement_per_post', 'posting_consistency', 'posting_momentum',
        'likes_per_post', 'comments_per_post', 'clicks_per_post',
        'campaign_commitment', 'virality_score', 'reach_efficiency',
        'engagement_density', 'engagement_x_growth', 'absolute_engagement',
        'follower_x_growth', 'eng_x_clicks', 'eng_rate_sq',
        'log_followers', 'log_engagement', 'log_likes', 'log_clicks',
    ]
    if exclude_cols:
        features = [f for f in features if f not in exclude_cols]
    return features


# ─────────────────────────────────────────────────────────────────────────────
# TARGET VARIABLE CREATORS
# ─────────────────────────────────────────────────────────────────────────────

def create_target_sales_class(df):
    """Trial 1: TOTAL_SALES -> High (>$10K) / Mid ($1K-$10K) / Low (<$1K)."""
    d = df.copy()
    def classify(s):
        if s > 10000:   return 'High Performer'
        elif s >= 1000: return 'Mid Performer'
        else:           return 'Low Performer'
    d['target_class'] = d['TOTAL_SALES'].apply(classify)
    return d


def create_target_engagement_class(df):
    """Trial 2: INSTAGRAM_ENGAGEMENT_RATE -> High (>5%) / Medium (2-5%) / Low (<2%)."""
    d = df.copy()
    def classify(e):
        if e > 5:    return 'High Engagement'
        elif e >= 2: return 'Medium Engagement'
        else:        return 'Low Engagement'
    d['target_class'] = d['INSTAGRAM_ENGAGEMENT_RATE'].apply(classify)
    return d


def create_target_fit_classification(df):
    """Trial 3: Use existing Fit_Classification column."""
    d = df.copy()
    d['target_class'] = d['Fit_Classification']
    return d


def create_target_orders_class(df):
    """Trial 4: TOTAL_ORDERS -> High (>100) / Mid (10-100) / Low (<10)."""
    d = df.copy()
    def classify(o):
        if o > 100:   return 'High Orders'
        elif o >= 10: return 'Mid Orders'
        else:         return 'Low Orders'
    d['target_class'] = d['TOTAL_ORDERS'].apply(classify)
    return d


def create_target_multi(df):
    """
    Trial 5: Combined Sales rank + Engagement rank -> 3 tiers.
    
    Sales rank: High=3, Mid=2, Low=1 (based on TOTAL_SALES thresholds)
    Engagement rank: High=3, Mid=2, Low=1 (based on ENGAGEMENT_RATE thresholds)
    Combined score = sales_rank + engagement_rank (range 2-6)
    Tiers: Top (5-6) / Mid (3-4) / Low (2)
    """
    d = df.copy()
    
    # Sales rank
    def sales_rank(s):
        if s > 10000:   return 3
        elif s >= 1000: return 2
        else:           return 1
    
    # Engagement rank
    def eng_rank(e):
        if e > 5:    return 3
        elif e >= 2: return 2
        else:        return 1
    
    d['_sales_rank'] = d['TOTAL_SALES'].apply(sales_rank)
    d['_eng_rank']   = d['INSTAGRAM_ENGAGEMENT_RATE'].apply(eng_rank)
    d['_combined']   = d['_sales_rank'] + d['_eng_rank']
    
    def classify(score):
        if score >= 5: return 'Top Tier'
        elif score >= 3: return 'Mid Tier'
        else:          return 'Low Tier'
    
    d['target_class'] = d['_combined'].apply(classify)
    d.drop(columns=['_sales_rank', '_eng_rank', '_combined'], inplace=True)
    return d


def create_target_binary(df):
    """Trial 6: Binary — High Performer (sales > $10K) = 1, Rest = 0."""
    d = df.copy()
    d['target_class'] = (d['TOTAL_SALES'] > 10000).astype(int).map({1: 'High', 0: 'Rest'})
    return d


# ─────────────────────────────────────────────────────────────────────────────
# DATA PREPARATION
# ─────────────────────────────────────────────────────────────────────────────

def prepare_data(df, target_col='target_class', test_size=0.2, random_state=42,
                 exclude_cols=None):
    """
    Prepare train/test split with stratification.
    
    Parameters
    ----------
    df : DataFrame
        The full DataFrame with features and target column.
    target_col : str
        Name of the target column.
    test_size : float
        Fraction for test set.
    random_state : int
        Random seed for reproducibility.
    exclude_cols : list, optional
        Additional feature columns to exclude.
        
    Returns
    -------
    X_train, X_test, y_train, y_test, label_encoder, feature_names
    """
    features = get_feature_columns(exclude_cols=exclude_cols)
    # Only use columns that exist in the DataFrame
    features = [f for f in features if f in df.columns]
    
    le = LabelEncoder()
    y = le.fit_transform(df[target_col])
    X = df[features].fillna(0)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, le, features
