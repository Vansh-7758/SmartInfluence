import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder, QuantileTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import warnings
import os
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CSV_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'Influencer_Classified_with_growth.csv')
df = pd.read_csv(CSV_PATH)
df['niche']       = df['niche'].fillna('').str.strip().str.lower()
df['description'] = df['description'].fillna('').str.strip().str.lower()
df['ACCOUNTNAME'] = df['ACCOUNTNAME'].fillna('').str.strip()

# ─────────────────────────────────────────────────────────────────────────────
# 2. TARGET — TOTAL_SALES classification
#    High Performer : > $10,000
#    Mid Performer  : $1,000 – $10,000
#    Low Performer  : < $1,000
#
#    Excluded (leakage): TOTAL_ORDERS, NEW_CUSTOMERS, AVG_ORDER_SIZE,
#    TOTAL_COMMISSION, PERCENTAGE_OF_BRAND_SALES
# ─────────────────────────────────────────────────────────────────────────────
def assign_sales_class(sales):
    if sales > 10000:   return 'High Performer'
    elif sales >= 1000: return 'Mid Performer'
    else:               return 'Low Performer'

df['sales_class'] = df['TOTAL_SALES'].apply(assign_sales_class)

print("\n  Sales class distribution:")
print(df['sales_class'].value_counts().to_string())

# ─────────────────────────────────────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
def engineer_features(d):
    d = d.copy()
    # activity ratios
    d['engagement_per_post'] = d['INSTAGRAM_TOTAL_ENGAGEMENT'] / (d['INSTAGRAM_TOTAL_POSTS'] + 1)
    d['posting_consistency'] = d['INSTAGRAM_POSTS_LAST_30_DAYS'] / (d['INSTAGRAM_POSTS_LAST_90_DAYS'] + 1)
    d['posting_momentum']    = d['INSTAGRAM_POSTS_LAST_90_DAYS'] / (d['INSTAGRAM_POSTS_LAST_180_DAYS'] + 1)
    d['likes_per_post']      = d['INSTAGRAM_TOTAL_LIKES'] / (d['INSTAGRAM_TOTAL_POSTS'] + 1)
    d['comments_per_post']   = d['INSTAGRAM_TOTAL_COMMENTS'] / (d['INSTAGRAM_TOTAL_POSTS'] + 1)
    d['clicks_per_post']     = d['REFERRAL_LINK_TOTAL_CLICKS'] / (d['INSTAGRAM_TOTAL_POSTS'] + 1)
    d['campaign_commitment'] = d['CAMPAIGN_OPT_INS'] / (d['INSTAGRAM_TOTAL_POSTS'] + 1)
    # quality signals
    d['virality_score']      = d['INSTAGRAM_TOTAL_COMMENTS'] / (d['INSTAGRAM_TOTAL_LIKES'] + 1)
    d['reach_efficiency']    = d['REFERRAL_LINK_TOTAL_CLICKS'] / (d['INSTAGRAM_FOLLOWER_COUNT'] + 1)
    d['engagement_density']  = d['INSTAGRAM_TOTAL_ENGAGEMENT'] / (d['INSTAGRAM_FOLLOWER_COUNT'] + 1)
    # interaction features
    d['engagement_x_growth'] = d['INSTAGRAM_ENGAGEMENT_RATE'] * d['Growth Rate']
    d['absolute_engagement'] = d['INSTAGRAM_FOLLOWER_COUNT'] * d['INSTAGRAM_ENGAGEMENT_RATE'] / 100
    d['follower_x_growth']   = d['INSTAGRAM_FOLLOWER_COUNT'] * d['Growth Rate']
    d['eng_x_clicks']        = d['INSTAGRAM_ENGAGEMENT_RATE'] * d['REFERRAL_LINK_TOTAL_CLICKS']
    d['eng_rate_sq']         = d['INSTAGRAM_ENGAGEMENT_RATE'] ** 2
    # log transforms
    d['log_followers']       = np.log1p(d['INSTAGRAM_FOLLOWER_COUNT'])
    d['log_engagement']      = np.log1p(d['INSTAGRAM_TOTAL_ENGAGEMENT'])
    d['log_likes']           = np.log1p(d['INSTAGRAM_TOTAL_LIKES'])
    d['log_clicks']          = np.log1p(d['REFERRAL_LINK_TOTAL_CLICKS'])
    # pagerank-style influence score (adapted from Paper 2)
    max_f = d['INSTAGRAM_FOLLOWER_COUNT'].max() or 1
    max_e = d['INSTAGRAM_ENGAGEMENT_RATE'].max() or 1
    max_g = d['Growth Rate'].max() or 1
    d['influence_score']    = (
        0.4 * (d['INSTAGRAM_FOLLOWER_COUNT'] / max_f) +
        0.4 * (d['INSTAGRAM_ENGAGEMENT_RATE'] / max_e) +
        0.2 * (d['Growth Rate'] / max_g)
    )
    d['influence_x_clicks'] = d['influence_score'] * d['REFERRAL_LINK_TOTAL_CLICKS']
    # brand-level aggregates
    for col in ['INSTAGRAM_ENGAGEMENT_RATE', 'REFERRAL_LINK_TOTAL_CLICKS', 'Growth Rate']:
        d[f'brand_mean_{col}'] = d.groupby('ACCOUNTNAME')[col].transform('mean')
        d[f'vs_brand_{col}']   = d[col] / (d[f'brand_mean_{col}'] + 1)
    # niche-level aggregates
    d['niche_freq']          = d['niche'].map(d['niche'].value_counts())
    d['niche_mean_eng']      = d.groupby('niche')['INSTAGRAM_ENGAGEMENT_RATE'].transform('mean')
    d['vs_niche_eng']        = d['INSTAGRAM_ENGAGEMENT_RATE'] / (d['niche_mean_eng'] + 1)
    return d

df = engineer_features(df)

# KMeans cluster features
cluster_feats   = ['INSTAGRAM_ENGAGEMENT_RATE', 'Growth Rate', 'INSTAGRAM_FOLLOWER_COUNT',
                   'INSTAGRAM_TOTAL_LIKES', 'REFERRAL_LINK_TOTAL_CLICKS']
kmeans          = KMeans(n_clusters=5, random_state=42, n_init=10)
df['fresh_cluster'] = kmeans.fit_predict(df[cluster_feats].fillna(0))
df['cluster_dist']  = kmeans.transform(df[cluster_feats].fillna(0)).min(axis=1)

# TF-IDF on niche + description (adapted from Paper 1)
df['combined_text']   = df['niche'] + ' ' + df['description']
tfidf_feat            = TfidfVectorizer(max_features=20, stop_words='english')
tfidf_feat_matrix     = tfidf_feat.fit_transform(df['combined_text']).toarray()
tfidf_cols            = [f'tfidf_{i}' for i in range(20)]
for i, c in enumerate(tfidf_cols):
    df[c] = tfidf_feat_matrix[:, i]

# Quantile transform on skewed features
qt          = QuantileTransformer(output_distribution='normal', random_state=42)
skewed_cols = ['INSTAGRAM_FOLLOWER_COUNT', 'INSTAGRAM_TOTAL_LIKES',
               'INSTAGRAM_TOTAL_ENGAGEMENT', 'REFERRAL_LINK_TOTAL_CLICKS']
qt_arr      = qt.fit_transform(df[skewed_cols].fillna(0))
qt_cols     = [f'qt_{c}' for c in skewed_cols]
for i, c in enumerate(qt_cols):
    df[c] = qt_arr[:, i]

FEATURES = [f for f in [
    'INSTAGRAM_ENGAGEMENT_RATE', 'Growth Rate', 'INSTAGRAM_FOLLOWER_COUNT',
    'INSTAGRAM_TOTAL_POSTS', 'INSTAGRAM_POSTS_LAST_30_DAYS',
    'INSTAGRAM_POSTS_LAST_90_DAYS', 'INSTAGRAM_POSTS_LAST_180_DAYS',
    'INSTAGRAM_POSTS_IN_LAST_1_YEAR', 'INSTAGRAM_TOTAL_LIKES',
    'INSTAGRAM_TOTAL_COMMENTS', 'INSTAGRAM_TOTAL_ENGAGEMENT',
    'COMMISSION_PERCENTAGE', 'REFERRAL_LINK_TOTAL_CLICKS',
    'CAMPAIGN_OPT_INS', 'FLAT_FEE_COMMISSION',
    'engagement_per_post', 'posting_consistency', 'posting_momentum',
    'likes_per_post', 'comments_per_post', 'clicks_per_post',
    'campaign_commitment', 'virality_score', 'reach_efficiency',
    'engagement_density', 'engagement_x_growth', 'absolute_engagement',
    'follower_x_growth', 'eng_x_clicks', 'eng_rate_sq',
    'log_followers', 'log_engagement', 'log_likes', 'log_clicks',
    'influence_score', 'influence_x_clicks',
    'brand_mean_INSTAGRAM_ENGAGEMENT_RATE', 'vs_brand_INSTAGRAM_ENGAGEMENT_RATE',
    'brand_mean_REFERRAL_LINK_TOTAL_CLICKS', 'vs_brand_REFERRAL_LINK_TOTAL_CLICKS',
    'brand_mean_Growth Rate', 'vs_brand_Growth Rate',
    'niche_freq', 'niche_mean_eng', 'vs_niche_eng',
    'fresh_cluster', 'cluster_dist',
] + qt_cols + tfidf_cols if f in df.columns]

# ─────────────────────────────────────────────────────────────────────────────
# 4. ENCODE + SPLIT + TRAIN
# ─────────────────────────────────────────────────────────────────────────────
le = LabelEncoder()
df['target'] = le.fit_transform(df['sales_class'])

X = df[FEATURES].fillna(0)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = XGBClassifier(
    n_estimators     = 400,
    max_depth        = 5,
    learning_rate    = 0.05,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    min_child_weight = 5,
    gamma            = 0.2,
    reg_alpha        = 0.5,
    reg_lambda       = 1.5,
    eval_metric      = 'mlogloss',
    random_state     = 42,
)
model.fit(X_train, y_train)

# ─────────────────────────────────────────────────────────────────────────────
# 5. ACCURACY REPORT
# ─────────────────────────────────────────────────────────────────────────────
y_pred          = model.predict(X_test)
test_accuracy   = accuracy_score(y_test, y_pred)
train_accuracy  = accuracy_score(y_train, model.predict(X_train))
bal_acc         = balanced_accuracy_score(y_test, y_pred)
gap             = train_accuracy - test_accuracy

cv              = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores       = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
cv_mean         = cv_scores.mean()
cv_std          = cv_scores.std()

# variables exposed for frontend
conf_matrix      = confusion_matrix(y_test, y_pred).tolist()
feat_importances = model.feature_importances_.tolist()
target_classes   = list(le.classes_)
class_dist       = df['sales_class'].value_counts().to_dict()

top_features_idx  = np.argsort(model.feature_importances_)[-6:]
top_feature_names = [FEATURES[i] for i in top_features_idx]
corr_matrix       = df[top_feature_names].corr().round(2).values.tolist()

print("\n" + "="*62)
print("         XGBOOST MODEL — ACCURACY REPORT")
print("="*62)
print(f"\n  Target variable      : Sales Performance Class")
print(f"  Thresholds           : High > $10K | Mid $1K-$10K | Low < $1K")
print(f"  Features             : {len(FEATURES)}")
print(f"\n  Train Accuracy       : {train_accuracy * 100:.2f}%")
print(f"  Train/Test Accuracy  : {test_accuracy * 100:.2f}%")
print(f"  Cross-Val Accuracy   : {cv_mean * 100:.2f}% (+/- {cv_std * 100:.2f}%)")
print(f"  CV Scores per fold   : {[f'{s*100:.1f}%' for s in cv_scores]}")
print(f"  Balanced Accuracy    : {bal_acc * 100:.2f}%")
print(f"  Train-Test Gap       : {gap * 100:.2f}%")
print(f"\n{classification_report(y_test, y_pred, target_names=le.classes_)}")
print("="*62)

# ─────────────────────────────────────────────────────────────────────────────
# 6. TFIDF FOR BRAND DESCRIPTION MATCHING
# ─────────────────────────────────────────────────────────────────────────────
brand_df     = df[['ACCOUNTNAME', 'niche', 'description']].drop_duplicates('ACCOUNTNAME').reset_index(drop=True)
tfidf        = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(brand_df['description'])

def get_similar_brands(input_desc, top_n=5):
    input_vec    = tfidf.transform([input_desc.lower()])
    similarities = cosine_similarity(input_vec, tfidf_matrix).flatten()
    top_indices  = similarities.argsort()[::-1][:top_n]
    return [brand_df.iloc[i]['ACCOUNTNAME'] for i in top_indices if similarities[i] > 0]

# ─────────────────────────────────────────────────────────────────────────────
# 7. RANKING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
SALES_RANK      = {'High Performer': 3, 'Mid Performer': 2, 'Low Performer': 1}
HIGH_ENG_THRESH = 5.0

def rank_and_display(filtered, brand_name, top_n, return_data=False):
    if filtered.empty:
        if return_data: return {"main": [], "hidden": []}
        print("\n  No influencers found.\n")
        return

    filtered     = filtered.copy()
    X_input      = filtered[FEATURES].fillna(0)
    pred_labels  = le.inverse_transform(model.predict(X_input))
    pred_proba   = model.predict_proba(X_input)
    high_idx     = list(le.classes_).index('High Performer')

    filtered['predicted_class'] = pred_labels
    filtered['sales_score']     = pd.Series(pred_labels, index=filtered.index).map(SALES_RANK)
    filtered['confidence']      = pred_proba[:, high_idx] * 100

    eng_max    = filtered['INSTAGRAM_ENGAGEMENT_RATE'].max() or 1
    growth_max = filtered['Growth Rate'].max() or 1
    conf_max   = filtered['confidence'].max() or 1

    filtered['rank_score'] = (
        (filtered['sales_score'] / 3)                         * 0.40 +
        (filtered['INSTAGRAM_ENGAGEMENT_RATE'] / eng_max)     * 0.30 +
        (filtered['Growth Rate'] / growth_max)                * 0.20 +
        (filtered['confidence'] / conf_max)                   * 0.10
    )

    ranked = filtered.sort_values('rank_score', ascending=False).head(top_n)
    main   = ranked[ranked['predicted_class'] == 'High Performer']
    hidden = ranked[
        (ranked['predicted_class'].isin(['Low Performer', 'Mid Performer'])) &
        (ranked['INSTAGRAM_ENGAGEMENT_RATE'] >= HIGH_ENG_THRESH)
    ]

    if return_data:
        return {
            "main":   main.fillna("").to_dict(orient="records"),
            "hidden": hidden.fillna("").to_dict(orient="records")
        }

    def print_table(rows, title):
        if rows.empty:
            return
        print(f"\n  {'─'*70}")
        print(f"  {title}")
        print(f"  {'─'*70}")
        print(f"  {'#':<4} {'Name':<28} {'Predicted':<16} {'Eng%':<9} {'Growth':<9} {'Conf%'}")
        print(f"  {'─'*70}")
        for i, (_, row) in enumerate(rows.iterrows(), 1):
            symbol = "★" if row['predicted_class'] == 'High Performer' else ("◆" if row['predicted_class'] == 'Mid Performer' else "○")
            print(
                f"  {i:<4} {str(row['NAME'])[:27]:<28} "
                f"{symbol} {row['predicted_class']:<14} "
                f"{row['INSTAGRAM_ENGAGEMENT_RATE']:<9.2f}"
                f"{row['Growth Rate']:<9.2f}"
                f"{row['confidence']:<6.1f}"
            )

    print(f"\n{'='*74}")
    print(f"  RANKED INFLUENCERS FOR : {brand_name.upper()}")
    print(f"{'='*74}")
    print_table(main,   "★  HIGH PERFORMERS — Predicted sales > $10,000")
    print_table(hidden, "◆○  MID / LOW — High Engagement (worth considering)")
    if main.empty and hidden.empty:
        print_table(ranked, "All ranked influencers")
    print(f"\n  Legend   : ★ High Performer (>$10K)   ◆ Mid ($1K-$10K)   ○ Low (<$1K)")
    print(f"  Hidden gems threshold : Engagement > {HIGH_ENG_THRESH}%")
    print(f"{'='*74}\n")

# ─────────────────────────────────────────────────────────────────────────────
# 8. MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline(brand_name, niches_input, brand_description, top_n=10, return_data=False):
    niches = [n.strip().lower() for n in niches_input.split(',') if n.strip()]
    if not return_data:
        print(f"\n  Niches to match : {niches}")

    # Step 1 — niche filter
    niche_mask     = df['niche'].apply(lambda n: any(niche in n or n in niche for niche in niches))
    niche_filtered = df[niche_mask]
    if not return_data: print(f"  [Niche Filter]  {len(niche_filtered)} influencers found.")

    # Step 2 — description TF-IDF match
    desc_filtered = pd.DataFrame()
    if brand_description.strip():
        similar_brands = get_similar_brands(brand_description, top_n=5)
        if not return_data: print(f"  [Desc Match]    Similar brands : {similar_brands}")
        desc_filtered  = df[df['ACCOUNTNAME'].isin(similar_brands)]
        if not return_data: print(f"  [Desc Match]    {len(desc_filtered)} influencers from description match.")

    # Step 3 — merge and deduplicate
    combined = pd.concat([niche_filtered, desc_filtered]).drop_duplicates(subset='INFLUENCER_ID')
    if not return_data: print(f"  [Combined]      {len(combined)} unique influencers going into ranking.")

    if combined.empty:
        if return_data: return {"main": [], "hidden": []}
        print("\n  No influencers found. Try broader niche keywords.\n")
        return

    return rank_and_display(combined, brand_name, top_n, return_data)

# ─────────────────────────────────────────────────────────────────────────────
# 9. USER INPUT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n  Model ready.")
    print(f"  Train/Test Accuracy : {test_accuracy * 100:.2f}%")
    print(f"  Cross-Val Accuracy  : {cv_mean * 100:.2f}% (+/- {cv_std * 100:.2f}%)")
    print(f"  Balanced Accuracy   : {bal_acc * 100:.2f}%")
    print("-" * 60)
    brand_name  = input("  Brand name                         : ").strip()
    niches      = input("  Brand niche(s) (comma separated)   : ").strip()
    brand_desc  = input("  Brand description (or press Enter) : ").strip()
    top_n_input = input("  How many results?                  : ").strip()
    top_n       = int(top_n_input) if top_n_input.isdigit() else 10
    print("-" * 60)
    run_pipeline(brand_name, niches, brand_desc, top_n)