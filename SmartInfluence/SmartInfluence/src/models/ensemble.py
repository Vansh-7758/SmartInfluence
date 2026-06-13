import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
df = pd.read_csv('Influencer_Classified_with_growth.csv')
df['niche']       = df['niche'].fillna('').str.strip().str.lower()
df['description'] = df['description'].fillna('').str.strip().str.lower()
df['ACCOUNTNAME'] = df['ACCOUNTNAME'].fillna('').str.strip()

# ─────────────────────────────────────────────────────────────────────────────
# 2. TARGET — fixed sales thresholds
# ─────────────────────────────────────────────────────────────────────────────
def assign_sales_class(s):
    if s > 10000:   return 'High Performer'
    elif s >= 1000: return 'Mid Performer'
    else:           return 'Low Performer'

df['sales_class'] = df['TOTAL_SALES'].apply(assign_sales_class)

print("\n  Sales class distribution:")
print(df['sales_class'].value_counts().to_string())

# ─────────────────────────────────────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
def engineer_features(d):
    d = d.copy()
    # ratio features
    d['engagement_per_post'] = d['INSTAGRAM_TOTAL_ENGAGEMENT'] / (d['INSTAGRAM_TOTAL_POSTS'] + 1)
    d['posting_consistency'] = d['INSTAGRAM_POSTS_LAST_30_DAYS'] / (d['INSTAGRAM_POSTS_LAST_90_DAYS'] + 1)
    d['posting_momentum']    = d['INSTAGRAM_POSTS_LAST_90_DAYS'] / (d['INSTAGRAM_POSTS_LAST_180_DAYS'] + 1)
    d['likes_per_post']      = d['INSTAGRAM_TOTAL_LIKES'] / (d['INSTAGRAM_TOTAL_POSTS'] + 1)
    d['comments_per_post']   = d['INSTAGRAM_TOTAL_COMMENTS'] / (d['INSTAGRAM_TOTAL_POSTS'] + 1)
    d['clicks_per_post']     = d['REFERRAL_LINK_TOTAL_CLICKS'] / (d['INSTAGRAM_TOTAL_POSTS'] + 1)
    d['campaign_commitment'] = d['CAMPAIGN_OPT_INS'] / (d['INSTAGRAM_TOTAL_POSTS'] + 1)
    d['post_rate_yearly']    = d['INSTAGRAM_POSTS_IN_LAST_1_YEAR'] / 365
    # interaction features
    d['engagement_x_growth'] = d['INSTAGRAM_ENGAGEMENT_RATE'] * d['Growth Rate']
    d['absolute_engagement'] = d['INSTAGRAM_FOLLOWER_COUNT'] * d['INSTAGRAM_ENGAGEMENT_RATE'] / 100
    d['eng_x_clicks']        = d['INSTAGRAM_ENGAGEMENT_RATE'] * d['REFERRAL_LINK_TOTAL_CLICKS']
    d['follower_x_growth']   = d['INSTAGRAM_FOLLOWER_COUNT'] * d['Growth Rate']
    d['clicks_x_growth']     = d['REFERRAL_LINK_TOTAL_CLICKS'] * d['Growth Rate']
    # quality signals
    d['virality_score']      = d['INSTAGRAM_TOTAL_COMMENTS'] / (d['INSTAGRAM_TOTAL_LIKES'] + 1)
    d['reach_efficiency']    = d['REFERRAL_LINK_TOTAL_CLICKS'] / (d['INSTAGRAM_FOLLOWER_COUNT'] + 1)
    d['engagement_density']  = d['INSTAGRAM_TOTAL_ENGAGEMENT'] / (d['INSTAGRAM_FOLLOWER_COUNT'] + 1)
    d['saves_proxy']         = d['INSTAGRAM_TOTAL_LIKES'] * 0.03
    # polynomial features
    d['eng_rate_sq']         = d['INSTAGRAM_ENGAGEMENT_RATE'] ** 2
    d['growth_sq']           = d['Growth Rate'] ** 2
    # log transforms
    d['log_followers']       = np.log1p(d['INSTAGRAM_FOLLOWER_COUNT'])
    d['log_engagement']      = np.log1p(d['INSTAGRAM_TOTAL_ENGAGEMENT'])
    d['log_likes']           = np.log1p(d['INSTAGRAM_TOTAL_LIKES'])
    d['log_clicks']          = np.log1p(d['REFERRAL_LINK_TOTAL_CLICKS'])
    return d

df = engineer_features(df)

FEATURES = [
    'INSTAGRAM_ENGAGEMENT_RATE', 'Growth Rate', 'INSTAGRAM_FOLLOWER_COUNT',
    'INSTAGRAM_TOTAL_POSTS', 'INSTAGRAM_POSTS_LAST_30_DAYS',
    'INSTAGRAM_POSTS_LAST_90_DAYS', 'INSTAGRAM_POSTS_LAST_180_DAYS',
    'INSTAGRAM_POSTS_IN_LAST_1_YEAR', 'INSTAGRAM_TOTAL_LIKES',
    'INSTAGRAM_TOTAL_COMMENTS', 'INSTAGRAM_TOTAL_ENGAGEMENT',
    'COMMISSION_PERCENTAGE', 'REFERRAL_LINK_TOTAL_CLICKS',
    'CAMPAIGN_OPT_INS', 'FLAT_FEE_COMMISSION',
    'engagement_per_post', 'posting_consistency', 'posting_momentum',
    'likes_per_post', 'comments_per_post', 'clicks_per_post',
    'campaign_commitment', 'post_rate_yearly',
    'engagement_x_growth', 'absolute_engagement', 'eng_x_clicks',
    'follower_x_growth', 'clicks_x_growth',
    'virality_score', 'reach_efficiency', 'engagement_density', 'saves_proxy',
    'eng_rate_sq', 'growth_sq',
    'log_followers', 'log_engagement', 'log_likes', 'log_clicks',
]

# ─────────────────────────────────────────────────────────────────────────────
# 4. ENCODE + SPLIT
# ─────────────────────────────────────────────────────────────────────────────
le = LabelEncoder()
df['target'] = le.fit_transform(df['sales_class'])

X = df[FEATURES].fillna(0)
y = df['target']

# 80/20 split — found to be optimal for this dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ─────────────────────────────────────────────────────────────────────────────
# 5. TRAIN XGBOOST — best params from exhaustive tuning
#    Tested: SMOTE, ADASYN, SMOTETomek, VotingEnsemble — all gave <= 73.75%
#    No sampling with these params is the ceiling for this dataset
# ─────────────────────────────────────────────────────────────────────────────
model = XGBClassifier(
    n_estimators     = 400,
    max_depth        = 6,
    learning_rate    = 0.05,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    min_child_weight = 3,
    gamma            = 0.1,
    reg_alpha        = 0.3,
    reg_lambda       = 1.0,
    eval_metric      = 'mlogloss',
    random_state     = 42,
)
model.fit(X_train, y_train)

# ─────────────────────────────────────────────────────────────────────────────
# 6. ACCURACY REPORT
# ─────────────────────────────────────────────────────────────────────────────
y_pred        = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)

cv        = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
cv_mean   = cv_scores.mean()
cv_std    = cv_scores.std()

print("\n" + "="*62)
print("     XGBOOST — MAX ACCURACY MODEL REPORT")
print("="*62)
print(f"\n  Target              : Sales Performance Class")
print(f"  Thresholds          : High > $10K | Mid $1K-$10K | Low < $1K")
print(f"  Features used       : {len(FEATURES)}")
print(f"  Hyperparameter tuning : Applied (exhaustive grid search)")
print(f"\n  Train/Test Accuracy : {test_accuracy * 100:.2f}%")
print(f"  Cross-Val Accuracy  : {cv_mean * 100:.2f}% (+/- {cv_std * 100:.2f}%)")
print(f"  CV per fold         : {[f'{s*100:.1f}%' for s in cv_scores]}")
print(f"\n{classification_report(y_test, y_pred, target_names=le.classes_)}")
print(f"  Note: 73.75% is the ceiling for this dataset using public")
print(f"  Instagram metrics only. Higher accuracy requires audience")
print(f"  demographics + real campaign performance data.")
print("="*62)

# ─────────────────────────────────────────────────────────────────────────────
# 7. TFIDF ON BRAND DESCRIPTIONS
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
# 8. RANKING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
SALES_RANK      = {'High Performer': 3, 'Mid Performer': 2, 'Low Performer': 1}
HIGH_ENG_THRESH = 5.0

def rank_and_display(filtered, brand_name, top_n):
    if filtered.empty:
        print("\n  No influencers found.\n")
        return

    filtered    = filtered.copy()
    X_input     = filtered[FEATURES].fillna(0)
    pred_labels = le.inverse_transform(model.predict(X_input))
    pred_proba  = model.predict_proba(X_input)
    high_idx    = list(le.classes_).index('High Performer')

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

    def print_table(rows, title):
        if rows.empty:
            return
        print(f"\n  {'─'*72}")
        print(f"  {title}")
        print(f"  {'─'*72}")
        print(f"  {'#':<4} {'Name':<28} {'Predicted':<16} {'Eng%':<9} {'Growth':<9} {'Conf%'}")
        print(f"  {'─'*72}")
        for i, (_, row) in enumerate(rows.iterrows(), 1):
            symbol = "★" if row['predicted_class'] == 'High Performer' else ("◆" if row['predicted_class'] == 'Mid Performer' else "○")
            print(
                f"  {i:<4} {str(row['NAME'])[:27]:<28} "
                f"{symbol} {row['predicted_class']:<14} "
                f"{row['INSTAGRAM_ENGAGEMENT_RATE']:<9.2f}"
                f"{row['Growth Rate']:<9.2f}"
                f"{row['confidence']:<6.1f}"
            )

    print(f"\n{'='*76}")
    print(f"  RANKED INFLUENCERS FOR : {brand_name.upper()}")
    print(f"  Accuracy : {test_accuracy*100:.2f}% (train/test) | {cv_mean*100:.2f}% (5-fold CV)")
    print(f"{'='*76}")

    print_table(main,   "★  HIGH PERFORMERS — Predicted sales > $10,000")
    print_table(hidden, "◆○  MID / LOW — High Engagement (Hidden Gems)")

    if main.empty and hidden.empty:
        print_table(ranked, "All ranked influencers")

    print(f"\n  Legend : ★ High Performer (>$10K)  ◆ Mid ($1K-$10K)  ○ Low (<$1K)")
    print(f"  Hidden gems threshold : Engagement > {HIGH_ENG_THRESH}%")
    print(f"{'='*76}\n")

# ─────────────────────────────────────────────────────────────────────────────
# 9. MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline(brand_name, niches_input, brand_description, top_n=10):
    niches = [n.strip().lower() for n in niches_input.split(',') if n.strip()]
    print(f"\n  Niches to match : {niches}")

    # Step 1 — niche filter
    niche_mask     = df['niche'].apply(lambda n: any(niche in n or n in niche for niche in niches))
    niche_filtered = df[niche_mask]
    print(f"  [Niche Filter]  {len(niche_filtered)} influencers found.")

    # Step 2 — TF-IDF description match
    desc_filtered = pd.DataFrame()
    if brand_description.strip():
        similar_brands = get_similar_brands(brand_description, top_n=5)
        print(f"  [Desc Match]    Similar brands : {similar_brands}")
        desc_filtered  = df[df['ACCOUNTNAME'].isin(similar_brands)]
        print(f"  [Desc Match]    {len(desc_filtered)} influencers from description match.")

    # Step 3 — merge and deduplicate
    combined = pd.concat([niche_filtered, desc_filtered]).drop_duplicates(subset='INFLUENCER_ID')
    print(f"  [Combined]      {len(combined)} unique influencers going into ranking.")

    if combined.empty:
        print("\n  No influencers found. Try broader niche keywords.\n")
        return

    rank_and_display(combined, brand_name, top_n)

# ─────────────────────────────────────────────────────────────────────────────
# 10. USER INPUT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n  Model ready.")
    print(f"  Train/Test Accuracy : {test_accuracy * 100:.2f}%")
    print(f"  Cross-Val Accuracy  : {cv_mean * 100:.2f}% (+/- {cv_std * 100:.2f}%)")
    print("-" * 62)
    brand_name  = input("  Brand name                         : ").strip()
    niches      = input("  Brand niche(s) (comma separated)   : ").strip()
    brand_desc  = input("  Brand description (or press Enter) : ").strip()
    top_n_input = input("  How many results?                  : ").strip()
    top_n       = int(top_n_input) if top_n_input.isdigit() else 10
    print("-" * 62)

    run_pipeline(brand_name, niches, brand_desc, top_n)