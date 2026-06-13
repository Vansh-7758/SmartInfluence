import pandas as pd
import numpy as np
from catboost import CatBoostClassifier  # Swapped from XGBoost
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
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CSV_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'Influencer_Classified_with_growth.csv')
df = pd.read_csv(CSV_PATH)
df['niche']       = df['niche'].fillna('').str.strip().str.lower()
df['description'] = df['description'].fillna('').str.strip().str.lower()
df['ACCOUNTNAME'] = df['ACCOUNTNAME'].fillna('').str.strip()

# ─────────────────────────────────────────────────────────────────────────────
# 2. CREATE SALES-BASED TARGET
# ─────────────────────────────────────────────────────────────────────────────
def assign_sales_class(sales):
    if sales > 10000:
        return 'High Performer'
    elif sales >= 1000:
        return 'Mid Performer'
    else:
        return 'Low Performer'

df['sales_class'] = df['TOTAL_SALES'].apply(assign_sales_class)

print("\n  Sales class distribution:")
print(df['sales_class'].value_counts().to_string())

# ─────────────────────────────────────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
def engineer_features(d):
    d = d.copy()
    d['engagement_per_post'] = d['INSTAGRAM_TOTAL_ENGAGEMENT'] / (d['INSTAGRAM_TOTAL_POSTS'] + 1)
    d['posting_consistency'] = d['INSTAGRAM_POSTS_LAST_30_DAYS'] / (d['INSTAGRAM_POSTS_LAST_90_DAYS'] + 1)
    d['posting_momentum']    = d['INSTAGRAM_POSTS_LAST_90_DAYS'] / (d['INSTAGRAM_POSTS_LAST_180_DAYS'] + 1)
    d['engagement_x_growth'] = d['INSTAGRAM_ENGAGEMENT_RATE'] * d['Growth Rate']
    d['absolute_engagement'] = d['INSTAGRAM_FOLLOWER_COUNT'] * d['INSTAGRAM_ENGAGEMENT_RATE'] / 100
    d['log_followers']       = np.log1p(d['INSTAGRAM_FOLLOWER_COUNT'])
    d['log_engagement']      = np.log1p(d['INSTAGRAM_TOTAL_ENGAGEMENT'])
    d['log_likes']           = np.log1p(d['INSTAGRAM_TOTAL_LIKES'])
    d['log_clicks']          = np.log1p(d['REFERRAL_LINK_TOTAL_CLICKS'])
    d['clicks_per_post']     = d['REFERRAL_LINK_TOTAL_CLICKS'] / (d['INSTAGRAM_TOTAL_POSTS'] + 1)
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
    'engagement_x_growth', 'absolute_engagement', 'log_followers',
    'log_engagement', 'log_likes', 'log_clicks', 'clicks_per_post',
]

# ─────────────────────────────────────────────────────────────────────────────
# 4. TRAIN CATBOOST
# ─────────────────────────────────────────────────────────────────────────────
le = LabelEncoder()
df['target'] = le.fit_transform(df['sales_class'])

X = df[FEATURES].fillna(0)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# CatBoost initialization
model = CatBoostClassifier(
    iterations=300,
    depth=6,
    learning_rate=0.05,
    random_seed=42,
    verbose=0,             # Keep it quiet like XGBoost
    loss_function='MultiClass'
)
model.fit(X_train, y_train)

# ─────────────────────────────────────────────────────────────────────────────
# 5. ACCURACY REPORT
# ─────────────────────────────────────────────────────────────────────────────
y_pred          = model.predict(X_test)
test_accuracy   = accuracy_score(y_test, y_pred)

cv              = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores       = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
cv_mean         = cv_scores.mean()
cv_std          = cv_scores.std()

print("\n" + "="*60)
print("         CATBOOST MODEL — ACCURACY REPORT")
print("="*60)
print(f"\n  Target variable      : Sales Performance Class")
print(f"  Thresholds           : High > $10K | Mid $1K-$10K | Low < $1K")
print(f"\n  Train/Test Accuracy  : {test_accuracy * 100:.2f}%")
print(f"  Cross-Val Accuracy   : {cv_mean * 100:.2f}% (+/- {cv_std * 100:.2f}%)")
print(f"  CV Scores per fold   : {[f'{s*100:.1f}%' for s in cv_scores]}")
print(f"\n{classification_report(y_test, y_pred, target_names=le.classes_)}")
print("="*60)

# ─────────────────────────────────────────────────────────────────────────────
# 6. TFIDF ON BRAND DESCRIPTIONS
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

def rank_and_display(filtered, brand_name, top_n):
    if filtered.empty:
        print("\n  No influencers found.\n")
        return

    filtered     = filtered.copy()
    X_input      = filtered[FEATURES].fillna(0)
    
    # CatBoost returns predictions as 2D array, we flatten for inverse_transform
    raw_preds    = model.predict(X_input).flatten()
    pred_labels  = le.inverse_transform(raw_preds)
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
        (filtered['confidence'] / (conf_max + 1e-9))          * 0.10
    )

    ranked   = filtered.sort_values('rank_score', ascending=False).head(top_n)
    main     = ranked[ranked['predicted_class'] == 'High Performer']
    hidden   = ranked[
        (ranked['predicted_class'].isin(['Low Performer', 'Mid Performer'])) &
        (ranked['INSTAGRAM_ENGAGEMENT_RATE'] >= HIGH_ENG_THRESH)
    ]

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
def run_pipeline(brand_name, niches_input, brand_description, top_n=10):
    niches = [n.strip().lower() for n in niches_input.split(',') if n.strip()]
    print(f"\n  Niches to match : {niches}")

    niche_mask     = df['niche'].apply(lambda n: any(niche in n or n in niche for niche in niches))
    niche_filtered = df[niche_mask]
    print(f"  [Niche Filter]  {len(niche_filtered)} influencers found.")

    desc_filtered = pd.DataFrame()
    if brand_description.strip():
        similar_brands = get_similar_brands(brand_description, top_n=5)
        print(f"  [Desc Match]    Similar brands : {similar_brands}")
        desc_filtered  = df[df['ACCOUNTNAME'].isin(similar_brands)]
        print(f"  [Desc Match]    {len(desc_filtered)} influencers from description match.")

    combined = pd.concat([niche_filtered, desc_filtered]).drop_duplicates(subset='INFLUENCER_ID')
    print(f"  [Combined]      {len(combined)} unique influencers going into ranking.")

    if combined.empty:
        print("\n  No influencers found. Try broader niche keywords.\n")
        return

    rank_and_display(combined, brand_name, top_n)

# ─────────────────────────────────────────────────────────────────────────────
# 9. USER INPUT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n  CatBoost Model ready.")
    print(f"  Train/Test Accuracy : {test_accuracy * 100:.2f}%")
    print(f"  Cross-Val Accuracy  : {cv_mean * 100:.2f}% (+/- {cv_std * 100:.2f}%)")
    print("-" * 60)
    brand_name  = input("  Brand name                         : ").strip()
    niches      = input("  Brand niche(s) (comma separated)   : ").strip()
    brand_desc  = input("  Brand description (or press Enter) : ").strip()
    top_n_input = input("  How many results?                  : ").strip()
    top_n       = int(top_n_input) if top_n_input.isdigit() else 10
    print("-" * 60)

    run_pipeline(brand_name, niches, brand_desc, top_n)