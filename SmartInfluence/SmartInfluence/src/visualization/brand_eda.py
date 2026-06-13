"""
Exploratory Data Analysis (EDA) for Brand.csv
Generates key visualizations and saves them to brand_eda_visuals/
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_PATH = 'DataSet_Pattern/Brand.csv'
OUTPUT_DIR = 'brand_eda_visuals'

# Style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Create output folder
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("=" * 70)
print("EXPLORATORY DATA ANALYSIS - BRAND.CSV")
print("=" * 70)
print("\nLoading data...")
df = pd.read_csv(DATA_PATH)
print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"\nColumns: {list(df.columns)}")

# ---------------------------------------------------------------------------
# 1. Basic overview
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("1. BASIC OVERVIEW")
print("=" * 70)
print("\nFirst 5 rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)
print("\nMissing values per column:")
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({'Count': missing, 'Percent': missing_pct})
missing_df = missing_df[missing_df['Count'] > 0].sort_values('Count', ascending=False)
if len(missing_df) > 0:
    print(missing_df)
else:
    print("No missing values.")
print("\nNumeric summary (describe):")
print(df.describe())

# Key numeric columns for analysis
key_numeric = [
    'INSTAGRAM_FOLLOWER_COUNT', 'INSTAGRAM_ENGAGEMENT_RATE', 'INSTAGRAM_TOTAL_ENGAGEMENT',
    'YOUTUBE_FOLLOWER_COUNT', 'YOUTUBE_ENGAGEMENT_RATE', 'YOUTUBE_TOTAL_ENGAGEMENT',
    'TIKTOK_FOLLOWER_COUNT', 'TIKTOK_ENGAGEMENT_RATE', 'TIKTOK_TOTAL_ENGAGEMENT',
    'REFERRAL_LINK_TOTAL_CLICKS', 'AVG_ORDER_SIZE', 'TOTAL_SALES', 'TOTAL_COMMISSION',
    'TOTAL_ORDERS', 'NEW_CUSTOMERS', 'COMMISSION_PERCENTAGE', 'CAMPAIGN_OPT_INS'
]
key_numeric = [c for c in key_numeric if c in df.columns]

# ---------------------------------------------------------------------------
# 2. Correlation heatmap (key metrics only to keep readable)
# ---------------------------------------------------------------------------
corr_cols = [c for c in ['INSTAGRAM_FOLLOWER_COUNT', 'INSTAGRAM_ENGAGEMENT_RATE', 'INSTAGRAM_TOTAL_ENGAGEMENT',
                         'TOTAL_SALES', 'TOTAL_ORDERS', 'REFERRAL_LINK_TOTAL_CLICKS', 'AVG_ORDER_SIZE',
                         'NEW_CUSTOMERS', 'TIKTOK_FOLLOWER_COUNT', 'YOUTUBE_FOLLOWER_COUNT'] if c in df.columns]
corr_df = df[corr_cols].copy()
for c in corr_df.columns:
    corr_df[c] = pd.to_numeric(corr_df[c], errors='coerce')
corr_matrix = corr_df.corr()

fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title('Correlation Heatmap – Key Brand & Influencer Metrics', fontsize=14, pad=16)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '01_correlation_heatmap_key_metrics.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUTPUT_DIR}/01_correlation_heatmap_key_metrics.png")

# ---------------------------------------------------------------------------
# 3. Distribution – Instagram follower count (log scale if skewed)
# ---------------------------------------------------------------------------
ig_followers = pd.to_numeric(df['INSTAGRAM_FOLLOWER_COUNT'], errors='coerce').dropna()
ig_followers = ig_followers[ig_followers > 0]
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(np.log10(ig_followers + 1), bins=60, edgecolor='black', alpha=0.7)
ax.set_xlabel('log10(Instagram Follower Count + 1)')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Instagram Follower Count (log scale)')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '02_distribution_instagram_follower_count.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUTPUT_DIR}/02_distribution_instagram_follower_count.png")

# ---------------------------------------------------------------------------
# 4. Distribution – Engagement rates (Instagram, YouTube, TikTok)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
for ax, (col, label) in zip(axes, [
    ('INSTAGRAM_ENGAGEMENT_RATE', 'Instagram'),
    ('YOUTUBE_ENGAGEMENT_RATE', 'YouTube'),
    ('TIKTOK_ENGAGEMENT_RATE', 'TikTok')
]):
    if col not in df.columns:
        ax.set_visible(False)
        continue
    data = pd.to_numeric(df[col], errors='coerce').dropna()
    data = data[data > 0]
    if len(data) > 0:
        ax.hist(data.clip(upper=50), bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel(f'{label} Engagement Rate (%)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'{label} Engagement Rate')
    ax.grid(True, alpha=0.3)
plt.suptitle('Engagement Rate by Platform', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '03_distribution_engagement_rate_by_platform.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUTPUT_DIR}/03_distribution_engagement_rate_by_platform.png")

# ---------------------------------------------------------------------------
# 5. Distribution – Total sales and total orders
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, col, label in zip(axes, ['TOTAL_SALES', 'TOTAL_ORDERS'], ['Total Sales ($)', 'Total Orders']):
    if col not in df.columns:
        continue
    data = pd.to_numeric(df[col], errors='coerce').dropna()
    data = data[data > 0]
    if len(data) > 0:
        ax.hist(np.log10(data + 1), bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel(f'log10({label} + 1)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Distribution of {label}')
    ax.grid(True, alpha=0.3)
plt.suptitle('Business Metrics: Sales & Orders', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '04_distribution_sales_and_orders.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUTPUT_DIR}/04_distribution_sales_and_orders.png")

# ---------------------------------------------------------------------------
# 6. Box plots – Key metrics (outliers visible)
# ---------------------------------------------------------------------------
box_cols = [c for c in ['INSTAGRAM_FOLLOWER_COUNT', 'INSTAGRAM_ENGAGEMENT_RATE', 'TOTAL_SALES', 'TOTAL_ORDERS',
                        'REFERRAL_LINK_TOTAL_CLICKS'] if c in df.columns]
box_df = df[box_cols].apply(pd.to_numeric, errors='coerce')
box_df = box_df.replace(0, np.nan).dropna(how='all')
# Log-scale for highly skewed
for c in ['INSTAGRAM_FOLLOWER_COUNT', 'TOTAL_SALES', 'TOTAL_ORDERS', 'REFERRAL_LINK_TOTAL_CLICKS']:
    if c in box_df.columns:
        box_df[c] = np.log10(box_df[c].fillna(0) + 1)
fig, ax = plt.subplots(figsize=(12, 6))
box_df.boxplot(ax=ax)
ax.set_ylabel('log10(value + 1) for counts/sales')
ax.set_title('Box Plots of Key Metrics (log scale for counts/sales)')
plt.xticks(rotation=25, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '05_boxplot_key_metrics_outliers.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUTPUT_DIR}/05_boxplot_key_metrics_outliers.png")

# ---------------------------------------------------------------------------
# 7. Scatter – Instagram followers vs total sales
# ---------------------------------------------------------------------------
x = pd.to_numeric(df['INSTAGRAM_FOLLOWER_COUNT'], errors='coerce')
y = pd.to_numeric(df['TOTAL_SALES'], errors='coerce')
valid = x.notna() & y.notna() & (x > 0) & (y >= 0)
x, y = x[valid], y[valid]
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(x, y, alpha=0.4, s=15)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Instagram Follower Count (log)')
ax.set_ylabel('Total Sales ($) (log)')
ax.set_title('Total Sales vs Instagram Follower Count')
r = np.corrcoef(np.log10(x + 1), np.log10(y + 1))[0, 1]
ax.text(0.05, 0.95, f'r = {r:.3f}', transform=ax.transAxes, fontsize=12, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '06_scatter_instagram_followers_vs_total_sales.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUTPUT_DIR}/06_scatter_instagram_followers_vs_total_sales.png")

# ---------------------------------------------------------------------------
# 8. Scatter – Engagement rate vs total orders
# ---------------------------------------------------------------------------
x = pd.to_numeric(df['INSTAGRAM_ENGAGEMENT_RATE'], errors='coerce')
y = pd.to_numeric(df['TOTAL_ORDERS'], errors='coerce')
valid = x.notna() & y.notna() & (x >= 0) & (y >= 0)
x, y = x[valid], y[valid]
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(x, y, alpha=0.4, s=15)
ax.set_yscale('log')
ax.set_xlabel('Instagram Engagement Rate (%)')
ax.set_ylabel('Total Orders (log)')
ax.set_title('Total Orders vs Instagram Engagement Rate')
r = np.corrcoef(x, np.log10(y + 1))[0, 1]
ax.text(0.05, 0.95, f'r = {r:.3f}', transform=ax.transAxes, fontsize=12, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '07_scatter_engagement_rate_vs_total_orders.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUTPUT_DIR}/07_scatter_engagement_rate_vs_total_orders.png")

# ---------------------------------------------------------------------------
# 9. Top brands by total sales (ACCOUNTNAME)
# ---------------------------------------------------------------------------
sales_by_brand = df.groupby('ACCOUNTNAME').agg(
    total_sales=('TOTAL_SALES', lambda x: pd.to_numeric(x, errors='coerce').sum())
).reset_index()
sales_by_brand = sales_by_brand.sort_values('total_sales', ascending=False).head(15)
sales_by_brand = sales_by_brand[sales_by_brand['total_sales'] > 0]
fig, ax = plt.subplots(figsize=(10, 7))
y_pos = range(len(sales_by_brand))
ax.barh(y_pos, sales_by_brand['total_sales'])
ax.set_yticks(y_pos)
ax.set_yticklabels(sales_by_brand['ACCOUNTNAME'], fontsize=10)
ax.set_xlabel('Total Sales ($)')
ax.set_title('Top 15 Brands by Total Sales (aggregated by ACCOUNTNAME)')
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '08_top_brands_by_total_sales.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUTPUT_DIR}/08_top_brands_by_total_sales.png")

# ---------------------------------------------------------------------------
# 10. Top brands by Instagram engagement (total engagement)
# ---------------------------------------------------------------------------
eng_by_brand = df.groupby('ACCOUNTNAME').agg(
    total_engagement=('INSTAGRAM_TOTAL_ENGAGEMENT', lambda x: pd.to_numeric(x, errors='coerce').sum())
).reset_index()
eng_by_brand = eng_by_brand.sort_values('total_engagement', ascending=False).head(15)
eng_by_brand = eng_by_brand[eng_by_brand['total_engagement'] > 0]
fig, ax = plt.subplots(figsize=(10, 7))
y_pos = range(len(eng_by_brand))
ax.barh(y_pos, eng_by_brand['total_engagement'])
ax.set_yticks(y_pos)
ax.set_yticklabels(eng_by_brand['ACCOUNTNAME'], fontsize=10)
ax.set_xlabel('Total Instagram Engagement')
ax.set_title('Top 15 Brands by Instagram Total Engagement')
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '09_top_brands_by_instagram_engagement.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUTPUT_DIR}/09_top_brands_by_instagram_engagement.png")

# ---------------------------------------------------------------------------
# 11. Missing values summary (bar chart)
# ---------------------------------------------------------------------------
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=True)
if len(missing) > 0:
    fig, ax = plt.subplots(figsize=(10, max(6, len(missing) * 0.35)))
    missing.plot(kind='barh', ax=ax)
    ax.set_xlabel('Number of Missing Values')
    ax.set_title('Missing Values per Column')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '10_missing_values_by_column.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/10_missing_values_by_column.png")

# ---------------------------------------------------------------------------
# 12. Platform presence (share of records with non-null, non-zero platform data)
# ---------------------------------------------------------------------------
has_ig = (pd.to_numeric(df['INSTAGRAM_FOLLOWER_COUNT'], errors='coerce').fillna(0) > 0).sum()
has_yt = (pd.to_numeric(df['YOUTUBE_FOLLOWER_COUNT'], errors='coerce').fillna(0) > 0).sum()
has_tt = (pd.to_numeric(df['TIKTOK_FOLLOWER_COUNT'], errors='coerce').fillna(0) > 0).sum()
n = len(df)
fig, ax = plt.subplots(figsize=(8, 5))
platforms = ['Instagram', 'YouTube', 'TikTok']
counts = [has_ig, has_yt, has_tt]
colors = ['#E1306C', '#FF0000', '#000000']
bars = ax.bar(platforms, counts, color=colors)
ax.set_ylabel('Number of Records with Platform Data')
ax.set_title('Platform Presence in Dataset (non-zero follower count)')
for b, c in zip(bars, counts):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 50, f'{c:,}\n({100*c/n:.1f}%)', ha='center', fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '11_platform_presence_count.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUTPUT_DIR}/11_platform_presence_count.png")

# ---------------------------------------------------------------------------
# 13. Account name frequency (top brands by number of influencer records)
# ---------------------------------------------------------------------------
top_accounts = df['ACCOUNTNAME'].value_counts().head(15)
top_accounts = top_accounts[top_accounts.index.astype(str).str.strip() != '']
if len(top_accounts) > 0:
    fig, ax = plt.subplots(figsize=(10, 7))
    y_pos = range(len(top_accounts))
    ax.barh(y_pos, top_accounts.values)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_accounts.index, fontsize=10)
    ax.set_xlabel('Number of Influencer Records')
    ax.set_title('Top 15 Brands by Number of Influencer Records (ACCOUNTNAME)')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '12_top_brands_by_influencer_count.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/12_top_brands_by_influencer_count.png")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("EDA COMPLETE – VISUALS SAVED")
print("=" * 70)
print(f"All visualizations saved to: {OUTPUT_DIR}/")
for f in sorted(os.listdir(OUTPUT_DIR)):
    if f.endswith('.png'):
        print(f"  • {f}")
