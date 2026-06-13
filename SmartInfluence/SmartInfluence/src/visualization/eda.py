import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Load the data
print("=" * 80)
print("EXPLORATORY DATA ANALYSIS - INFLUENCER METRICS")
print("=" * 80)
print("\nLoading data...")
df = pd.read_csv('Influencer_identification_metrics_instagram_10k_04_02.csv')

print(f"\nDataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"\nColumn Names:")
for i, col in enumerate(df.columns, 1):
    print(f"{i:2d}. {col}")

# Define main and secondary attributes early (used in cleaning section)
main_attributes = [
    'INSTAGRAM_FOLLOWER_COUNT',      # Primary: Reach/audience size
    'INSTAGRAM_ENGAGEMENT_RATE',      # Primary: Quality of engagement
    'INSTAGRAM_TOTAL_ENGAGEMENT',     # Primary: Total engagement volume
    'INSTAGRAM_TOTAL_POSTS',          # Primary: Content activity
    'TOTAL_SALES',                    # Primary: Business performance
    'TOTAL_ORDERS',                   # Primary: Conversion metrics
    'REFERRAL_LINK_TOTAL_CLICKS',     # Primary: Click-through performance
]

secondary_attributes = [
    'INSTAGRAM_TOTAL_LIKES',          # Secondary: Engagement breakdown
    'INSTAGRAM_TOTAL_COMMENTS',       # Secondary: Engagement quality indicator
    'INSTAGRAM_POSTS_LAST_30_DAYS',   # Secondary: Recent activity
    'INSTAGRAM_POSTS_LAST_90_DAYS',   # Secondary: Recent activity trend
    'AVG_ORDER_SIZE',                 # Secondary: Customer value
    'TOTAL_COMMISSION',               # Secondary: Revenue generated
    'NEW_CUSTOMERS',                  # Secondary: Customer acquisition
    'COMMISSION_PERCENTAGE',          # Secondary: Partnership terms
    'CAMPAIGN_OPT_INS',               # Secondary: Campaign participation
]

# ============================================================================
# 1. BASIC DATA OVERVIEW
# ============================================================================
print("\n" + "=" * 80)
print("1. BASIC DATA OVERVIEW")
print("=" * 80)

print("\nFirst few rows:")
print(df.head())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing Count': missing,
    'Missing Percentage': missing_pct
})
missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
if len(missing_df) > 0:
    print(missing_df)
else:
    print("No missing values found!")

print("\nBasic Statistics:")
print(df.describe())

# ============================================================================
# 2. DATA CLEANING
# ============================================================================
print("\n" + "=" * 80)
print("2. DATA CLEANING")
print("=" * 80)

# Store original shape
original_shape = df.shape
print(f"\nOriginal dataset shape: {original_shape[0]} rows, {original_shape[1]} columns")

# Create a copy for cleaning
df_cleaned = df.copy()
cleaning_log = []

# 2.1 Check for duplicates
print("\n" + "-" * 80)
print("2.1 DUPLICATE DETECTION")
print("-" * 80)

total_duplicates = df_cleaned.duplicated().sum()
duplicates_by_influencer_id = df_cleaned.duplicated(subset=['INFLUENCER_ID']).sum()
duplicates_by_instagram_id = df_cleaned.duplicated(subset=['INSTAGRAM_USER_ID']).sum()
duplicates_by_name_account = df_cleaned.duplicated(subset=['NAME', 'ACCOUNTNAME']).sum()

print(f"Total exact duplicates: {total_duplicates}")
print(f"Duplicates by INFLUENCER_ID: {duplicates_by_influencer_id}")
print(f"Duplicates by INSTAGRAM_USER_ID: {duplicates_by_instagram_id}")
print(f"Duplicates by NAME + ACCOUNTNAME: {duplicates_by_name_account}")

# Show sample duplicates
if duplicates_by_influencer_id > 0:
    print("\nSample duplicate records (by INFLUENCER_ID):")
    dup_mask = df_cleaned.duplicated(subset=['INFLUENCER_ID'], keep=False)
    dup_samples = df_cleaned[dup_mask].sort_values('INFLUENCER_ID').head(10)
    print(dup_samples[['INFLUENCER_ID', 'NAME', 'ACCOUNTNAME', 'INSTAGRAM_FOLLOWER_COUNT', 'TOTAL_SALES']].to_string(index=False))

# Remove exact duplicates
if total_duplicates > 0:
    df_cleaned = df_cleaned.drop_duplicates()
    cleaning_log.append(f"Removed {total_duplicates} exact duplicate rows")
    print(f"\n✓ Removed {total_duplicates} exact duplicate rows")

# For influencer-level duplicates, keep the record with highest engagement or sales
if duplicates_by_influencer_id > 0:
    # Group by INFLUENCER_ID and keep the row with highest TOTAL_SALES (or highest engagement if sales is 0)
    df_cleaned = df_cleaned.sort_values(['INFLUENCER_ID', 'TOTAL_SALES', 'INSTAGRAM_TOTAL_ENGAGEMENT'], 
                                        ascending=[True, False, False])
    df_cleaned = df_cleaned.drop_duplicates(subset=['INFLUENCER_ID'], keep='first')
    removed = original_shape[0] - len(df_cleaned) - total_duplicates
    if removed > 0:
        cleaning_log.append(f"Removed {removed} duplicate influencer records (kept highest performing)")
        print(f"✓ Removed {removed} duplicate influencer records (kept highest performing)")

# 2.2 Handle Missing Values
print("\n" + "-" * 80)
print("2.2 MISSING VALUES HANDLING")
print("-" * 80)

missing_before = df_cleaned.isnull().sum()
missing_cols = missing_before[missing_before > 0]

if len(missing_cols) > 0:
    print("\nMissing values before cleaning:")
    for col, count in missing_cols.items():
        pct = (count / len(df_cleaned)) * 100
        print(f"  {col}: {count} ({pct:.2f}%)")
    
    # Handle missing NAME - fill with "Unknown" or drop if critical
    if 'NAME' in missing_cols:
        df_cleaned['NAME'] = df_cleaned['NAME'].fillna('Unknown')
        cleaning_log.append(f"Filled {missing_cols['NAME']} missing NAME values with 'Unknown'")
        print(f"✓ Filled {missing_cols['NAME']} missing NAME values")
    
    # Handle missing INSTAGRAM_USER_ID - can be dropped or filled
    if 'INSTAGRAM_USER_ID' in missing_cols:
        # Fill with 0 or drop - depends on use case
        df_cleaned = df_cleaned.dropna(subset=['INSTAGRAM_USER_ID'])
        cleaning_log.append(f"Removed {missing_cols['INSTAGRAM_USER_ID']} rows with missing INSTAGRAM_USER_ID")
        print(f"✓ Removed {missing_cols['INSTAGRAM_USER_ID']} rows with missing INSTAGRAM_USER_ID")
else:
    print("No missing values found!")

# 2.3 Handle Invalid Values
print("\n" + "-" * 80)
print("2.3 INVALID VALUES HANDLING")
print("-" * 80)

invalid_log = []

# Check for negative values where they shouldn't exist
numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
non_negative_cols = ['INSTAGRAM_FOLLOWER_COUNT', 'INSTAGRAM_TOTAL_LIKES', 'INSTAGRAM_TOTAL_COMMENTS',
                     'INSTAGRAM_TOTAL_POSTS', 'INSTAGRAM_TOTAL_ENGAGEMENT', 'TOTAL_SALES', 
                     'TOTAL_ORDERS', 'REFERRAL_LINK_TOTAL_CLICKS', 'NEW_CUSTOMERS', 'CAMPAIGN_OPT_INS']

for col in non_negative_cols:
    if col in df_cleaned.columns:
        negative_count = (df_cleaned[col] < 0).sum()
        if negative_count > 0:
            invalid_log.append(f"{col}: {negative_count} negative values")
            # Replace negatives with 0
            df_cleaned.loc[df_cleaned[col] < 0, col] = 0
            print(f"✓ Fixed {negative_count} negative values in {col} (set to 0)")

# Check for impossible engagement rates (>100%)
if 'INSTAGRAM_ENGAGEMENT_RATE' in df_cleaned.columns:
    impossible_er = (df_cleaned['INSTAGRAM_ENGAGEMENT_RATE'] > 100).sum()
    if impossible_er > 0:
        # Cap at 100% or recalculate
        df_cleaned.loc[df_cleaned['INSTAGRAM_ENGAGEMENT_RATE'] > 100, 'INSTAGRAM_ENGAGEMENT_RATE'] = 100
        invalid_log.append(f"INSTAGRAM_ENGAGEMENT_RATE: {impossible_er} values > 100% (capped at 100%)")
        print(f"✓ Capped {impossible_er} engagement rates > 100%")

# Check for zero followers but non-zero engagement (data inconsistency)
if 'INSTAGRAM_FOLLOWER_COUNT' in df_cleaned.columns and 'INSTAGRAM_TOTAL_ENGAGEMENT' in df_cleaned.columns:
    inconsistent = ((df_cleaned['INSTAGRAM_FOLLOWER_COUNT'] == 0) & 
                    (df_cleaned['INSTAGRAM_TOTAL_ENGAGEMENT'] > 0)).sum()
    if inconsistent > 0:
        invalid_log.append(f"Found {inconsistent} records with 0 followers but >0 engagement")
        print(f"⚠ Found {inconsistent} records with 0 followers but >0 engagement (flagged)")

# Check for posts count inconsistencies
if 'INSTAGRAM_TOTAL_POSTS' in df_cleaned.columns and 'INSTAGRAM_POSTS_LAST_30_DAYS' in df_cleaned.columns:
    inconsistent_posts = (df_cleaned['INSTAGRAM_POSTS_LAST_30_DAYS'] > df_cleaned['INSTAGRAM_TOTAL_POSTS']).sum()
    if inconsistent_posts > 0:
        invalid_log.append(f"Found {inconsistent_posts} records where posts_30d > total_posts")
        print(f"⚠ Found {inconsistent_posts} records with inconsistent post counts (flagged)")

if len(invalid_log) == 0:
    print("No invalid values found!")

# 2.4 Handle Outliers (Flag, don't remove - for analysis)
print("\n" + "-" * 80)
print("2.4 OUTLIER ANALYSIS")
print("-" * 80)

print("Outliers detected (using IQR method - flagged but not removed):")
outlier_counts = {}
for col in main_attributes:
    if col in df_cleaned.columns:
        Q1 = df_cleaned[col].quantile(0.25)
        Q3 = df_cleaned[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = ((df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound)).sum()
        outlier_counts[col] = outliers
        if outliers > 0:
            print(f"  {col}: {outliers} outliers ({outliers/len(df_cleaned)*100:.2f}%)")

# Create outlier flag column (optional - for later analysis)
df_cleaned['IS_OUTLIER'] = False
for col in main_attributes:
    if col in df_cleaned.columns:
        Q1 = df_cleaned[col].quantile(0.25)
        Q3 = df_cleaned[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_cleaned.loc[(df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound), 'IS_OUTLIER'] = True

outlier_count = df_cleaned['IS_OUTLIER'].sum()
print(f"\nTotal records flagged as outliers: {outlier_count} ({outlier_count/len(df_cleaned)*100:.2f}%)")

# 2.5 Data Type Corrections
print("\n" + "-" * 80)
print("2.5 DATA TYPE CORRECTIONS")
print("-" * 80)

# Ensure integer columns are integers
int_cols = ['INFLUENCER_ID', 'INSTAGRAM_TOTAL_LIKES', 'INSTAGRAM_TOTAL_COMMENTS', 
            'INSTAGRAM_TOTAL_POSTS', 'INSTAGRAM_TOTAL_ENGAGEMENT', 'INSTAGRAM_FOLLOWER_COUNT',
            'TOTAL_ORDERS', 'REFERRAL_LINK_TOTAL_CLICKS', 'NEW_CUSTOMERS', 'CAMPAIGN_OPT_INS']

for col in int_cols:
    if col in df_cleaned.columns:
        # Convert to int, handling NaN
        df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce').fillna(0).astype(int)

print("✓ Converted numeric columns to appropriate integer types")

# 2.6 Summary of Cleaning
print("\n" + "-" * 80)
print("2.6 CLEANING SUMMARY")
print("-" * 80)

final_shape = df_cleaned.shape
rows_removed = original_shape[0] - final_shape[0]
cols_added = final_shape[1] - original_shape[1]

print(f"Original shape: {original_shape[0]} rows × {original_shape[1]} columns")
print(f"Final shape: {final_shape[0]} rows × {final_shape[1]} columns")
print(f"Rows removed: {rows_removed} ({rows_removed/original_shape[0]*100:.2f}%)")
if cols_added > 0:
    print(f"Columns added: {cols_added} (outlier flag)")

if cleaning_log:
    print("\nCleaning actions performed:")
    for action in cleaning_log:
        print(f"  • {action}")

# 2.7 Save Cleaned Dataset
print("\n" + "-" * 80)
print("2.7 SAVING CLEANED DATASET")
print("-" * 80)

cleaned_filename = 'Influencer_identification_metrics_instagram_10k_04_02_cleaned.csv'
df_cleaned.to_csv(cleaned_filename, index=False)
print(f"✓ Saved cleaned dataset to: {cleaned_filename}")

# Update df to cleaned version for rest of analysis
df = df_cleaned.copy()

# ============================================================================
# 3. IDENTIFY MAIN ATTRIBUTES (Primary Metrics)
# ============================================================================
print("\n" + "=" * 80)
print("2. MAIN ATTRIBUTES (Primary Metrics for Influencer Identification)")
print("=" * 80)

# Main attributes are typically the most important metrics for influencer evaluation
# (Already defined above)

print("\nMain Attributes Identified:")
for i, attr in enumerate(main_attributes, 1):
    print(f"{i}. {attr}")

# Statistics for main attributes
print("\n" + "-" * 80)
print("Statistics for Main Attributes:")
print("-" * 80)
main_stats = df[main_attributes].describe()
print(main_stats)

# ============================================================================
# 4. IDENTIFY SECONDARY ATTRIBUTES (Important Supporting Metrics)
# ============================================================================
print("\n" + "=" * 80)
print("4. SECONDARY ATTRIBUTES (Important Supporting Metrics)")
print("=" * 80)

# Secondary attributes provide context and additional insights
# (Already defined above)

print("\nSecondary Attributes Identified:")
for i, attr in enumerate(secondary_attributes, 1):
    print(f"{i}. {attr}")

# Statistics for secondary attributes
print("\n" + "-" * 80)
print("Statistics for Secondary Attributes:")
print("-" * 80)
secondary_stats = df[secondary_attributes].describe()
print(secondary_stats)

# ============================================================================
# 5. CORRELATION ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("5. CORRELATION ANALYSIS")
print("=" * 80)

# Combine main and secondary attributes for correlation
all_key_attributes = main_attributes + secondary_attributes
numeric_cols = [col for col in all_key_attributes if col in df.columns and df[col].dtype in ['int64', 'float64']]

correlation_matrix = df[numeric_cols].corr()

print("\nTop Correlations (Absolute Value > 0.5):")
corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        corr_val = correlation_matrix.iloc[i, j]
        if abs(corr_val) > 0.5:
            corr_pairs.append({
                'Attribute 1': correlation_matrix.columns[i],
                'Attribute 2': correlation_matrix.columns[j],
                'Correlation': corr_val
            })

corr_df = pd.DataFrame(corr_pairs).sort_values('Correlation', key=abs, ascending=False)
print(corr_df.to_string(index=False))

# ============================================================================
# 6. DISTRIBUTION ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("6. DISTRIBUTION ANALYSIS")
print("=" * 80)

# Check for skewness in main attributes
print("\nSkewness Analysis (|skew| > 1 indicates highly skewed):")
skewness_data = []
for attr in main_attributes:
    if attr in df.columns:
        skew_val = df[attr].skew()
        skewness_data.append({
            'Attribute': attr,
            'Skewness': skew_val,
            'Interpretation': 'Highly Right-Skewed' if skew_val > 1 else 'Highly Left-Skewed' if skew_val < -1 else 'Near Normal'
        })

skew_df = pd.DataFrame(skewness_data)
print(skew_df.to_string(index=False))

# ============================================================================
# 7. OUTLIER DETECTION (Detailed)
# ============================================================================
print("\n" + "=" * 80)
print("7. OUTLIER DETECTION (Using IQR Method)")
print("=" * 80)

outlier_summary = []
for attr in main_attributes:
    if attr in df.columns:
        Q1 = df[attr].quantile(0.25)
        Q3 = df[attr].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[attr] < lower_bound) | (df[attr] > upper_bound)]
        outlier_count = len(outliers)
        outlier_pct = (outlier_count / len(df)) * 100
        
        outlier_summary.append({
            'Attribute': attr,
            'Outliers Count': outlier_count,
            'Outlier Percentage': f"{outlier_pct:.2f}%",
            'Lower Bound': f"{lower_bound:.2f}",
            'Upper Bound': f"{upper_bound:.2f}"
        })

outlier_df = pd.DataFrame(outlier_summary)
print(outlier_df.to_string(index=False))

# ============================================================================
# 8. BUSINESS METRICS ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("8. BUSINESS METRICS ANALYSIS")
print("=" * 80)

# Calculate derived metrics
if 'TOTAL_ORDERS' in df.columns and 'TOTAL_ORDERS' in df.columns:
    df['CONVERSION_RATE'] = (df['TOTAL_ORDERS'] / df['REFERRAL_LINK_TOTAL_CLICKS'].replace(0, np.nan)) * 100
    df['REVENUE_PER_FOLLOWER'] = df['TOTAL_SALES'] / df['INSTAGRAM_FOLLOWER_COUNT'].replace(0, np.nan)
    df['ENGAGEMENT_PER_POST'] = df['INSTAGRAM_TOTAL_ENGAGEMENT'] / df['INSTAGRAM_TOTAL_POSTS'].replace(0, np.nan)

print("\nDerived Metrics Summary:")
derived_metrics = ['CONVERSION_RATE', 'REVENUE_PER_FOLLOWER', 'ENGAGEMENT_PER_POST']
for metric in derived_metrics:
    if metric in df.columns:
        print(f"\n{metric}:")
        print(f"  Mean: {df[metric].mean():.2f}")
        print(f"  Median: {df[metric].median():.2f}")
        print(f"  Std Dev: {df[metric].std():.2f}")

# ============================================================================
# 9. CATEGORICAL ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("9. CATEGORICAL ANALYSIS")
print("=" * 80)

categorical_cols = ['ACCOUNTNAME', 'NAME']
for col in categorical_cols:
    if col in df.columns:
        print(f"\n{col} Distribution (Top 10):")
        print(df[col].value_counts().head(10))

# ============================================================================
# 10. VISUALIZATIONS
# ============================================================================
print("\n" + "=" * 80)
print("10. GENERATING VISUALIZATIONS")
print("=" * 80)

# Create visualizations directory
import os
os.makedirs('eda_visualizations', exist_ok=True)

# 10.1 Correlation Heatmap
print("\nGenerating correlation heatmap...")
plt.figure(figsize=(16, 12))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Heatmap - Main & Secondary Attributes', fontsize=16, pad=20)
plt.tight_layout()
plt.savefig('eda_visualizations/correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: eda_visualizations/correlation_heatmap.png")

# 10.2 Distribution of Main Attributes
print("\nGenerating distribution plots for main attributes...")
fig, axes = plt.subplots(3, 3, figsize=(18, 15))
axes = axes.ravel()

for idx, attr in enumerate(main_attributes[:9]):
    if attr in df.columns:
        data = df[attr].dropna()
        if len(data) > 0:
            axes[idx].hist(data, bins=50, edgecolor='black', alpha=0.7)
            axes[idx].set_title(f'Distribution of {attr}', fontsize=10)
            axes[idx].set_xlabel(attr)
            axes[idx].set_ylabel('Frequency')
            axes[idx].grid(True, alpha=0.3)

# Hide unused subplots
for idx in range(len(main_attributes[:9]), 9):
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig('eda_visualizations/main_attributes_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: eda_visualizations/main_attributes_distribution.png")

# 10.3 Box plots for main attributes (to show outliers)
print("\nGenerating box plots for main attributes...")
fig, axes = plt.subplots(3, 3, figsize=(18, 15))
axes = axes.ravel()

for idx, attr in enumerate(main_attributes[:9]):
    if attr in df.columns:
        data = df[attr].dropna()
        if len(data) > 0:
            axes[idx].boxplot(data, vert=True)
            axes[idx].set_title(f'Box Plot: {attr}', fontsize=10)
            axes[idx].set_ylabel(attr)
            axes[idx].grid(True, alpha=0.3)

# Hide unused subplots
for idx in range(len(main_attributes[:9]), 9):
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig('eda_visualizations/main_attributes_boxplots.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: eda_visualizations/main_attributes_boxplots.png")

# 10.4 Scatter plots for key relationships
print("\nGenerating scatter plots for key relationships...")
key_pairs = [
    ('INSTAGRAM_FOLLOWER_COUNT', 'INSTAGRAM_ENGAGEMENT_RATE'),
    ('INSTAGRAM_FOLLOWER_COUNT', 'TOTAL_SALES'),
    ('INSTAGRAM_ENGAGEMENT_RATE', 'TOTAL_SALES'),
    ('INSTAGRAM_TOTAL_ENGAGEMENT', 'TOTAL_ORDERS'),
]

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.ravel()

for idx, (x_col, y_col) in enumerate(key_pairs):
    if x_col in df.columns and y_col in df.columns:
        data = df[[x_col, y_col]].dropna()
        if len(data) > 0:
            axes[idx].scatter(data[x_col], data[y_col], alpha=0.5, s=20)
            axes[idx].set_xlabel(x_col)
            axes[idx].set_ylabel(y_col)
            axes[idx].set_title(f'{y_col} vs {x_col}', fontsize=12)
            axes[idx].grid(True, alpha=0.3)
            
            # Add correlation coefficient
            corr = data[x_col].corr(data[y_col])
            axes[idx].text(0.05, 0.95, f'r = {corr:.3f}', transform=axes[idx].transAxes,
                          verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('eda_visualizations/key_relationships_scatter.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: eda_visualizations/key_relationships_scatter.png")

# 10.5 Top performers analysis
print("\nGenerating top performers analysis...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Top 10 by follower count
if 'INSTAGRAM_FOLLOWER_COUNT' in df.columns:
    top_followers = df.nlargest(10, 'INSTAGRAM_FOLLOWER_COUNT')[['NAME', 'INSTAGRAM_FOLLOWER_COUNT']]
    axes[0, 0].barh(range(len(top_followers)), top_followers['INSTAGRAM_FOLLOWER_COUNT'])
    axes[0, 0].set_yticks(range(len(top_followers)))
    axes[0, 0].set_yticklabels(top_followers['NAME'], fontsize=8)
    axes[0, 0].set_xlabel('Follower Count')
    axes[0, 0].set_title('Top 10 by Follower Count')
    axes[0, 0].grid(True, alpha=0.3, axis='x')

# Top 10 by engagement rate
if 'INSTAGRAM_ENGAGEMENT_RATE' in df.columns:
    top_engagement = df.nlargest(10, 'INSTAGRAM_ENGAGEMENT_RATE')[['NAME', 'INSTAGRAM_ENGAGEMENT_RATE']]
    axes[0, 1].barh(range(len(top_engagement)), top_engagement['INSTAGRAM_ENGAGEMENT_RATE'])
    axes[0, 1].set_yticks(range(len(top_engagement)))
    axes[0, 1].set_yticklabels(top_engagement['NAME'], fontsize=8)
    axes[0, 1].set_xlabel('Engagement Rate (%)')
    axes[0, 1].set_title('Top 10 by Engagement Rate')
    axes[0, 1].grid(True, alpha=0.3, axis='x')

# Top 10 by total sales
if 'TOTAL_SALES' in df.columns:
    top_sales = df.nlargest(10, 'TOTAL_SALES')[['NAME', 'TOTAL_SALES']]
    axes[1, 0].barh(range(len(top_sales)), top_sales['TOTAL_SALES'])
    axes[1, 0].set_yticks(range(len(top_sales)))
    axes[1, 0].set_yticklabels(top_sales['NAME'], fontsize=8)
    axes[1, 0].set_xlabel('Total Sales ($)')
    axes[1, 0].set_title('Top 10 by Total Sales')
    axes[1, 0].grid(True, alpha=0.3, axis='x')

# Top 10 by total orders
if 'TOTAL_ORDERS' in df.columns:
    top_orders = df.nlargest(10, 'TOTAL_ORDERS')[['NAME', 'TOTAL_ORDERS']]
    axes[1, 1].barh(range(len(top_orders)), top_orders['TOTAL_ORDERS'])
    axes[1, 1].set_yticks(range(len(top_orders)))
    axes[1, 1].set_yticklabels(top_orders['NAME'], fontsize=8)
    axes[1, 1].set_xlabel('Total Orders')
    axes[1, 1].set_title('Top 10 by Total Orders')
    axes[1, 1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('eda_visualizations/top_performers.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: eda_visualizations/top_performers.png")

# ============================================================================
# 11. SUMMARY AND RECOMMENDATIONS
# ============================================================================
print("\n" + "=" * 80)
print("11. SUMMARY AND KEY INSIGHTS")
print("=" * 80)

print("\n" + "-" * 80)
print("MAIN ATTRIBUTES (Use for Primary Influencer Evaluation):")
print("-" * 80)
print("1. INSTAGRAM_FOLLOWER_COUNT - Primary reach metric")
print("2. INSTAGRAM_ENGAGEMENT_RATE - Quality indicator (higher = better)")
print("3. INSTAGRAM_TOTAL_ENGAGEMENT - Overall engagement volume")
print("4. INSTAGRAM_TOTAL_POSTS - Content activity level")
print("5. TOTAL_SALES - Business performance metric")
print("6. TOTAL_ORDERS - Conversion metric")
print("7. REFERRAL_LINK_TOTAL_CLICKS - Click-through performance")

print("\n" + "-" * 80)
print("SECONDARY ATTRIBUTES (Use for Detailed Analysis):")
print("-" * 80)
print("1. INSTAGRAM_TOTAL_LIKES - Engagement breakdown")
print("2. INSTAGRAM_TOTAL_COMMENTS - Engagement quality (comments > likes)")
print("3. INSTAGRAM_POSTS_LAST_30_DAYS - Recent activity indicator")
print("4. AVG_ORDER_SIZE - Customer value metric")
print("5. TOTAL_COMMISSION - Revenue generated")
print("6. NEW_CUSTOMERS - Customer acquisition")
print("7. COMMISSION_PERCENTAGE - Partnership terms")
print("8. CAMPAIGN_OPT_INS - Campaign participation")

print("\n" + "-" * 80)
print("KEY INSIGHTS:")
print("-" * 80)
print(f"• Dataset contains {len(df)} influencer records")
print(f"• {len(main_attributes)} main attributes identified for primary evaluation")
print(f"• {len(secondary_attributes)} secondary attributes for detailed analysis")
print(f"• Visualizations saved in 'eda_visualizations/' directory")
print(f"• Check correlation heatmap for attribute relationships")
print(f"• Review distribution plots to understand data spread")
print(f"• Examine top performers to identify best influencers")

print("\n" + "=" * 80)
print("EDA COMPLETE!")
print("=" * 80)
print("\nAll visualizations have been saved to 'eda_visualizations/' directory")
print("Review the output above for detailed statistics and insights.")

