import pandas as pd

# -------------------- Load Data --------------------
df = pd.read_csv("Influencer_metrics_with_sales_percentage.csv")

# -------------------- Clean Brand Names --------------------
df['ACCOUNTNAME'] = df['ACCOUNTNAME'].astype(str).str.strip().str.lower()

# -------------------- 1. Total Unique Brands --------------------
total_brands = df['ACCOUNTNAME'].nunique()
print("\nTotal Unique Brands:", total_brands)

# -------------------- 2. List of Distinct Brands --------------------
unique_brands = sorted(df['ACCOUNTNAME'].unique())

print("\nDistinct Brand Names:\n")
for brand in unique_brands:
    print(brand)

# -------------------- 3. Count of Influencers per Brand --------------------
brand_counts = df['ACCOUNTNAME'].value_counts()

print("\n\nBrand-wise Influencer Count:\n")
print(brand_counts)

# -------------------- 4. Save to CSV (optional but useful) --------------------
brand_counts.to_csv("brand_counts.csv")
pd.Series(unique_brands).to_csv("unique_brands.csv", index=False)

print("\nFiles saved: brand_counts.csv, unique_brands.csv")