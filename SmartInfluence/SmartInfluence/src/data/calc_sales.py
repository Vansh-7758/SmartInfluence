import pandas as pd

# Load the CSV data
df = pd.read_csv('/Users/palak/Desktop/SmartInfluence/Influencer_identification_metrics_instagram_10k_04_02.csv')

# Calculate total sales by brand (account name) for each row
brand_total_sales = df.groupby('ACCOUNTNAME')['TOTAL_SALES'].transform('sum')

# Add a column for the percentage of total brand sales
# We multiply by 100 to get a percentage
df['PERCENTAGE_OF_BRAND_SALES'] = (df['TOTAL_SALES'] / brand_total_sales) * 100

# Fill any NaN values with 0 (which happens if a brand had 0 total sales)
df['PERCENTAGE_OF_BRAND_SALES'] = df['PERCENTAGE_OF_BRAND_SALES'].fillna(0)

# Save to a new CSV file
output_path = '/Users/palak/Desktop/SmartInfluence/Influencer_metrics_with_sales_percentage.csv'
df.to_csv(output_path, index=False)
print(f"Successfully calculated percentages and saved new dataset to:\n{output_path}\n")

# Display a few examples from the top of the file
print("Example Top 10 rows:")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
print(df[['NAME', 'ACCOUNTNAME', 'TOTAL_SALES', 'PERCENTAGE_OF_BRAND_SALES']].head(10))
