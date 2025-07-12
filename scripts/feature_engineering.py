import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load the cleaned dataset
df = pd.read_csv("data/cleaned_data.csv")

# Rename columns to match your dataset's format
df.rename(columns={
    'X3 distance to the nearest MRT station': 'distance_to_mrt',
    'Y house price of unit area': 'price_per_unit_area'
}, inplace=True)

# Create derived feature
df['price_per_mrt_distance'] = df['price_per_unit_area'] / (df['distance_to_mrt'] + 1)

# Show first 5 rows to verify
print(df[['price_per_unit_area', 'distance_to_mrt', 'price_per_mrt_distance']].head())

# Logging info
logging.info("✅ Created derived feature: price_per_mrt_distance")

# Optional: Save the modified DataFrame (not required, but helpful)
df.to_csv("data/feature_engineered_data.csv", index=False)
