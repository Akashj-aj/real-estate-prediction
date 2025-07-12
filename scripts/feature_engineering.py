import pandas as pd
import logging
import os

logging.basicConfig(level=logging.INFO)

# Load the cleaned data
df = pd.read_csv("data/cleaned_data.csv")

# Rename for simplicity
df.rename(columns={
    'X3 distance to the nearest MRT station': 'distance_to_mrt',
    'Y house price of unit area': 'price_per_unit_area'
}, inplace=True)

# Create the new feature
df['price_per_mrt_distance'] = df['price_per_unit_area'] / (df['distance_to_mrt'] + 1)

# Show results
print(df[['price_per_unit_area', 'distance_to_mrt', 'price_per_mrt_distance']].head())

# Logging
logging.info("✅ Created derived feature: price_per_mrt_distance")

# Optional: Save new file
os.makedirs("data", exist_ok=True)
df.to_csv("data/feature_engineered_data.csv", index=False)
