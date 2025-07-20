import pandas as pd
import joblib
from math import sqrt
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor

# -------------------------------------------------
# Paths
# -------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent       # models/ -> project root
DATA_DIR = ROOT / "data"
MODEL_PATH = ROOT / "models" / "final_model.pkl"
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

# Try these dataset paths in order (first that exists is used)
candidates = [
    DATA_DIR / "cleaned_data_with_transaction_year.csv",
    DATA_DIR / "feature_engineered_data.csv",
    DATA_DIR / "cleaned_data.csv",
    DATA_DIR / "Real estate.csv",
]
DATA_PATH = next((p for p in candidates if p.exists()), None)
if DATA_PATH is None:
    raise FileNotFoundError(
        "No dataset found. Looked for:\n" + "\n".join(str(p) for p in candidates)
    )

print(f"📥 Loading data from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

# -------------------------------------------------
# Standardize column names (handles original/raw column names)
# -------------------------------------------------
df = df.rename(columns={
    "X1 transaction date": "transaction_date",
    "X2 house age": "house_age",
    "X3 distance to the nearest MRT station": "distance_to_mrt",
    "X4 number of convenience stores": "num_convenience_stores",
    "X5 latitude": "latitude",
    "X6 longitude": "longitude",
    "Y house price of unit area": "price_per_unit_area",
    "num_stores": "num_convenience_stores",  # unify alternate names
})

# Derive transaction_year if missing
if "transaction_year" not in df.columns:
    if "transaction_date" in df.columns:
        df["transaction_year"] = df["transaction_date"].astype(float).astype(int)
    else:
        # fall back to a default or raise
        raise ValueError("No transaction_year or transaction_date column available to build year.")

# -------------------------------------------------
# Target (what we predict)
# -------------------------------------------------
target_col = "price_per_unit_area" if "price_per_unit_area" in df.columns else (
    "price" if "price" in df.columns else None
)
if target_col is None:
    raise ValueError("Neither 'price_per_unit_area' nor 'price' found in dataset.")

# -------------------------------------------------
# Features: use EXACTLY the ones in the Streamlit UI
# -------------------------------------------------
feature_cols = [
    "transaction_year",
    "house_age",
    "distance_to_mrt",
    "num_convenience_stores",
    "latitude",
    "longitude",
]

missing = [c for c in feature_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required feature columns: {missing}")

X = df[feature_cols]
y = df[target_col]

# -------------------------------------------------
# Train / tune model
# -------------------------------------------------
from sklearn.metrics import mean_squared_error, r2_score  # only for reporting

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [10, None],
    "min_samples_split": [2, 5],
}

grid_search = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_grid=param_grid,
    scoring="neg_mean_squared_error",
    cv=5,
    n_jobs=-1,
)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# quick metrics
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)  # older sklearn only supports this form
rmse = sqrt(mse)
r2 = r2_score(y_test, y_pred)
print(f"✅ Model trained. RMSE={rmse:.4f}, R²={r2:.4f}")

# -------------------------------------------------
# Save bundle (model + metadata)
# -------------------------------------------------
bundle = {
    "model": best_model,
    "feature_cols": feature_cols,
    "target_col": target_col,
}
joblib.dump(bundle, MODEL_PATH)
print(f"💾 Saved tuned model bundle to {MODEL_PATH}")
