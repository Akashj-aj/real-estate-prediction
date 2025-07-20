import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor

# Load dataset
df = pd.read_csv("data/cleaned_data_with_transaction_year.csv")

# Prepare features and target
X = df.drop("price_per_unit_area", axis=1)
y = df["price_per_unit_area"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define hyperparameter grid
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [10, None],
    "min_samples_split": [2, 5]
}

# Grid search to tune Random Forest
grid_search = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_grid=param_grid,
    scoring="neg_mean_squared_error",
    cv=5,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# Get best model
best_model = grid_search.best_estimator_

# Save the best model to file
joblib.dump(best_model, "models/final_model.pkl")

print("Tuned Random Forest model saved to models/final_model.pkl")
