# **Real Estate Price Prediction**

## **Project Overview**
This project predicts real estate prices (NT$/ping) based on features such as transaction year, house age, distance to MRT stations, number of nearby convenience stores, and location coordinates (latitude and longitude).  
The project includes **data cleaning, feature engineering, model building, evaluation, and a Streamlit web app** for user-friendly price predictions.

---

## **Features**
- **Exploratory Data Analysis (EDA):** Insights into dataset distributions, correlations, and trends.
- **Data Preprocessing:** Cleaning, feature engineering (e.g., extracting transaction year), and encoding.
- **Model Building:** Training multiple machine learning models including Linear Regression, Decision Tree, Random Forest, XGBoost, and Tuned Random Forest.
- **Model Comparison:** Metrics such as RMSE and R² used to select the best-performing model.
- **Streamlit App:** Interactive web app to input property details and get predictions.
- **Logging:** Predictions made via the app are stored in `logs/predictions.csv`.

---

## **Dataset**
- **File:** `data/cleaned_data_with_transaction_year.csv`
- **Target Variable:** `price_per_unit_area`
- **Features Used for Model:**
  - `transaction_year`
  - `house_age`
  - `distance_to_mrt`
  - `num_convenience_stores`
  - `latitude`
  - `longitude`

---

## **Model Performance**
The following table summarizes the performance of models (evaluated on test data):

| **Model**            | **RMSE**  | **R²**   |
|----------------------|-----------|----------|
| Tuned Random Forest  | 5.58      | 0.81     |
| Random Forest        | 5.69      | 0.81     |
| XGBoost              | 6.23      | 0.77     |
| Linear Regression    | 7.35      | 0.68     |
| Decision Tree        | 7.73      | 0.64     |

The **Tuned Random Forest model** was chosen as the final model and saved as `models/final_model.pkl`.

---

## **How to Run Locally**

### **1. Clone Repository**
```bash
git clone https://github.com/your-username/real-estate-prediction.git
cd real-estate-prediction
```

### **2. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3. Run the App**
```bash
streamlit run app/real_estate_app.py
```

---

## **Live Demo**
The app is deployed using **Streamlit Cloud**.  
**Live App Link:** [Real Estate Price Predictor](https://real-estate-prediction-1.streamlit.app)

---

## **Future Scope**
This project can be extended with:
- Advanced ML models like LightGBM and CatBoost.
- Integration with real estate APIs for live data.
- Interactive maps and bulk predictions.
- Cloud deployment on AWS or GCP with CI/CD pipelines.

See the full list of ideas in **[future_scope.md](future_scope.md)**.

---

## **Team Members**
This project was developed collaboratively by:
- **Akash J**
- **Rajana Durga Pavan Kumar**
- **M. Surya Teja**
- **Borigi Jyothiradhithya**
- **Hema Sudabathula**
- **Anushka Parihar**
- **Yuvanesh S**
- **Reddy Bunny Sridhar Reddy**

---

## **License**
This project is open-source and available under the **MIT License**.
