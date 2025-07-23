# 🏠 Real Estate Price Predictor

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5.0-orange.svg)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An advanced machine learning web application for predicting real estate prices using property characteristics. Built with Streamlit and powered by optimized Random Forest algorithms, featuring a modern dark-themed UI with dynamic animations.

![Real Estate Predictor Demo](assets/demo-screenshot.png)

## ✨ Features

- **🤖 Advanced ML Prediction**: Optimized Random Forest model with feature importance analysis
- **🎨 Modern UI**: Clean dark theme with gradient animations and responsive design
- **📊 Interactive Visualizations**: Dynamic feature importance charts with matplotlib integration
- **⚡ Real-time Predictions**: Instant price predictions with input validation
- **📱 Responsive Design**: Mobile-friendly interface with modern CSS animations
- **🔧 Comprehensive Analysis**: Detailed EDA notebooks and model comparison studies

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Akashj-aj/real-estate-prediction.git
   cd real-estate-prediction
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app/real_estate_app.py
   ```

4. **Open your browser** and navigate to `http://localhost:8501`

## 📊 Dataset

The project uses Taiwan real estate transaction data with the following features:

- **Transaction Year**: Year of property transaction (2012-2013)
- **House Age**: Age of the property in years
- **Distance to MRT**: Distance to nearest MRT station in meters
- **Number of Convenience Stores**: Count of nearby convenience stores
- **Latitude & Longitude**: Geographic coordinates
- **Price**: Target variable - property price per unit area (NT$/ping)

### Data Files
- `data/Real estate.csv` - Original dataset
- `data/cleaned_data.csv` - Preprocessed data
- `data/cleaned_data_with_transaction_year.csv` - Final processed data

## 🧠 Machine Learning Pipeline

### Model Development Process

1. **Exploratory Data Analysis (EDA)**
   - Basic statistical analysis and data distribution
   - Null value handling and outlier detection
   - Feature correlation analysis
   - Geographic and temporal trend analysis

2. **Feature Engineering**
   - Data cleaning and preprocessing
   - Categorical encoding
   - Feature scaling and normalization
   - Feature importance analysis

3. **Model Training & Evaluation**
   - Multiple algorithm comparison (Linear Regression, Decision Tree, Random Forest, XGBoost)
   - Hyperparameter tuning with GridSearchCV
   - Cross-validation and performance metrics
   - Model selection based on R² score and RMSE

### Models Implemented

| Model | R² Score | RMSE | Status |
|-------|----------|------|--------|
| Linear Regression | 0.73 | 8.2 | ✅ Baseline |
| Decision Tree | 0.81 | 6.9 | ✅ Implemented |
| **Random Forest** | **0.87** | **5.4** | ⭐ **Selected** |
| XGBoost | 0.84 | 6.1 | ✅ Alternative |

## 📁 Project Structure

```
real-estate-prediction/
├── app/
│   ├── real_estate_app.py      # Main Streamlit application
│   └── style.css               # Modern UI styling with animations
├── data/
│   ├── Real estate.csv         # Original dataset
│   ├── cleaned_data.csv        # Preprocessed data
│   └── cleaned_data_with_transaction_year.csv
├── models/
│   ├── final_model.pkl         # Trained Random Forest model
│   └── scaler.pkl             # Feature scaler object
├── notebooks/
│   ├── week1_basic_eda.ipynb           # Initial data exploration
│   ├── week2_null_handling.ipynb       # Data cleaning
│   ├── week2_outliers_duplicates.ipynb # Outlier analysis
│   ├── week3_*.ipynb                   # Feature analysis
│   ├── week4_*.ipynb                   # Model development
│   └── data_analysis_by_graph.ipynb    # Comprehensive analysis
├── scripts/
│   ├── clean_columns.py        # Data preprocessing utilities
│   ├── encode_categorical.py   # Encoding functions
│   ├── feature_engineering.py  # Feature creation
│   ├── save_model.py          # Model serialization
│   └── test_model_predictions.py # Model testing
├── requirements.txt           # Python dependencies
├── LICENSE                   # Project license
└── README.md                # Project documentation
```

## 🛠️ Technical Stack

### Core Technologies
- **Framework**: Streamlit 1.37.0
- **ML Libraries**: scikit-learn 1.5.0, XGBoost 2.0.3
- **Data Processing**: pandas 2.2.2, numpy 1.26.4
- **Visualization**: matplotlib 3.9.0, seaborn 0.13.2
- **Model Persistence**: joblib 1.4.2

### UI/UX Features
- **Responsive Design**: Mobile-first responsive layout
- **Modern Animations**: CSS keyframe animations and transitions
- **Dynamic Themes**: Medium dark theme with gradient effects
- **Font System**: Montserrat (headings) + Source Sans Pro (body)
- **Interactive Elements**: Hover effects and smooth transitions

## 🎯 Usage Guide

### Input Parameters

1. **Transaction Year**: Select between 2012 or 2013
2. **House Age**: Use slider to set property age (0-50 years)
3. **Distance to MRT**: Enter distance in meters
4. **Convenience Stores**: Set count using slider (0-20)
5. **Coordinates**: Input latitude and longitude values

### Output

- **Price Prediction**: Estimated property value in NT$/ping
- **Feature Importance**: Interactive chart showing which features most influence the prediction
- **Confidence Metrics**: Model performance indicators

## 📈 Model Performance

### Key Metrics
- **Accuracy (R² Score)**: 87%
- **Mean Absolute Error**: 4.2 NT$/ping
- **Root Mean Square Error**: 5.4 NT$/ping
- **Training Time**: < 2 seconds
- **Prediction Time**: < 0.1 seconds

### Feature Importance Ranking
1. **Latitude** (0.28) - Geographic location
2. **Longitude** (0.24) - Geographic location  
3. **Distance to MRT** (0.21) - Transportation access
4. **House Age** (0.15) - Property condition
5. **Convenience Stores** (0.08) - Local amenities
6. **Transaction Year** (0.04) - Market timing

## 🔬 Development Notebooks

The `notebooks/` directory contains detailed analysis:

- **Week 1**: Basic EDA and data understanding
- **Week 2**: Data cleaning, null handling, outlier detection
- **Week 3**: Feature analysis, price distributions, correlations
- **Week 4**: Model development, comparison, and final selection

## 🚀 Deployment

### Local Development
```bash
streamlit run app/real_estate_app.py --server.port 8501
```

### Production Deployment
The application can be deployed on:
- **Streamlit Cloud**: Direct GitHub integration
- **Heroku**: With `Procfile` configuration
- **Docker**: Containerized deployment
- **AWS/GCP**: Cloud platform deployment

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add docstrings to functions
- Include unit tests for new features
- Update documentation for API changes

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Taiwan real estate market data providers
- Streamlit community for excellent documentation
- scikit-learn team for robust ML algorithms
- Open source contributors and maintainers

## 📞 Contact

**Akash J**
- GitHub: [@Akashj-aj](https://github.com/Akashj-aj)
- Project Link: [https://github.com/Akashj-aj/real-estate-prediction](https://github.com/Akashj-aj/real-estate-prediction)

---

⭐ **Star this repository if you found it helpful!**

---

*Built with ❤️ using Streamlit and scikit-learn*

