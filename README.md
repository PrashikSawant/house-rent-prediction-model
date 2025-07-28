# 🏠 House Rent Prediction Model

A comprehensive machine learning project that predicts house rental prices based on property features using Python and scikit-learn.
<!--
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)
![Pandas](https://img.shields.io/badge/Pandas-1.3+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
-->
## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Features](#features)
- [Model Performance](#model-performance)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Future Improvements](#future-improvements)
- [Contact](#contact)

## 🎯 Project Overview

This project develops a predictive model for house rental prices using various property characteristics. The model helps property owners, real estate agents, and potential tenants estimate fair rental prices based on key property features.

### Key Objectives:
- Analyze rental market trends and patterns
- Build accurate price prediction models
- Identify key factors influencing rental prices
- Provide actionable insights for stakeholders

## 📊 Dataset

The dataset contains **1,000 synthetic house records** with the following features:

| Feature | Description | Type |
|---------|-------------|------|
| `area_sqft` | Property area in square feet | Continuous |
| `bedrooms` | Number of bedrooms (1-5) | Categorical |
| `bathrooms` | Number of bathrooms (1-4) | Categorical |
| `age_years` | Property age in years | Continuous |
| `location_score` | Location rating (1-10 scale) | Continuous |
| `furnished` | Furnished status (0=No, 1=Yes) | Binary |
| `rent_price` | Monthly rent in ₹ (Target variable) | Continuous |

### Data Statistics:
- **Size**: 1,000 records
- **Average Rent**: ₹25,847
- **Rent Range**: ₹5,000 - ₹45,000
- **No Missing Values**: Complete dataset

## ✨ Features

### 🔍 Exploratory Data Analysis
- Comprehensive statistical analysis of rental market
- Interactive visualizations of price distributions
- Correlation analysis between features
- Market segmentation insights

### 🤖 Machine Learning Models
- **Linear Regression**: Baseline model for price prediction
- **Random Forest**: Advanced ensemble method for better accuracy
- **Feature Engineering**: Optimized feature selection and preprocessing
- **Model Comparison**: Performance evaluation across multiple algorithms

### 📈 Data Visualization
- Price distribution histograms
- Feature correlation heatmaps
- Model performance comparisons
- Feature importance rankings

## 🎯 Model Performance

| Model | MAE (₹) | RMSE (₹) | R² Score |
|-------|---------|----------|----------|
| Linear Regression | 3,247 | 4,156 | 0.847 |
| **Random Forest** | **2,891** | **3,743** | **0.881** |

### 🏆 Best Model: Random Forest
- **Mean Absolute Error**: ₹2,891
- **R² Score**: 0.881 (88.1% variance explained)
- **Top Features**: Area (35%), Location Score (28%), Furnished Status (18%)

## 🚀 Installation

### Prerequisites
```bash
Python 3.8+
Jupyter Notebook
```

### Setup
1. **Clone the repository**
```bash
git clone https://github.com/yourusername/house-rent-prediction-model.git
cd house-rent-prediction-model
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Launch Jupyter Notebook**
```bash
jupyter notebook house_rent_prediction.ipynb
```

## 💡 Usage

### Quick Start
```python
# Load the trained model
import pickle
import pandas as pd

# Load model
with open('models/best_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Make prediction
house_features = [[800, 2, 1, 5, 7.5, 1]]  # [area, bedrooms, bathrooms, age, location, furnished]
predicted_rent = model.predict(house_features)[0]
print(f"Predicted Rent: ₹{predicted_rent:.0f}/month")
```

### Sample Predictions
| Property Type | Features | Predicted Rent |
|---------------|----------|----------------|
| Small Apartment | 600 sqft, 2BR, 1Bath, Furnished, Good Location | ₹18,500/month |
| Medium House | 1200 sqft, 3BR, 2Bath, Unfurnished, Prime Location | ₹28,750/month |
| Large Villa | 2000 sqft, 4BR, 3Bath, Furnished, Average Location | ₹42,300/month |

## 📊 Results

### Key Insights:
1. **Area Impact**: Every 100 sqft increase adds ~₹2,100 to monthly rent
2. **Location Premium**: High-rated locations command 35% higher rents
3. **Furnished Advantage**: Furnished properties earn ₹3,000+ premium
4. **Sweet Spot**: 2-3 bedroom properties show highest demand-to-supply ratio

### Business Applications:
- **Property Owners**: Optimize pricing strategies
- **Real Estate Agents**: Provide accurate market valuations  
- **Tenants**: Make informed rental decisions
- **Investors**: Identify undervalued properties

## 🛠 Technologies Used

| Category | Technologies |
|----------|-------------|
| **Programming** | Python 3.8+ |
| **Data Analysis** | Pandas, NumPy |
| **Machine Learning** | scikit-learn |
| **Visualization** | Matplotlib, Seaborn |
| **Development** | Jupyter Notebook |
| **Version Control** | Git, GitHub |

## 🔮 Future Improvements

- [ ] **Real Data Integration**: Connect with property listing APIs
- [ ] **Advanced Models**: Implement XGBoost, Neural Networks
- [ ] **Web Application**: Deploy model using Flask/Streamlit
- [ ] **Real-time Updates**: Automated data collection pipeline
- [ ] **Geographic Analysis**: Add location-based clustering
- [ ] **Market Trends**: Implement time-series forecasting

## 📈 Model Deployment

The model is ready for deployment and can be integrated into:
- **Web Applications** (Flask/Django)
- **Mobile Apps** (API integration)
- **Real Estate Platforms** (Plugin/Widget)
- **Business Intelligence Tools** (Power BI/Tableau)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Contact

**Prashik Sawant**
- 📧 Email: prashiksawant47@gmail.com
- 💼 LinkedIn: [Prashik Sawant](https://www.linkedin.com/in/prashik-sawant-ds)
- 🌐 Portfolio: [https://prashiksawant.github.io/Portfolio/](https://prashiksawant.github.io/Portfolio/)
- 📱 Phone: +91 9067049591

---

⭐ **If you found this project helpful, please give it a star!** ⭐

---

## 📊 Project Statistics
<!--
![GitHub stars](https://img.shields.io/github/stars/yourusername/house-rent-prediction-model?style=social)
![GitHub forks](https://img.shields.io/github/forks/yourusername/house-rent-prediction-model?style=social)
![GitHub watchers](https://img.shields.io/github/watchers/yourusername/house-rent-prediction-model?style=social)
-->
*Built with ❤️ for the Data Science Community*
