# 🏠 Egypt House Price Prediction

An interactive Streamlit app that forecasts residential property prices in Egypt using machine learning. Users can input specific property features and receive a real-time price prediction based on a trained model.

> Built with 💡 by [Islam Abd Aljawad](https://www.linkedin.com/in/islamabdaljawad/) , [Mohamed Adel](https://www.linkedin.com/in/islamabdaljawad/) , [Rana Ashraf](https://www.linkedin.com/in/islamabdaljawad/),[Ahmed Abdelbaset](https://www.linkedin.com/in/islamabdaljawad/),[Habiba Kandil](https://www.linkedin.com/in/islamabdaljawad/)

---

## 📸 App Preview

![App Screenshot](https://github.com/islamabdaljawad/forecast-egypt-property-values/blob/main/Images/streamlit%20web.JPG)  

---

## 🔍 Project Overview

This project aims to help users estimate property values in Egypt using historical real estate data. The app:

- Analyzes and cleans real listing data
- Trains and compares multiple ML models
- Uses the best-performing model for predictions
- Provides a clean UI built with Streamlit

---

## 🧹 Data Cleaning

### 🧩 Problems in the Raw Data

- Inconsistent and messy **city, compound, and delivery** names  
- Missing or null values in key fields like `price`, `area`, `delivery_date`  
- Outliers in `price` (too low/high) and `area` (too small/large)  
- Duplicate or irrelevant columns  
- Text values mixed in Arabic and English, with typos

---

### ✅ Solutions Applied

- Standardized city and compound names (e.g., "6th of October" → "6 October")  
- Removed or imputed rows with missing or irrelevant values  
- Filtered extreme values using domain knowledge  
- Converted `delivery_date` to "delivery time in months"  
- Grouped low-frequency compounds under `"Unknown"`  
- Cleaned up all category label inconsistencies

---

### 🔧 Data Cleaning Process

1. **Drop irrelevant columns** (e.g., agent info, IDs, links)  
2. **Handle missing data**:
   - Removed rows with missing target (`price`)
   - Dropped features with over 30% missing values
3. **Fix categorical labels** (standard casing, language unification)  
4. **Create new features**:
   - `delivery_time_in_months` from current date and delivery date  
   - Binary `furnished` feature
5. **Outlier filtering**:
   - `price` between 100,000 and 50,000,000 EGP  
   - `area` between 20 and 1000 sqm  
6. **Exported cleaned dataset** for modeling and app integration

---

## 🤖 Model Selection & Evaluation

Several machine learning regression models were trained and tested to predict house prices.

### 📊 Models and Their Results

| Model               | R² Score | MAE (EGP)   | RMSE (EGP)   |
|--------------------|----------|-------------|--------------|
| **Linear Regression** | 0.69     | 641,000     | 1,121,000    |
| **Ridge Regression**  | 0.70     | 629,000     | 1,097,000    |
| **Decision Tree**     | 0.76     | 561,000     | 1,008,000    |
| **XGBoost**           | 0.82     | 505,000     | 910,000      |
| **Random Forest**     | **0.84** | **487,000** | **862,000**  |

---

### 🏆 Best Performing Model

- **Random Forest Regressor**  
- Tuned using **GridSearchCV**
- Achieved the best results:
  - R² Score: **0.84**
  - MAE: **487,000 EGP**
  - RMSE: **862,000 EGP**

---

## 🛠 Features of the App

- Predicts prices for multiple property types (Apartment, Villa, Duplex, etc.)  
- Supports inputs for:
  - Property type, city, area, furnishing status  
  - Bedrooms, bathrooms, delivery term and date  
  - Compound (or unknown), payment method  
- User-friendly real-time predictions in Streamlit

---

## 🗂 Project Structure

```
.
├── main.py                # Streamlit app
├── src/
│   ├── utils.py           # Prediction logic
│   ├── filter.py          # Input filters and options
│   └── model.pkl          # Trained ML model
├── notebooks/
│   ├── egypt-houses-market-analysis.ipynb
│   └── forecasting-house-prices-in-egypt.ipynb
├── requirements.txt       # Dependencies
└── README.md              # Project documentation
```

---

## 🧪 Notebooks

- `egypt-houses-market-analysis.ipynb`: Exploratory analysis, cleaning strategy  
- `forecasting-house-prices-in-egypt.ipynb`: Model training and evaluation  

---

## 🚀 How to Run the App Locally

1. **Clone the repo**  
2. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```
3. **Launch the app**  
   ```bash
   streamlit run main.py
   ```

---

## 🔮 Future Improvements

- Deploy to Streamlit Cloud  
- Add map visualizations for city/compound pricing  
- Use SHAP for explainability  
- Allow CSV upload for batch predictions

---

## 👤 Author

**Islam Abd Aljawad Ahmed**  
[LinkedIn](https://www.linkedin.com/in/islamabdaljawad) • [GitHub](https://github.com/islamabdaljawad) • [Portfolio](https://islamabdaljawad.github.io) • [Email](mailto:islam.abdaljawad01@gmail.com)

---

## 📜 License

This project is licensed under the MIT License.

---

## ⭐ Support

If you found this project helpful, feel free to give it a ⭐ on GitHub and share it with others!

---

Let me know if you'd like a badge section, a hosted version added, or an interactive table in the README!
