# 🏡 Egypt House Price Prediction

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20App-success?logo=streamlit)](https://share.streamlit.io/YOUR_STREAMLIT_APP_LINK_HERE)
[![MLflow](https://img.shields.io/badge/MLflow-enabled-blue)](https://mlflow.org/)

An interactive Streamlit application that forecasts residential property prices in Egypt using machine learning. Users can input specific property features and receive a real-time price prediction based on a trained model.

> Built with 💡 by [Islam Abd Aljawad](https://www.linkedin.com/in/islamabdaljawad/) , [Mohamed Adel](https://www.linkedin.com/in/mohamed-adel-14885b248?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app) , [Rana Ashraf](https://www.linkedin.com/in/rana-ashraf-349a52198/) , [Ahmed Abdelbaset](https://www.linkedin.com/in/ahmed-samy-abdelbaset-40060620a?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app) , [Habiba Kandil](https://www.linkedin.com/in/islamabdaljawad/)
---

## 📑 Table of Contents

- [Project Overview](#project-overview)
- [Project Goals](#project-goals)
- [Technologies Used](#technologies-used)
- [Repository Structure](#repository-structure)
- [Data Overview](#data-overview)
- [Data Preprocessing Details](#data-preprocessing-details)
- [Model Training](#model-training)
- [Streamlit Web Application](#streamlit-web-application)
- [Dashboard Visualizations](#dashboard-visualizations)
- [How to Run the Project Locally](#how-to-run-the-project-locally)
- [Deployment Recommendations](#deployment-recommendations)
- [Future Enhancements](#future-enhancements)
- [Contributors](#contributors)
- [License](#license)
- [Contact](#contact)

---

## Project Overview

The **Egypt House Price Prediction** project provides a machine learning-based solution to predict residential real estate prices in Egypt. It is designed to help buyers, sellers, real estate agents, and financial institutions assess property values based on historical data trends.

The system is fully automated — starting from raw data preprocessing, model training and evaluation, experiment tracking, to an easy-to-use **Streamlit** web application for end-users.

---

## Project Goals

- Develop a **robust prediction model** for estimating house prices based on multiple features.
- Build a **modularized and scalable codebase** for future extension (e.g., adding more features, bigger datasets).
- Create a **user-friendly interface** for non-technical users using **Streamlit**.
- Enable **experiment management and model tracking** through **MLflow**.

---

## Technologies Used

| Category            | Tools                          |
|---------------------|--------------------------------|
| Programming Language| Python 3.9+                    |
| Web App             | Streamlit                      |
| Machine Learning    | Scikit-learn, XGBoost          |
| Data Processing     | Pandas, NumPy                  |
| Visualization       | Matplotlib, Plotly             |
| Model Tracking      | MLflow                         |
| Packaging/Deployment| YAML                           |

---

## Repository Structure

```
Egypt-House-Price-Prediction/
├── app.py                  # Main Streamlit application
├── app.yaml                # Streamlit app deployment configuration
├── data/
│   ├── Egypt_Real_Estate.csv   # Raw dataset
│   └── cleaned_data.csv        # Processed dataset
├── dashboard/
│   └── dashboard.py         # Streamlit dashboard to explore data insights
├── models/
│   └── best_model.pkl       # Saved trained model
├── mlruns/                  # MLflow experiments and model tracking
├── presentation/
│   └── Project_presentation.pptx   # Final project presentation materials
├── filter.py                # Code for filtering available data options
├── preprocess.py            # Scripts for data cleaning and feature engineering
├── train.py                 # Model training pipeline
├── predict.ipynb            # Jupyter notebook for making manual predictions
├── utils.py                 # Helper functions (loaders, preprocessors, etc.)
├── requirements.txt         # List of all Python dependencies
└── README.md                # Main project documentation
```

---

## Data Overview

The project uses a real estate dataset containing property listings from Egypt.

| Feature        | Description                                 |
|----------------|---------------------------------------------|
| `Location`     | City/Area where the property is located     |
| `Area`         | Size of the property in square meters       |
| `Bedrooms`     | Number of bedrooms                          |
| `Bathrooms`    | Number of bathrooms                         |
| `Price`        | Property selling price (target variable)    |
| `Type`         | Apartment/Villa/Studio (if available)       |
| `Delivery Term`| Finished/Semi-Finished                      |
| `Furnished`    | Whether the property is furnished or not    |

**Additional engineered features:**

- **Price per sqm**
- **Property age** (if available)

---

## Data Preprocessing Details

**File:** `preprocess.py`

- **Handling Missing Values:**
  - Rows missing critical information (e.g., `Location`, `Price`, `Area`) were dropped.
  - Categorical features were filled with "Unknown" where necessary.

- **Feature Encoding:**
  - `Location` and `Type` were label encoded for model compatibility.

- **Outlier Removal:**
  - Properties with an unusually low or high price per sqm were removed to prevent model distortion.

- **Normalization:**
  - Log-transformation was applied to `Price` to handle skewness.

---

## Model Training

**File:** `train.py`

**Model Candidates:**

- **Linear Regression:** As baseline model.
- **Random Forest Regressor:** Ensemble learning.
- **XGBoost Regressor:** Used with hyperparameter tuning to achieve the best results.

**Model Evaluation Metrics:**

- **R² Score** (Coefficient of Determination)
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)

**Cross-validation:**

- 5-Fold Cross-validation was used to validate model robustness.

**Experiment Tracking:**

- **MLflow** was used to track:
  - Model versions.
  - Parameters (like n_estimators, max_depth for XGBoost).
  - Metrics per experiment run.
  - Artifacts like trained model binaries.

---

## Streamlit Web Application

**File:** `app.py`

The Streamlit app allows users to interactively:

- Select **Location** from a dropdown.
- Adjust **Area**, **Number of Bedrooms**, and **Number of Bathrooms** using sliders.
- Predict property price dynamically.
- View a **data dashboard**:
  - Price distributions.
  - Area-to-price correlations.
  - City-wise average prices.

**App Pages:**

| Page     | Function                                      |
|----------|-----------------------------------------------|
| Home     | Model introduction and prediction form        |
---

## Dashboard Visualizations

**File:** `dashboard/dashboard.py`

Interactive, live graphs:

- Average property prices by location (bar chart).
- Price distribution histogram.
- Scatter plot (Area vs Price).
- Correlation heatmap of numeric features.

**Powered by Plotly** for dynamic, zoomable charts.

---

## How to Run the Project Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/islamabdaljawad/Egypt-House-Price-Prediction.git
   cd Egypt-House-Price-Prediction
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # For Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app:**
   ```bash
   streamlit run app.py
   ```

The app should open automatically in your browser: [http://localhost:8501](http://localhost:8501)

---

## Deployment Recommendations

- **Local Deployment:** Using `streamlit run app.py` locally.
- **Streamlit Cloud:** Deploy directly to [Streamlit Cloud](https://streamlit.io/cloud) (ensure `app.yaml` is configured).
- **Heroku Deployment (Optional):**
  - Create a `Procfile` and push the repo to Heroku.

---

## Future Enhancements

| Feature             | Details                                             |
|---------------------|-----------------------------------------------------|
| Real-time Listings  | Scrape new property listings to retrain models periodically |
| Deep Learning Models| Implement deep neural networks (DNN Regressors)     |
| Mobile Version      | Optimize the Streamlit app for mobile users         |
| Multiple Models     | Allow user to select between "best accuracy" or "fastest" model |
| Automatic Retraining| Setup CI/CD pipelines to update the model every X months |

---

## Contributors

| Name                                 | Role               | LinkedIn                                            |
|--------------------------------------|--------------------|-----------------------------------------------------|
| [Islam Abd Aljawad](https://www.linkedin.com/in/islam-abd-aljawad/) | Data Scientist     | [LinkedIn](https://www.linkedin.com/in/islam-abd-aljawad/) |
| [Mohamed Adel](https://www.linkedin.com/in/mohamed-adel/)             | ML Engineer        | [LinkedIn](https://www.linkedin.com/in/mohamed-adel/)      |
| [Rana Ashraf](https://www.linkedin.com/in/rana-ashraf/)               | Data Analyst       | [LinkedIn](https://www.linkedin.com/in/rana-ashraf/)        |
| [Ahmed Abdelbaset](https://www.linkedin.com/in/ahmed-abdelbaset/)     | ML Engineer        | [LinkedIn](https://www.linkedin.com/in/ahmed-abdelbaset/)   |
| [Habiba Kandil](https://www.linkedin.com/in/habiba-kandil/)           | Project Manager    | [LinkedIn](https://www.linkedin.com/in/habiba-kandil/)      |

---

## License

This project is open-source and available under the **MIT License**.  
Feel free to use, distribute, and modify it with attribution.

---

## Contact

For any questions, improvements, or business inquiries, please reach out:

- Mohamed Adel: [LinkedIn](https://www.linkedin.com/in/mohamed-adel/)
- Open an [Issue](https://github.com/islamabdaljawad/Egypt-House-Price-Prediction/issues) on GitHub.

---

Let's make house price prediction 
