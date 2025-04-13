# forecast-egypt-property-values
Thanks for sharing the project files! Based on the contents of your `main.py` and the notebooks you uploaded, here's an updated and accurate `README.md` tailored to your project:

---

```markdown
# 🏠 Egypt House Price Prediction

This project is an interactive **Streamlit** web application that predicts the price of residential properties in Egypt based on user inputs and a trained machine learning model.

Built to help users — buyers, sellers, and real estate professionals — estimate property prices quickly and reliably using real data and advanced regression techniques.

---

## 📌 Features

- 🏡 Predict prices based on features like area, bedrooms, bathrooms, location, furnishing, and more.
- 📊 User-friendly interface built with Streamlit.
- 🧠 Machine Learning model (Random Forest) trained on real Egyptian property market data.
- 🔧 Modular architecture with clean code separation.

---

## 🧪 How It Works

1. The user enters property details via a web form.
2. These details are passed to a model pipeline through the `predict_price()` function.
3. The trained model returns a price estimate.

---

## 🗂 Project Structure

```
forecast-egypt-property-values/
├── main.py                       # Main Streamlit app
├── src/
│   ├── utils.py                  # Price prediction function
│   ├── filter.py                # Dropdown values for UI (cities, property types, etc.)
│   └── model.pkl                 # Trained model (RandomForest)
├── notebooks/
│   ├── forecasting-house-prices-in-egypt.ipynb
│   └── egypt-houses-market-analysis.ipynb
├── requirements.txt
└── README.md
```

---

## 🚀 Run the App Locally

### 1. Clone the repository

```bash
git clone https://github.com/islamabdaljawad/forecast-egypt-property-values.git
cd forecast-egypt-property-values
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch the app

```bash
streamlit run main.py
```

---

## 📋 Input Features

- **Property Type**  
- **Bedrooms / Bathrooms**  
- **Area (sqm)**  
- **Furnished**  
- **Level (floor)**  
- **Compound name or Unknown**  
- **Payment Option** (Cash / Installment / Both)  
- **Delivery Date (in months)**  
- **Delivery Term** (e.g., Finished, Core & Shell)  
- **City**  

---

## 🧠 Model Info

- **Algorithm:** RandomForestRegressor
- **Tuning:** GridSearchCV (hyperparameter optimization)
- **Metrics Evaluated:** R² Score, MAE, RMSE
- **Trained on:** Real Egyptian property listings

---

## 📈 Notebooks

- `egypt-houses-market-analysis.ipynb`: EDA & market trends  
- `forecasting-house-prices-in-egypt.ipynb`: Model training and evaluation

---

## ✨ Future Work

- [ ] Add model explainability (SHAP, feature importance)
- [ ] Add map-based visualization for location insights
- [ ] Deploy to Streamlit Cloud
- [ ] Integrate XGBoost for comparison

---

## 👨‍💻 Author

**Islam Abd Aljawad Ahmed**  
[📧 Email](mailto:islam.abdaljawad01@gmail.com)  
[🔗 LinkedIn](https://www.linkedin.com/in/islamabdaljawad/)  
[💻 Portfolio](https://islamabdaljawad.github.io/)  

---

## ⭐️ Star This Repo

If you find this project useful, feel free to give it a ⭐️ and share it!

```

Let me know if you'd like to include a GIF/demo screenshot of the app or want help deploying it!
