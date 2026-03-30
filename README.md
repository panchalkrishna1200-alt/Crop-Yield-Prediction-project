# 🌾 CropCast — Crop Yield Prediction

A machine learning project that predicts **crop yield (kg/hectare)** based on state,
district, season, crop type, year, and area — built with Python, Scikit-learn, XGBoost, and Streamlit.

---

## 📁 Project Structure

```
crop_yield_project/
├── app.py                   ← Streamlit dashboard (main entry point)
├── train.py                 ← CLI training script
├── requirements.txt
├── data/
│   └── crop_production.csv  ← Put your Kaggle CSV here
├── models/
│   └── best_model.pkl       ← Auto-generated after training
└── utils/
    ├── __init__.py
    ├── data_loader.py        ← Load, clean, encode data
    ├── eda.py                ← EDA charts
    └── model.py              ← Train, evaluate, save models
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download the dataset
Go to: https://www.kaggle.com/datasets/abhinand05/crop-production-in-india  
Download `crop_production.csv` and place it in the `data/` folder.

### 3. (Option A) Launch the Streamlit app
```bash
streamlit run app.py
```
Then follow the tabs:
1. **Load Data** → upload your CSV
2. **Explore (EDA)** → visualise the data
3. **Train Models** → compare all algorithms
4. **Predict Yield** → enter inputs and get prediction

### 3. (Option B) Train from command line
```bash
python train.py --data data/crop_production.csv
```
This prints a full comparison table and saves the best model to `models/best_model.pkl`.

---

## 📊 Models Compared

| Model              | Typical R² |
|--------------------|-----------|
| Linear Regression  | ~0.60     |
| Random Forest      | ~0.85–0.90|
| Gradient Boosting  | ~0.84–0.88|
| XGBoost            | ~0.87–0.92|
| LightGBM           | ~0.87–0.92|

---

## 🎯 Target Variable
**Yield** = Production (tonnes) / Area (hectares) — expressed in **kg/hectare**

---

## 📝 Key Concepts Demonstrated
- Regression on continuous target variable
- Label encoding of categorical features
- Ensemble methods (Random Forest, Gradient Boosting, XGBoost, LightGBM)
- Hyperparameter tuning with RandomizedSearchCV
- 5-fold cross-validation
- MAE, RMSE, R² evaluation metrics
- Streamlit deployment with interactive prediction

---

*For B.Tech / B.Sc (CS / Agriculture / Data Science) — ML Projects*
