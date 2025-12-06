# 📈 Customer Churn Prediction & Analysis  
*A DEPI (Digital Egypt Pioneers Initiative) Project – Team 80*

This project was completed as part of the Digital Egypt Pioneers Initiative (DEPI) in collaboration with the Egyptian Ministry of Communications and Information Technology and IBM.  
Team 80 successfully developed a Customer Churn Prediction System using data science, statistical analysis, and machine learning techniques.

The goal of the project is to predict which customers are likely to churn and provide insights that help businesses improve retention and customer satisfaction.

---

## 👥 Team Members
- **Youssef Mohamed Soliman**
- **Mohamed Ashraf Gaber**
- **Amr Yousry Badr**
- **Ahmed Hamdy Elsayed**
- **Marwan Ahmed Mohamed**

---

## 📊 Dataset

The dataset used is an E-commerce customer dataset with:

- **5630 rows**
- **20 columns**

It contains customer demographics, behavioral patterns, transaction activity, and satisfaction metrics.

### Key Features

| Feature | Description |
|--------|-------------|
| CustomerID | Unique ID |
| Churn | Target variable (1/0) |
| Tenure | Customer relationship length |
| PreferredLoginDevice | Mobile, Phone, etc. |
| CityTier | Customer location tier |
| WarehouseToHome | Distance (km) |
| PreferredPaymentMode | Payment method |
| Gender | Male/Female |
| HourSpendOnApp | Time spent using app |
| NumberOfDeviceRegistered | Count of devices |
| PreferedOrderCat | Preferred order category |
| SatisfactionScore | 1–5 scale |
| MaritalStatus | Married / Single |
| NumberOfAddress | Addresses saved |
| Complain | Whether complaint was raised |
| CouponUsed | Coupons used last month |
| OrderCount | Orders placed last month |
| DaySinceLastOrder | Days since last purchase |
| CashbackAmount | Cashback received |

---

# 🚀 Project Milestones  
All milestones (1–4) are completed. Milestone 5 was removed as requested.

---

# ✔️ Milestone 1 — Data Collection, Cleaning & Exploration

### Tasks Completed
- Loaded Excel dataset using pandas.
- Performed `.info()`, `.describe()`, `.shape()`.
- Found **1856 missing values** → handled with:
  - Median imputation (Tenure, CouponUsed)
  - Mean imputation (HourSpendOnApp)
- Standardized inconsistent categories (e.g., “Mobile Phone” → “Phone”).
- Visualizations:
  - Churn distribution (16.8% churn)
  - Countplots
  - Histograms

---

# ✔️ Milestone 2 — Statistical Analysis & Feature Engineering

### Statistical Tests
- **T-test**: Tenure, CashbackAmount, WarehouseToHome → significant difference.
- **Chi-square**: Complain, Payment Mode, Marital Status → strong association.
- **ANOVA**: Tenure differs across CityTier.

### Feature Engineering
- Label Encoding
- Log transforms for skewed data:
  - WarehouseToHome, CouponUsed, CashbackAmount
- Scaling using StandardScaler
- Added binary features (e.g., Gender_boolean)

### Feature Selection
Top features selected using RFE:
- Tenure  
- Complain  
- NumberOfAddress  
- CashbackAmount  
- WarehouseToHome  

### Visualizations
- Heatmaps  
- Churn trends  
- Behavior-based insights  

---

# ✔️ Milestone 3 — Model Development & Optimization

### Models Trained
- Logistic Regression
- Random Forest (200 estimators)
- XGBoost

### Techniques Used
- Stratified 80/20 split
- Pipelines (scaling + model)
- 5-fold Stratified Cross-Validation
- Metrics:
  - ROC-AUC
  - Precision/Recall/F1
  - Confusion Matrix

### Results

| Model | CV ROC-AUC | Test ROC-AUC |
|-------|-------------|---------------|
| Logistic Regression | ~0.88 | ~0.87 |
| Random Forest | ~0.97 | ~0.997 |
| XGBoost | ~0.97 | ~0.997 |

### Best Model
🏆 **XGBoost** — highest accuracy, precision, and recall.

---

# ✔️ Milestone 4 — Deployment & MLOps

### Implemented
- Exported model as `.pkl`
- FastAPI real-time prediction endpoint
- Preprocessing pipeline for incoming requests

---

## 🧰 Technologies & Tools

| Category | Tools |
|----------|-------|
| Language | Python 3 |
| Data | pandas, numpy |
| Visualization | matplotlib, seaborn, plotly |
| Statistics | SciPy |
| Machine Learning | scikit-learn, XGBoost |
| EDA | ydata-profiling |
| Deployment | FastAPI, Uvicorn |
| Serialization | joblib |

---

# 🏁 Project Status
🎉 **Completed**  
The churn prediction model and API are fully implemented.
