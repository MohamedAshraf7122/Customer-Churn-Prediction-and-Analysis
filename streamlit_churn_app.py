import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import os

# --- Configuration & Constants ---
st.set_page_config(layout="wide", page_title="Customer Churn Predictor", initial_sidebar_state="expanded")

# Set Dark Theme & Custom CSS (Enhanced for contrast and aesthetics)
st.markdown(
    """
    <style>
    /* General App Styling - Dark Theme Enhancements */
    .st-emotion-cache-18ni7ap { /* Main App Padding */
        padding-top: 1rem;
    }
    .st-emotion-cache-18ni7ap { /* Sidebar Padding */
        padding-left: 0.5rem;
        padding-right: 0.5rem;
    }
    
    /* Custom Header/Title */
    .main-header {
        color: #B6F092; /* Bright green/yellow for contrast */
        text-shadow: 2px 2px 5px rgba(0,0,0,0.5);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        padding-bottom: 10px;
    }
    
    /* Metrics Styling (KPI Cards) */
    .st-emotion-cache-16ya58p { /* Metric Value */
        font-size: 2.5rem;
        color: #FF5E8E; /* Pink/Red for Churn metrics */
        font-weight: bold;
    }
    .st-emotion-cache-10trblm { /* Metric Label */
        font-size: 1rem;
        color: #90d7f8; /* Light Blue for labels */
        font-weight: 500;
    }
    
    /* Sidebar Header Styling */
    .st-emotion-cache-k7vsyk { 
        font-size: 1.5rem;
        color: #FFB648; /* Orange for visibility */
        font-weight: bold;
    }
    
    /* Streamlit Button Styling */
    .st-emotion-cache-6cwpf4 a {
        background-color: #00cc96;
        color: white;
    }
    
    /* Input/Result Background (for consistency) */
    .st-emotion-cache-12fm52b {
        background-color: #1a2333; 
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Constants & Mappings (Crucial for your model's preprocessing) ---
MODEL_PATH = 'xgboost_model.pkl'
DATA_PATH = 'E Commerce Dataset only.xlsx'
TARGET_COL = 'Churn'

# Global mappings for Encoding (Extracted from Main 83.ipynb)
# Note: Using explicit dicts here since the original notebook used LabelEncoder but saved the mappings.
MAPPING_OF_PAYMENT = {'CC': 0, 'COD': 1, 'Cash on Delivery': 2, 'Credit Card': 3, 'Debit Card': 4, 'E wallet': 5, 'UPI': 6}
MAPPING_OF_ORDER_CAT = {'Fashion': 0, 'Grocery': 1, 'Laptop & Accessory': 2, 'Mobile': 3, 'Others': 4, 'Phone': 5}
MAPPING_OF_MS = {'Divorced': 0, 'Married': 1, 'Single': 2}


# --- 1. Data Preprocessing Functions ---
@st.cache_data
def load_and_preprocess_train_data(data_file):
    """Loads, cleans, transforms, and splits the original training data."""
    if not os.path.exists(data_file):
        # We allow running if the model file already exists, but training requires the file.
        return None, None, None, None, None

    try:
        ecomerce_df = pd.read_excel(data_file)
    except Exception as e:
        st.error(f"Error loading training data: {e}")
        return None, None, None, None, None

    # --- Data Cleaning (Imputation) ---
    ecomerce_df['Tenure'].fillna(ecomerce_df['Tenure'].median(), inplace=True)
    ecomerce_df['WarehouseToHome'].fillna(ecomerce_df['WarehouseToHome'].median(), inplace=True)
    ecomerce_df['HourSpendOnApp'].fillna(ecomerce_df['HourSpendOnApp'].mean(), inplace=True)
    ecomerce_df['OrderAmountHikeFromlastYear'].fillna(ecomerce_df['OrderAmountHikeFromlastYear'].mean(), inplace=True)
    ecomerce_df['CouponUsed'].fillna(ecomerce_df['CouponUsed'].median(), inplace=True)
    ecomerce_df['OrderCount'].fillna(ecomerce_df['OrderCount'].mean(), inplace=True)
    ecomerce_df['DaySinceLastOrder'].fillna(ecomerce_df['DaySinceLastOrder'].median(), inplace=True)
    ecomerce_df["OrderCount"] = ecomerce_df["OrderCount"].round()

    # --- Feature Engineering & Encoding (Matching notebook steps) ---
    ecomerce_df["PreferredLoginDevice"] = ecomerce_df["PreferredLoginDevice"].astype(str).str.replace("Mobile Phone", "Phone")
    ecomerce_df["PreferedOrderCat"] = ecomerce_df["PreferedOrderCat"].astype(str).str.replace("Mobile Phone", "Phone")
    
    ecomerce_df["PreferredPaymentMode"] = ecomerce_df["PreferredPaymentMode"].map(MAPPING_OF_PAYMENT)
    ecomerce_df["PreferedOrderCat"] = ecomerce_df["PreferedOrderCat"].map(MAPPING_OF_ORDER_CAT)
    ecomerce_df["MaritalStatus"] = ecomerce_df["MaritalStatus"].map(MAPPING_OF_MS)
    ecomerce_df["Gender"] = ecomerce_df["Gender"].map({'Male': 1, 'Female': 0})
    ecomerce_df["PreferredLoginDevice"] = ecomerce_df["PreferredLoginDevice"].map({'Phone': 1, 'Computer': 0})
    
    # Fill NaNs created by mapping for robustness (using mode/0 as general fallback)
    ecomerce_df["PreferredPaymentMode"].fillna(ecomerce_df["PreferredPaymentMode"].mode()[0], inplace=True)
    ecomerce_df["PreferedOrderCat"].fillna(ecomerce_df["PreferedOrderCat"].mode()[0], inplace=True)
    ecomerce_df["MaritalStatus"].fillna(ecomerce_df["MaritalStatus"].mode()[0], inplace=True)
    ecomerce_df["Gender"].fillna(ecomerce_df["Gender"].mode()[0], inplace=True)
    ecomerce_df["PreferredLoginDevice"].fillna(ecomerce_df["PreferredLoginDevice"].mode()[0], inplace=True)
    
    ecomerce_df["PreferredPaymentMode"] = ecomerce_df["PreferredPaymentMode"].astype(int)
    ecomerce_df["PreferedOrderCat"] = ecomerce_df["PreferedOrderCat"].astype(int)
    ecomerce_df["MaritalStatus"] = ecomerce_df["MaritalStatus"].astype(int)
    ecomerce_df["Gender"] = ecomerce_df["Gender"].astype(int)
    ecomerce_df["PreferredLoginDevice"] = ecomerce_df["PreferredLoginDevice"].astype(int)

    # Log Transformation
    logscale_feature = ["WarehouseToHome", "NumberOfAddress", "CouponUsed", "OrderCount", "DaySinceLastOrder", "CashbackAmount"]
    for i in logscale_feature:
        ecomerce_df[i] = np.log1p(ecomerce_df[i])

    # Drop redundant/unused columns
    ecomerce_df.drop(columns=['CustomerID'], inplace=True, errors='ignore') # CustomerID dropped in notebook final model

    # Splitting data
    X = ecomerce_df.drop(TARGET_COL, axis=1)
    y = ecomerce_df[TARGET_COL]
    
    feature_columns = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    return X_train, X_test, y_train, y_test, feature_columns

# --- 2. Model Training/Loading ---
def load_or_train_model(X_train, y_train):
    """Loads a pre-trained model or trains a new XGBoost model."""
    if X_train is None and not os.path.exists(MODEL_PATH):
        return None
        
    try:
        model = joblib.load(MODEL_PATH)
        st.sidebar.success("Model loaded successfully from xgboost_model.pkl")
        
    except FileNotFoundError:
        if X_train is None:
            st.sidebar.error("Cannot train: Training data not available.")
            return None

        st.sidebar.warning("xgboost_model.pkl not found. Training a new model...")
        
        # XGBoost Classifier Pipeline (best performing model in your notebook was XGBoost)
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('model', XGBClassifier(objective="binary:logistic", use_label_encoder=False, eval_metric='logloss'))
        ])
        
        try:
            model.fit(X_train, y_train)
            joblib.dump(model, MODEL_PATH)
            st.sidebar.success(f"Model trained and saved as {MODEL_PATH}")
            
            y_proba = model.predict_proba(X_train)[:, 1]
            st.sidebar.info(f"Training ROC-AUC: {roc_auc_score(y_train, y_proba):.4f}")
            
        except Exception as e:
            st.sidebar.error(f"Error during model training: {e}")
            return None
            
    return model

# --- 3. Prediction Logic (Pre-processing for new data) ---
def preprocess_and_predict_batch(df_raw, model, feature_columns):
    """Preprocesses a raw DataFrame and returns predictions."""
    df = df_raw.copy()
    
    df.drop(columns=['CustomerID'], inplace=True, errors='ignore')

    # Apply all preprocessing steps found in the notebook:
    # Handling NAs - Using the median/mean of the batch data as a placeholder if NAs exist
    df['Tenure'].fillna(df['Tenure'].median(), inplace=True)
    df['WarehouseToHome'].fillna(df['WarehouseToHome'].median(), inplace=True)
    df['HourSpendOnApp'].fillna(df['HourSpendOnApp'].mean(), inplace=True)
    df['OrderAmountHikeFromlastYear'].fillna(df['OrderAmountHikeFromlastYear'].mean(), inplace=True)
    df['CouponUsed'].fillna(df['CouponUsed'].median(), inplace=True)
    df['OrderCount'].fillna(df['OrderCount'].mean(), inplace=True)
    df['DaySinceLastOrder'].fillna(df['DaySinceLastOrder'].median(), inplace=True)
    df["OrderCount"] = df["OrderCount"].round()

    # Feature Engineering & Encoding (using the global mappings)
    df["PreferredLoginDevice"] = df["PreferredLoginDevice"].astype(str).str.replace("Mobile Phone", "Phone")
    df["PreferedOrderCat"] = df["PreferedOrderCat"].astype(str).str.replace("Mobile Phone", "Phone")
    
    df["PreferredPaymentMode"] = df["PreferredPaymentMode"].map(MAPPING_OF_PAYMENT)
    df["PreferedOrderCat"] = df["PreferedOrderCat"].map(MAPPING_OF_ORDER_CAT)
    df["MaritalStatus"] = df["MaritalStatus"].map(MAPPING_OF_MS)
    df["Gender"] = df["Gender"].map({'Male': 1, 'Female': 0})
    df["PreferredLoginDevice"] = df["PreferredLoginDevice"].map({'Phone': 1, 'Computer': 0})

    # Fill NaNs created by mapping for robustness (if input values don't match keys)
    df["PreferredPaymentMode"].fillna(df["PreferredPaymentMode"].mode()[0], inplace=True)
    df["PreferedOrderCat"].fillna(df["PreferedOrderCat"].mode()[0], inplace=True)
    df["MaritalStatus"].fillna(df["MaritalStatus"].mode()[0], inplace=True)
    df["Gender"].fillna(df["Gender"].mode()[0], inplace=True)
    df["PreferredLoginDevice"].fillna(df["PreferredLoginDevice"].mode()[0], inplace=True)
    
    df["PreferredPaymentMode"] = df["PreferredPaymentMode"].astype(int)
    df["PreferedOrderCat"] = df["PreferedOrderCat"].astype(int)
    df["MaritalStatus"] = df["MaritalStatus"].astype(int)
    df["Gender"] = df["Gender"].astype(int)
    df["PreferredLoginDevice"] = df["PreferredLoginDevice"].astype(int)

    # Log Transformation
    logscale_feature = ["WarehouseToHome", "NumberOfAddress", "CouponUsed", "OrderCount", "DaySinceLastOrder", "CashbackAmount"]
    for i in logscale_feature:
        df[i] = df[i].apply(lambda x: np.log1p(x) if x >= 0 else np.log1p(0)) 

    # Align columns with the model's expected features
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_columns]

    # Predict
    probabilities = model.predict_proba(df)[:, 1]
    predictions = model.predict(df)
    
    df_raw['churn_probability'] = probabilities
    df_raw['churn_prediction'] = predictions.astype(int)
    
    return df_raw

# --- 4. Streamlit App Structure ---

st.markdown('<h1 class="main-header">☁️ Customer Churn Prediction Dashboard (Dark Theme)</h1>', unsafe_allow_html=True)
st.markdown("---")

# Load/Train Model in Sidebar
st.sidebar.markdown('<h3 class="st-emotion-cache-k7vsyk">⚙️ Model Status & Tools</h3>', unsafe_allow_html=True)

# Add a prominent image/logo to the sidebar
st.sidebar.image("https://images.unsplash.com/photo-1542470761239-05244199f1fa?q=80&w=2835&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D", caption="Data Science Model", use_column_width=True)
st.sidebar.markdown("---")

# Model Loading Logic
try:
    X_train, X_test, y_train, y_test, feature_columns = load_and_preprocess_train_data(DATA_PATH)
    model = load_or_train_model(X_train, y_train)

except Exception as e:
    # If loading failed, model is None.
    model = None
    feature_columns = None


# --- Batch Prediction Section ---
st.header("📂 Batch Prediction (Upload CSV)")
st.markdown("Upload a CSV or Excel file containing new customer data for mass prediction.")

uploaded_file = st.file_uploader("Choose a CSV or Excel file (Columns must match original data)", type=['csv', 'xlsx'])

if model is None:
    st.warning("Cannot run predictions. Model failed to load or train. Ensure the training file is present or upload one.")
elif uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            data_to_predict = pd.read_csv(uploaded_file)
        else:
            data_to_predict = pd.read_excel(uploaded_file)
            
        st.info(f"File '{uploaded_file.name}' loaded successfully. Rows: {len(data_to_predict)}")
        
        with st.spinner('Running XGBoost Predictions...'):
            results = preprocess_and_predict_batch(data_to_predict.copy(), model, feature_columns)
        
        st.success("Prediction Complete!")
        
        # --- Display Metrics and Results ---
        st.markdown("### 📊 Batch Performance Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Customers Analyzed", len(results))
        
        churn_count = int(results['churn_prediction'].sum())
        col2.metric("Predicted Churns (1)", churn_count, delta_color="inverse")

        churn_rate = results['churn_prediction'].mean()
        col3.metric("Churn Rate", f"{churn_rate*100:.2f}%", delta_color="inverse")

        st.markdown("### Top 10 High-Risk Predictions")
        st.dataframe(results.sort_values(by='churn_probability', ascending=False).head(10).style.highlight_max(axis=0, subset=['churn_probability']))

        st.markdown("### Predicted Churn Distribution")
        
        fig = px.pie(
            results.groupby('churn_prediction').size().reset_index(name='Count'),
            names='churn_prediction',
            values='Count',
            title='Predicted Churn Distribution (0=Stay, 1=Churn)',
            color_discrete_sequence=['#90d7f8', '#FF5E8E']
        )
        st.plotly_chart(fig, use_container_width=True)

        csv_data = results.to_csv(index=False).encode('utf-8')
        st.download_button(
            "⬇️ Download Results CSV",
            data=csv_data,
            file_name='churn_predictions_results.csv',
            mime='text/csv'
        )

    except Exception as e:
        st.error(f"An error occurred during file processing or prediction: {e}")

st.markdown("---")

# ----------------------------------------------------
# --- Single Customer Prediction (Interactive Form - IN SIDEBAR) ---
# ----------------------------------------------------

if model is not None:
    st.header("🎯 Single Customer Prediction Result")
    st.info("Use the interactive form in the sidebar (left) to enter customer data and get predictions.")
    
    # --- Define the Form ---
    with st.sidebar.form(key='single_prediction_form'):
        st.markdown("<h5>Enter Customer Details:</h5>", unsafe_allow_html=True)
        
        # Segment 1: Key Performance Indicators (Continuous/Numeric)
        col_s1, col_s2 = st.columns(2)
        
        tenure = col_s1.slider("Tenure (Months)", min_value=0.0, max_value=61.0, value=1.0) # Default to low tenure for high churn demo
        cashback_amount = col_s2.number_input("Cashback Amount ($)", min_value=0.0, max_value=350.0, value=50.0) # Default to low cashback
        
        # Segment 2: Categorical Features
        gender = st.radio("Gender", options=["Male", "Female"], index=0, horizontal=True) # Default Male
        complain = st.radio("Has Complaint?", options=["No (0)", "Yes (1)"], index=1, horizontal=True) # Default Yes (1)
        login_device = st.radio("Preferred Login Device", options=["Phone", "Computer"], index=0, horizontal=True) 
        
        payment_mode = st.selectbox("Preferred Payment Mode", options=list(MAPPING_OF_PAYMENT.keys()), index=0) # Default CC
        order_cat = st.selectbox("Preferred Order Category", options=list(MAPPING_OF_ORDER_CAT.keys()), index=3) # Default Mobile
        marital_status = st.selectbox("Marital Status", options=list(MAPPING_OF_MS.keys()), index=2) # Default Single

        # Segment 3: Other Numeric Features (Condensed)
        st.markdown("---")
        st.markdown("<h6>Other Details:</h6>", unsafe_allow_html=True)
        
        city_tier = st.slider("City Tier (1-3)", min_value=1, max_value=3, value=1)
        warehouse_to_home = st.number_input("Warehouse To Home (Distance)", min_value=5.0, max_value=150.0, value=30.0)
        hour_spend = st.number_input("Hours Spent on App", min_value=0.0, max_value=5.0, value=1.0)
        devices = st.slider("Devices Registered (1-6)", min_value=1, max_value=6, value=2)
        satisfaction = st.slider("Satisfaction Score (1-5)", min_value=1, max_value=5, value=1) # Default Low Satisfaction
        num_address = st.number_input("Number of Addresses", min_value=1, max_value=25, value=1)
        hike_amount = st.number_input("Order Hike from Last Year (%)", min_value=11.0, max_value=30.0, value=11.0)
        coupon_used = st.number_input("Coupons Used", min_value=0.0, max_value=16.0, value=0.0)
        order_count = st.number_input("Total Order Count", min_value=1.0, max_value=20.0, value=1.0)
        days_since = st.number_input("Days Since Last Order", min_value=0.0, max_value=50.0, value=15.0) # Default High Days Since

        submitted = st.form_submit_button("🔮 Predict CHURN")

    # --- Prediction Execution & Results (Displayed on Main Page) ---
    if submitted:
        # Construct the raw data dict based on form inputs
        raw_data = {
            "CustomerID": 0,
            "Tenure": tenure,
            "PreferredLoginDevice": login_device,
            "CityTier": city_tier,
            "WarehouseToHome": warehouse_to_home,
            "PreferredPaymentMode": payment_mode,
            "Gender": gender,
            "HourSpendOnApp": hour_spend,
            "NumberOfDeviceRegistered": devices,
            "PreferedOrderCat": order_cat,
            "SatisfactionScore": satisfaction,
            "MaritalStatus": marital_status,
            "Complain": 1 if "Yes" in complain else 0,
            "NumberOfAddress": num_address,
            "OrderAmountHikeFromlastYear": hike_amount,
            "CouponUsed": coupon_used,
            "OrderCount": order_count,
            "DaySinceLastOrder": days_since,
            "CashbackAmount": cashback_amount
        }
        
        df_raw = pd.DataFrame([raw_data])
        
        with st.spinner('Calculating Churn Risk...'):
            single_result = preprocess_and_predict_batch(df_raw.copy(), model, feature_columns)
            
        probability = single_result['churn_probability'].iloc[0]
        prediction = single_result['churn_prediction'].iloc[0]
        
        # Display Results on the Main Page
        st.markdown("<h2 style='color: #FFB648;'>🎯 Prediction for Single Customer</h2>", unsafe_allow_html=True)
        
        result_col1, result_col2 = st.columns(2)
        
        result_col1.metric("Churn Probability", f"{probability*100:.2f}%", delta_color="inverse")
        
        status = "**CHURN RISK (1)**" if prediction == 1 else "**STAY (0)**"
        color = "#FF5E8E" if prediction == 1 else "#00cc96"
        result_col2.markdown(f"<p style='font-size: 1.5rem; color: {color};'>Predicted Status: {status}</p>", unsafe_allow_html=True)

        st.markdown("### Risk Meter")
        st.progress(float(probability))

        if prediction == 1:
            st.warning("🚨 This customer profile shows a high risk of churn.")
        else:
            st.success("✅ This customer profile is stable and likely to remain active.")

else:
    st.info("Cannot enable single prediction form. Model failed to load or train. Ensure the training file is present and the model is trained.")

st.markdown("---")

# Footer / helpful notes
st.markdown("""
**Notes & Tips**

- This app trains a model ONCE and saves it as `xgboost_model.pkl`. If you want to retrain, delete that file and click Train.
- Ensure uploaded prediction CSV contains the same raw columns used during training (before one-hot encoding). The app will auto-align one-hot columns when possible.
- For production deployment on Azure App Service or Azure ML, containerize this app using Docker and expose the Streamlit port.
""")