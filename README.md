# 📈 Customer Churn Prediction and Analysis

This project is a key requirement for the **Digital Egypt Pioneers Initiative (DEPI)**, managed by the Egyptian Ministry of Communications and Information Technology in partnership with IBM. Our team, Team 80, is undertaking the "Customer Churn Prediction and Analysis" project to build a machine learning model that predicts which customers are likely to stop using a service.

The primary goal is to leverage data science techniques to identify customers at risk of leaving, enabling the company to implement proactive retention strategies. 💡

## 👥 Team Members
* Youssef Mohamed Soliman
* Mohamed Ashraf Gaber
* Amr Yousry Badr
* Ahmed Hamdy Elsayed
* Marwan Ahmed Mohamed

---

## 📊 Dataset
The project utilizes an E-commerce dataset provided in an Excel file (`E Commerce Dataset only.xlsx`). The dataset contains 5630 rows and 20 columns, with features describing customer demographics, behavior, and transaction history.

### Dataset Features:
| Variable | Description |
| :--- | :--- |
| `CustomerID` | Unique customer ID |
| `Churn` | Churn Flag (1 for churned, 0 for not churned) |
| `Tenure` | Tenure of the customer in the organization |
| `PreferredLoginDevice` | Preferred login device of the customer |
| `CityTier` | City tier of the customer |
| `WarehouseToHome` | Distance between the warehouse and the customer's home |
| `PreferredPaymentMode` | Preferred payment method of the customer |
| `Gender` | Gender of the customer |
| `HourSpendOnApp` | Number of hours spent on the mobile app or website |
| `NumberOfDeviceRegistered`| Total number of devices registered for a particular customer |
| `PreferedOrderCat` | Preferred order category in the last month |
| `SatisfactionScore` | Satisfactory score from the customer on the service |
| `MaritalStatus` | Marital status of the customer |
| `NumberOfAddress` | Total number of addresses added for a particular customer |
| `Complain` | Any complaint raised in the last month (1 for yes, 0 for no) |
| `OrderAmountHikeFromlastYear`| Percentage increase in order amount from the last year |
| `CouponUsed` | Total number of coupons used in the last month |
| `OrderCount` | Total number of orders placed in the last month |
| `DaySinceLastOrder` | Days since the last order by the customer |
| `CashbackAmount` | Average cashback in the last month |

---

## 🚀 Project Milestones & Progress

The project is divided into five milestones. Currently, **Milestones 1, 2, and 3 are complete**.

### ✔️ Milestone 1: Data Collection, Exploration, and Preprocessing
**Objective:** To collect, explore, and preprocess the customer churn data to prepare it for analysis and model building.

1.  **Data Loading 📥:** The dataset was loaded from an Excel file into a pandas DataFrame.
2.  **Initial Exploration 🔍:**
    * The dataset was examined for its structure (`.shape`), data types (`.info()`), and basic statistics (`.describe()`).
    * A check for missing data revealed a total of **1856 null values** across columns like `Tenure`, `WarehouseToHome`, and `OrderCount`.
    * No duplicate rows were found in the dataset.
3.  **Data Cleaning & Preprocessing 🧹:**
    * **Missing Values:** Null values in numerical columns were handled by imputing the **median** for skewed data (e.g., `Tenure`, `CouponUsed`) and the **mean** for normally distributed data (e.g., `HourSpendOnApp`). This ensures the dataset is complete without significantly altering the original data distribution.
    * **Data Consistency:** The `PreferredLoginDevice` column had inconsistent values ("Mobile Phone" and "Phone"), which were standardized to just "Phone".
4.  **Exploratory Data Analysis (EDA) 📊:**
    * A pie chart of the `Churn` variable showed an imbalanced dataset, with **16.8%** of customers having churned and **83.2%** remaining.
    * Count plots were used to visualize the distribution of categorical variables like `PreferredLoginDevice`, `Gender`, and `MaritalStatus`.
    * A histogram of `HourSpendOnApp` was generated to understand user engagement patterns.

### ✔️ Milestone 2: Advanced Data Analysis & Feature Engineering
**Objective:** To perform deeper data analysis and engineer new features to improve the model's predictive power.

1.  **Advanced Statistical Analysis 🧪:**
    * Statistical tests were conducted to identify significant relationships between features and the `Churn` target variable.
    * **T-tests** confirmed that features like `Tenure`, `WarehouseToHome`, and `CashbackAmount` have significantly different means for churned vs. non-churned customers (p-value < 0.05).
    * **Chi-squared tests** revealed significant associations between churn and categorical features like `Complain`, `PreferredPaymentMode`, and `MaritalStatus` (p-value < 0.05).
    * **ANOVA** was used to compare the means of `Tenure` across different `CityTier` categories.
2.  **Feature Engineering ⚙️:**
    * **Encoding:** Categorical features with no inherent order, such as `PreferredPaymentMode` and `MaritalStatus`, were converted into numerical format using **Label Encoding**. Boolean features like `Gender_boolean` were created for binary categories.
    * **Transformation:** To handle skewness, a **log transformation** was applied to numerical features like `WarehouseToHome`, `CouponUsed`, and `CashbackAmount`.
    * **Scaling:** All numerical features were scaled using `StandardScaler` to ensure they have a mean of 0 and a standard deviation of 1. This is crucial for models sensitive to the scale of input features.
3.  **Feature Selection 🎯:**
    * **Recursive Feature Elimination (RFE)** was used with both Logistic Regression and Random Forest models to rank features based on their importance for predicting churn. `Tenure`, `Complain`, `NumberOfAddress`, `CashbackAmount`, and `WarehouseToHome` consistently ranked high.
4.  **Advanced Visualization 📈:**
    * A heatmap was created to visualize the relationship between categorical variables like `Complain` and `Churn`.
    * Generated line plots to analyze the churn rate across different satisfaction scores for each product category.
    * Additional visualizations explored relationships between `Churn`, `Tenure`, `Complain`, `SatisfactionScore`, `NumberOfAddress`, and `DaySinceLastOrder`.

### ✔️ Milestone 3: Machine Learning Model Development and Optimization
**Objective:** To build, train, and optimize machine learning models to predict churn.

1.  **Data Splitting:** The dataset was split into training (80%) and testing (20%) sets, using stratification to maintain the original churn proportion in both sets.
2.  **Model Selection & Pipeline:** Three classification models were chosen:
    * Logistic Regression (with balanced class weights)
    * Random Forest Classifier (with balanced class weights and 200 estimators)
    * XGBoost Classifier
    * A `Pipeline` was created for each model, combining `StandardScaler` for preprocessing with the classifier.
3.  **Model Training & Cross-Validation:** Models were trained on the training data. A 5-fold `StratifiedKFold` cross-validation strategy was used to evaluate the generalization performance of each model based on the ROC-AUC metric.
4.  **Model Evaluation:** Models were evaluated on the held-out test set using:
    * **ROC-AUC Score:** To measure the model's ability to distinguish between churned and non-churned customers.
    * **Classification Report:** Providing precision, recall, and F1-score for both classes.
    * **Confusion Matrix:** To visualize the true positives, false positives, true negatives, and false negatives.
5.  **Model Comparison:**
    * **Logistic Regression:** Showed reasonable performance (Mean CV ROC-AUC: ~0.88, Test ROC-AUC: ~0.87), providing a good baseline.
    * **Random Forest:** Demonstrated excellent performance (Mean CV ROC-AUC: ~0.97, Test ROC-AUC: ~0.997).
    * **XGBoost:** Also achieved excellent performance (Mean CV ROC-AUC: ~0.97, Test ROC-AUC: ~0.997).
    * **Conclusion:** Based on the test set ROC-AUC and the detailed classification report (showing strong precision, recall, and F1-scores, particularly for the churn class), the **XGBoost Classifier** was selected as the best-performing model for this task.

---

## 💻 Technologies and Libraries Used
* **Python 3**
* **Pandas:** For data manipulation and analysis.
* **NumPy:** For numerical operations.
* **Matplotlib & Seaborn:** For static data visualizations.
* **Plotly:** For interactive data visualizations.
* **SciPy:** For statistical analysis (t-tests, chi-squared tests, ANOVA).
* **Scikit-learn:** For feature engineering (`LabelEncoder`, `StandardScaler`), feature selection (`RFE`), modeling (`LogisticRegression`, `RandomForestClassifier`), model evaluation (`train_test_split`, `cross_val_score`, `StratifiedKFold`, `classification_report`, `confusion_matrix`, `roc_auc_score`), and pipelines (`Pipeline`).
* **XGBoost:** For the gradient boosting classification model (`XGBClassifier`).
* **YData-Profiling:** For generating initial detailed EDA reports.

---

## ➡️ Next Steps
The project will now proceed to the subsequent milestones as outlined in the project plan.

* **Milestone 4: MLOps, Deployment, and Monitoring** ☁️: Implement MLOps practices, deploy the best-performing model (XGBoost) as an API, and set up a monitoring system.
* **Milestone 5: Final Documentation and Presentation** 🎤: Compile a comprehensive final report and present the project's findings and business impact to stakeholders.
