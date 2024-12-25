# **Horse Race Outcome Prediction Model**


This project aims to predict the outcomes of horse races based on historical race and horse data. The model is designed to determine if a horse will ***Win***, ***Place*** or ***Lose*** using advanced machine learning techniques. Additionally, the project includes feature engineering, dataset balancing, and thorough evaluation of model performance.

**Datasets**
---

- Races Dataset (all_races.csv): Contains details about races, including racecourse, conditions, and prize money.
- Horses Dataset (all_horses.csv): Contains details about horses participating in the races, such as age, weight, jockey, and outcomes.

**Libraries Used**
---

- **Data Manipulation:** pandas, numpy
- **Visualization:** matplotlib, seaborn, plotly
- **Scaling and Encoding:** sklearn.preprocessing (StandardScaler, OrdinalEncoder, LabelEncoder)
- **Machine Learning:** scikit-learn, imbalanced-learn, lightgbm, xgboost
- **Others:** os, warnings

**Data Cleaning and Preprocessing**
---

- **Data Loading:**
Combined multiple CSV files into a single dataset for Races and Horses.
Merged datasets on rid.
- **Handling Missing Values:**
Imputed missing values based on the nature of the columns (e.g., median for numerical columns, mode for categorical columns).
- **Data Type Conversion:**
Ensured appropriate data types for each column (e.g., int, float, category).
- **Outlier Treatment:**
Identified and treated outliers in numerical columns such as winningTime, prize, and decimalPrice.
- **Feature Engineering:**
Created new columns: race_time_period, is_weekend, and others to capture temporal and categorical insights.

**Exploratory Data Analysis (EDA)**
---

- **Target Distribution:**
Analyzed class distribution of res_win and res_place to understand imbalance.
- **Feature Relationships:**
Visualized correlations using heatmaps.
Examined trends using box plots and scatter plots for numerical features.

**Scaling and Encoding:**
---

- **Encoding Categorical Features:**
  - Used methods such as One-Hot Encoding, Ordinal Encoding, or Label Encoding depending on the nature of the categorical variables.
  - For high-cardinality categorical variables, applied techniques like target encoding to reduce dimensionality.
- **Scaling Numerical Features:**
  - Applied Standard Scaling (subtracting the mean and dividing by the standard deviation) to ensure a mean of 0 and standard deviation of 1.

**Balancing the Dataset:**
---

- Used SMOTE to handle imbalanced target classes.
- Ensured balanced classes in the training dataset.

**Model Building**
---

- **Baseline Models:**
Logistic Regression, Decision Trees, Random Forest.
- **Advanced Models:**
  - Gradient Boosting Models: LightGBM.
  - Tuned hyperparameters using RandomizedSearchCV.
- **Feature Scaling:**
Standardized numerical features using StandardScaler.

**Evaluation Metrics**
---

- **Metrics Used:**
  - Precision, Recall, F1-Score, Accuracy for each class.
  - Area Under the ROC Curve (AUC-ROC).
- **Confusion Matrix:**
  - Visualized classification results for better understanding of model performance.

