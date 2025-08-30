# ❤️ Heart Disease Prediction App - Full Guide

## Part 1: How to Use the Heart Disease Prediction App

### Step 1: Open the App
👉 [Click here to use the app directly](https://heart-disease-prediction-with-insights.streamlit.app/)

*(Alternative: Run the Jupyter Notebook locally – see Part 2 below)*

### Step 2: Load the Dataset
- The dataset (`dataset.csv`) is loaded using **pandas**.
- Includes patient information like **age, sex, blood pressure, cholesterol, etc.**

### Step 3: Explore and Clean the Data
- Checks for **missing values, duplicates, or irrelevant columns**.
- May include **label encoding or normalization** if required.

### Step 4: Feature Selection and Preprocessing
- Splits the dataset into **features (X)** and **target (y)**.
- May perform **scaling (StandardScaler / MinMaxScaler)**.
- Optionally visualizes **correlations/distributions** using seaborn/matplotlib.

### Step 5: Split the Dataset
- Dataset is split into **training and testing sets** (e.g., 70:30 or 80:20).
- Done using `train_test_split` from sklearn.

### Step 6: Train Machine Learning Models
Trains one or more classifiers:
- Logistic Regression
- Random Forest Classifier
- K-Nearest Neighbors
- Support Vector Machine (SVM)

Each model is fitted on the training data.

### Step 7: Evaluate the Models
- Makes predictions on the test set.
- Displays performance metrics:
  - Accuracy score
  - Confusion matrix
  - Precision, Recall, F1-score
- May also include **ROC Curve** or **classification report**.

### Step 8: Make Custom Predictions
- Provides a section to input custom values (e.g., age, cholesterol).
- Model predicts whether the person has **heart disease (1)** or **not (0)**.

---

## Part 2: How to Run or Modify the App

### Step 1: Install Dependencies
Make sure the following Python libraries are installed:
pandas
numpy
matplotlib
seaborn
scikit-learn
streamlit

### Step 2: Launch Jupyter Notebook
- Open **Anaconda Prompt** or **Terminal**.
- Navigate to the project folder and run:

### Step 3: Open the Notebook File
-Click on Heart Disease Prediction.ipynb.
-Run all cells sequentially from top to bottom.

### Step 4: Modify or Extend
-Add more models or tune hyperparameters.
-Save the best model using joblib/pickle for deployment.

## Part 3: Workflow Diagram (Code Flow)
[Heart Disease Prediction.ipynb] 
      ↓ Import Libraries 
      ↓ Load Dataset (dataset.csv) 
      ↓ Clean & Preprocess Data 
      ↓ Split into Train/Test 
      ↓ Train ML Models (LogReg, RF, KNN, SVM) 
      ↓ Evaluate Models (metrics, confusion matrix) 
      ↓ Input custom patient data 
      ↓ Predict Heart Disease Risk (0 or 1)

## Part 4: Function Descriptions
-load_data() → Reads CSV file into a pandas DataFrame.
-preprocess_data(df) → Cleans, scales, or encodes data as needed.
-train_models(X_train, y_train) → Trains ML models on training data.
-evaluate_model(model, X_test, y_test) → Calculates accuracy + confusion matrix.
-predict_custom_input(model, input_values) → Returns prediction for user-defined input.

## Part 5: Key Components and Files
-File: Heart Disease Prediction.ipynb → Jupyter Notebook containing all logic.
-File: dataset.csv → Heart disease dataset (UCI Heart dataset or similar).

Libraries Used:
pandas
numpy
matplotlib
seaborn
scikit-learn
streamlit
