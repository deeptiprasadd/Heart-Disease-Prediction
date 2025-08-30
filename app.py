import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ------------------------------
# Load Data
# ------------------------------
@st.cache_data
def load_data():
    data = pd.read_csv("dataset.csv")   # <-- make sure dataset.csv is in same folder
    return data

data = load_data()

# Rename columns to full names
feature_names = {
    "age": "Age",
    "sex": "Sex (0 = Male, 1 = Female)",
    "cp": "Chest Pain Type",
    "trestbps": "Resting Blood Pressure (mm Hg)",
    "chol": "Serum Cholesterol (mg/dl)",
    "fbs": "Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)",
    "restecg": "Resting Electrocardiographic Results",
    "thalach": "Maximum Heart Rate Achieved",
    "exang": "Exercise Induced Angina (1 = Yes, 0 = No)",
    "oldpeak": "ST Depression Induced by Exercise",
    "slope": "Slope of Peak Exercise ST Segment",
    "ca": "Number of Major Vessels (0-3) Colored by Fluoroscopy",
    "thal": "Thalassemia (0 = Normal; 1 = Fixed Defect; 2 = Reversible Defect)"
}

data = data.rename(columns=feature_names)

# ------------------------------
# Model Training
# ------------------------------
X = data.drop("target", axis=1)
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
acc = accuracy_score(y_test, model.predict(X_test))

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("❤️ Heart Disease Prediction Dashboard")
st.sidebar.title("Navigation")
choice = st.sidebar.radio("Go to", ["Prediction", "Dataset & Insights"])

# ------------------------------
# Prediction Page
# ------------------------------
if choice == "Prediction":
    st.header("Enter Patient Details")

    user_input = {}

    # Age
    user_input["Age"] = st.slider("Age", 15, 75, 40)

    # Sex
    sex_choice = st.radio("Sex", ["Male", "Female"])
    user_input["Sex (0 = Male, 1 = Female)"] = 0 if sex_choice == "Male" else 1

    # Other features
    for col in X.columns:
        if col not in ["Age", "Sex (0 = Male, 1 = Female)"]:
            user_input[col] = st.number_input(
                f"{col}",
                float(data[col].min()),
                float(data[col].max()),
                float(data[col].mean())
            )

    input_df = pd.DataFrame([user_input])

    if st.button("Predict"):
        prediction = model.predict(input_df)[0]
        if prediction == 1:
            st.error("🚨 High chance of Heart Disease!")
        else:
            st.success("✅ Low chance of Heart Disease.")

    st.info(f"Model Accuracy: {acc*100:.2f}%")

# ------------------------------
# Dataset & Visualization Page
# ------------------------------
elif choice == "Dataset & Insights":
    st.subheader("Dataset Overview")
    st.write(data.head())

    st.subheader("Target Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x="target", data=data, ax=ax)
    st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.subheader("Feature Importance")
    importances = model.feature_importances_
    feature_imp = pd.DataFrame({"feature": X.columns, "importance": importances})
    feature_imp = feature_imp.sort_values(by="importance", ascending=False)

    fig, ax = plt.subplots()
    sns.barplot(x="importance", y="feature", data=feature_imp, ax=ax)
    st.pyplot(fig)
