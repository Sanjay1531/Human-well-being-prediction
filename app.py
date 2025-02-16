import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ------------------------------
# 1. Load Specific Real-World Data
# ------------------------------
data = {
    "Sleep": [8, 5, 6, 7, 6.5, 7.5, 4, 5.5, 8, 9, 5, 6],
    "Exercise": [60, 20, 30, 45, 25, 50, 15, 35, 70, 80, 20, 40],
    "Diet": [9, 4, 6, 8, 5, 8, 3, 7, 10, 9, 4, 6],
    "Stress": [2, 7, 5, 3, 6, 4, 8, 5, 2, 3, 7, 4],
    "Wellbeing": [1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1]
}

df = pd.DataFrame(data)

# ------------------------------
# 2. Data Preprocessing
# ------------------------------
X = df[['Sleep', 'Exercise', 'Diet', 'Stress']]
y = df['Wellbeing']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------
# 3. Train Logistic Regression Model
# ------------------------------
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Model Evaluation
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)


st.title("💚 Human Wellbeing Prediction (Green AI)")

# Show dataset
st.write("### 📊 Sample Data")
st.dataframe(df)

# Show correlation heatmap
st.write("### 🔥 Correlation Heatmap")
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
st.pyplot(fig)

# User Input
st.write("### 🧑‍⚕️ Enter Your Health Data")
sleep = st.slider("Sleep (hours)", min_value=3, max_value=10, value=7)
exercise = st.slider("Exercise (mins/day)", min_value=0, max_value=120, value=30)
diet = st.slider("Diet Score (1-10)", min_value=1, max_value=10, value=7)
stress = st.slider("Stress Level (1-10)", min_value=1, max_value=10, value=4)

# Predict Wellbeing
input_data = np.array([[sleep, exercise, diet, stress]])
input_scaled = scaler.transform(input_data)
prediction = model.predict(input_scaled)[0]

st.write("### 🏥 Wellbeing Prediction Result")
if prediction == 1:
    st.success("✅ You have good wellbeing! Maintain your healthy habits. 💪")
else:
    st.warning("⚠️ Your wellbeing is at risk! Try improving sleep, exercise, or managing stress.")

# Show accuracy
st.write(f"*Model Accuracy:* {accuracy:.2f}")


# Show Visualization
st.write("### 📊 Wellbeing Distribution")
fig, ax = plt.subplots(figsize=(5, 4))
sns.countplot(x='Wellbeing', data=df, palette="coolwarm", ax=ax)
st.pyplot(fig)
