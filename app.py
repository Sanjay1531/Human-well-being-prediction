import streamlit as st
import pandas as pd

data = {
    "Sleep": [8, 5, 6, 7, 6.5, 7.5, 4, 5.5, 8, 9, 5, 6],
    "Exercise": [60, 20, 30, 45, 25, 50, 15, 35, 70, 80, 20, 40],
    "Diet": [9, 4, 6, 8, 5, 8, 3, 7, 10, 9, 4, 6],
    "Stress": [2, 7, 5, 3, 6, 4, 8, 5, 2, 3, 7, 4],
    "Wellbeing": [1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1]
}

df = pd.DataFrame(data)


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
