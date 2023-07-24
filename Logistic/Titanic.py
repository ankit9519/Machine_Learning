import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Disable the warning about Pyplot Global Use
st.set_option("deprecation.showPyplotGlobalUse", False)

st.title("Titanic Survival Prediction")

# Load the Titanic dataset
df = pd.read_csv(r"D:\Machine Learning\Datasets\titanic.txt")

# Data Preprocessing
y = df["Survived"]
x = df.drop(columns="Survived")
x.drop(columns=["PassengerId", "Name", "Ticket", "Fare", "Cabin"], inplace=True)
label = LabelEncoder()
for i in x.columns:
    x[i] = label.fit_transform(x[i])
scaler = StandardScaler()
x_new = scaler.fit_transform(x)

# Train the Logistic Regression model
x_train, x_test, y_train, y_test = train_test_split(
    x_new, y, test_size=0.20, random_state=585
)
logmodel = LogisticRegression()
logmodel.fit(x_train, y_train)

# Show count plots for other features
st.subheader("Feature Visualization")
for col in x.columns:
    if col == "Age":
        continue
    st.subheader(f"Count Plot for {col}")
    plt.figure(figsize=(8, 6))
    if x[col].dtype == "object":
        sns.countplot(x=col, data=x)
    else:
        sns.countplot(x=col, data=x, palette="viridis")
    plt.xlabel(col)
    plt.ylabel("Count")
    st.pyplot()

# Show histogram for 'Age'
st.subheader("Histogram for Age")
plt.figure(figsize=(8, 6))
sns.histplot(x=df["Age"], kde=True)
plt.xlabel("Age")
plt.ylabel("Frequency")
st.pyplot()


# Predict function
def predict_survival(pclass, gender, age, siblings, parents, embark):
    gender = 0 if gender.lower() in ["f", "female"] else 1
    embark_mapping = {"S": 2, "C": 1, "Q": 0}
    embark = embark_mapping.get(embark.upper(), -1)
    test_data = [[pclass, gender, age, siblings, parents, embark]]
    prediction = logmodel.predict(test_data)[0]
    return "alive" if prediction == 1 else "dead"


# Streamlit App for prediction
st.write("Enter the following details to predict survival:")
pclass = st.slider("PClass", 1, 3)
gender = st.selectbox("Gender", ["Female", "Male"])
age = st.slider("Age", 0, 100)
siblings = st.slider("Number of Siblings", 0, 8)
parents = st.slider("Number of Parents", 0, 9)
embark = st.selectbox("Embarked from (S, C, Q)", ["S", "C", "Q"])

if st.button("Predict"):
    result = predict_survival(pclass, gender, age, siblings, parents, embark)
    st.write(f"The person is {result}")

# Display accuracy score
st.subheader("Model Accuracy")
y_pred = logmodel.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Accuracy: {accuracy:.2f}")
