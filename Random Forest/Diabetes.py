import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv(r"C:\DataSets\diabetes.csv")

# Data Preprocessing
# Check for missing values
df.isnull().sum()

# Impute missing values with mean
for i in df.columns:
    df[i] = df[i].fillna(df[i].mean())

# Remove outliers using IQR
for i in df.columns:
    Q1 = df[i].quantile(0.25)
    Q3 = df[i].quantile(0.75)
    IQR = Q3 - Q1
    df[i] = np.where((df[i] >= Q1 - 1.5 * IQR) & (df[i] <= Q3 + 1.5 * IQR), df[i], np.nan)

# Drop rows with missing values after outlier removal
df.dropna(inplace=True)

# Split the data into features (X) and target (y)
y = df['Outcome']
x = df.drop(columns='Outcome')

# Calculate VIF for feature selection
def vif_score(x):
    vif_data = pd.DataFrame()
    vif_data['Features'] = df.columns
    vif_data['Score'] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    return vif_data

for i in x.columns:
    x[i] = x[i].fillna(x[i].mean())

col_list = []  # to store all the numerical columns
for i in x.columns:
    if ((x[i].dtypes != "object") & (i != "charges")):
        col_list.append(i)

X = x[col_list]
vif_data = pd.DataFrame()
vif_data['feature'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]

# Drop columns with high VIF
x.drop(columns=['BloodPressure', 'BMI', 'Age'], inplace=True)

# Standardize the features
scaler = StandardScaler()
x_new = scaler.fit_transform(x)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_new, y, test_size=0.2, random_state=195)

# Train the RandomForestClassifier
rdf = RandomForestClassifier(random_state=42)
rdf.fit(x_train, y_train)

# Calculate accuracy score on the test set
y_pred = rdf.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)

# Create a Streamlit app
def main():
    st.title("Diabetes Prediction App using Random Forest")
    st.subheader("Enter the details for prediction:")

    # Get user input for prediction
    pregnancies = st.number_input("Pregnancies", min_value=0, step=1, value=0)
    glucose = st.number_input("Glucose Level", min_value=0, value=100)
    skin_thickness = st.number_input("Skin Thickness", min_value=0, step=1, value=20)
    insulin = st.number_input("Insulin Level", min_value=0, value=80)
    diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, value=0.5)

    # Predict button
    if st.button("Predict"):
        test_data = np.array([pregnancies, glucose, skin_thickness, insulin, diabetes_pedigree], ndmin=2)
        test_data_scaled = scaler.transform(test_data)

        # Make predictions
        prediction = rdf.predict(test_data_scaled)

        # Show the result
        st.subheader("Prediction Result:")
        if prediction[0] == 0:
            st.write("The individual is not diabetic.")
        else:
            st.write("The individual is diabetic.")

        # Display accuracy score
        st.subheader("Accuracy Score:")
        st.write(f"The accuracy of the model on the test set is: {accuracy:.2f}")

if __name__ == "__main__":
    main()
