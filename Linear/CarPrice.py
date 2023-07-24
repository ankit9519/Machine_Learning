import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def load_data():
    df = pd.read_csv(
        r"D:\Machine Learning\Datasets\CarPrice_Assignment.csv", index_col="car_ID"
    )
    return df


def preprocess_data(df):
    # Drop rows with missing values in 'price'
    df = df.dropna(subset=["price"], how="any")

    return df


def train_model(X, y, random_state=5):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_state
    )
    lm = LinearRegression()
    lm.fit(x_train, y_train)
    score = lm.score(x_test, y_test)

    return lm, x_test, y_test, score


def main():
    st.title("Car Price Prediction")

    df = load_data()
    df = preprocess_data(df)

    st.write("## Exploratory Data Analysis")
    st.write(df.head())

    # Extract features and target
    X = df[
        [
            "wheelbase",
            "carlength",
            "carwidth",
            "curbweight",
            "enginesize",
            "boreratio",
            "horsepower",
            "citympg",
            "highwaympg",
        ]
    ]
    y = df["price"]

    st.write("Please use the form below to predict car prices using the model:")
    wheelbase = st.number_input("Wheelbase:")
    carlength = st.number_input("Car Length:")
    carwidth = st.number_input("Car Width:")
    curbweight = st.number_input("Curb Weight:")
    enginesize = st.number_input("Engine Size:")
    boreratio = st.number_input("Bore Ratio:")
    horsepower = st.number_input("Horsepower:")
    citympg = st.number_input("City MPG:")
    highwaympg = st.number_input("Highway MPG:")

    random_state = st.slider("Random State", min_value=0, max_value=100, value=5)

    if st.button("Predict"):
        lm, x_test, y_test, score = train_model(X, y, random_state=random_state)
        features = [
            [
                wheelbase,
                carlength,
                carwidth,
                curbweight,
                enginesize,
                boreratio,
                horsepower,
                citympg,
                highwaympg,
            ]
        ]
        predicted_price = lm.predict(features)[0]
        st.write(f"Predicted Car Price: ${predicted_price:.2f}")

        st.write("## Actual vs. Predicted Prices")
        fig, ax = plt.subplots()
        ax.scatter(y_test, lm.predict(x_test), color="blue", label="Predicted Price")
        ax.scatter(y_test, y_test, color="red", label="Actual Price")
        plt.xlabel("Actual Price")
        plt.ylabel("Predicted Price")
        plt.title("Actual vs. Predicted Prices")
        plt.legend()
        st.pyplot(fig)


if __name__ == "__main__":
    main()
