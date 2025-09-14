import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

# Load CSV
df = pd.read_csv("../assets/House_Rent_Dataset.csv")

# Make sure outputs directory exists
os.makedirs("outputs", exist_ok=True)

cities = ["Kolkata", "Mumbai", "Bangalore", "Delhi", "Chennai", "Hyderabad"]
Xvals = ["BHK", "Size"]   # list of predictors

for Xval in Xvals:
    for city in cities:
        city_df = df[df["City"] == city]

        X = city_df[[Xval]].values
        y = city_df["Rent"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred_test = model.predict(X_test)

        tolerance = 0.5  # 10% relative error allowed
        errors = np.abs(y_pred_test - y_test) / y_test
        correct_mask = errors <= tolerance
        test_prediction = ~correct_mask

        plt.figure(figsize=(7, 5))
        # plt.plot(X, model.predict(X), color="green", label="Linear fit")
        plt.scatter(X_train, y_train, color="blue", label="Train data", s=4)

        plt.scatter(
            X_test[correct_mask],
            y_pred_test[correct_mask],
            color="red",
            label="Correct prediction",
            s=7,
        )

        plt.scatter(
            X_test[test_prediction],
            y_pred_test[test_prediction],
            color="orange",
            label="Test prediction",
            s=7,
        )

        plt.title(f"Linear Regression ({Xval}) - {city}")
        plt.xlabel(Xval)
        plt.ylabel("Rent")
        plt.legend()

        # Save to outputs directory
        filename = f"outputs/{city}_{Xval}_regression.png"
        plt.savefig(filename)
        plt.close()

        print(f"Saved plot: {filename}")

