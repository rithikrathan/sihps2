# rent_predictor.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# 1. Load dataset
data = pd.read_csv("../assets/House_Rent_Dataset.csv")

# 2. Drop useless columns
data = data.drop(columns=["Point of Contact"])

# 3. Handle Floor column (split into two numbers)
def process_floor(val):
    try:
        cur, total = val.split(" out of ")
        cur = 0 if cur.strip().lower() == "ground" else int(cur)
        total = int(total)
        return cur, total
    except:
        return 0, 0

data[["current_floor", "total_floors"]] = data["Floor"].apply(
        lambda x: pd.Series(process_floor(str(x)))
        )
data = data.drop(columns=["Floor"])

# 4. Encode categorical columns
encoders = {}
for col in ["Area Type", "Area Locality", "City", "Furnishing Status", "Tenant Preferred"]:
    enc = LabelEncoder()
    data[col] = enc.fit_transform(data[col])
    encoders[col] = enc

# 5. Define features and target
X = data.drop("Rent", axis=1)
y = data["Rent"]

# 6. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
        )

# 7. Train model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# 8. Prediction function with user input
def predict_rent():
    print("\n--- Rent Prediction ---")
    bhk = int(input("Enter number of BHK: "))
    size = int(input("Enter size in sqft: "))
    floor = input("Enter floor (e.g., 'Ground out of 3', '2 out of 5'): ")
    cur_floor, total_floor = process_floor(floor)
    area_type = input("Enter Area Type (Super Area/Carpet Area/Build Area): ")
    area_locality = input("Enter Locality: ")
    city = input("Enter City: ")
    furnishing = input("Enter Furnishing Status (Furnished/Semi-Furnished/Unfurnished): ")
    tenant = input("Enter Tenant Preferred (e.g., Family/Bachelors/Company): ")
    bathroom = int(input("Enter number of Bathrooms: "))

    # encode categorical values
    area_type = encoders["Area Type"].transform([area_type])[0]
    area_locality = encoders["Area Locality"].transform([area_locality])[0]
    city = encoders["City"].transform([city])[0]
    furnishing = encoders["Furnishing Status"].transform([furnishing])[0]
    tenant = encoders["Tenant Preferred"].transform([tenant])[0]

    # prepare input row
    row = [[
        bhk, size, area_type, area_locality, city,
        furnishing, tenant, bathroom, cur_floor, total_floor
        ]]

    return model.predict(row)[0]

# Run prediction loop
if __name__ == "__main__":
    while True:
        prediction = predict_rent()
        print(f"\nPredicted Rent: ₹{int(prediction)}\n")
        cont = input("Do you want to predict again? (y/n): ")
        if cont.lower() != "y":
            break

