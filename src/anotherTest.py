import pandas as pd

data = pd.read_csv("../assets/House_Rent_Dataset.csv")

fields = [
    "Area Type",
    "Area Locality",
    "City",
    "Furnishing Status",
    "Tenant Preferred",
    "Point of Contact"
]

# Collect unique values
with open("unique_values.txt", "w", encoding="utf-8") as f:
    for field in fields:
        if field in data.columns:
            uniques = data[field].dropna().unique()
            f.write(f"{field}:\n")
            for val in uniques:
                f.write(f"  {val}\n")
            f.write("\n")

print("Unique values written to unique_values.txt")

