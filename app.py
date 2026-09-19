import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv("data/customer_churn_data.csv")

data = data.drop("CustomerID", axis=1)

numeric_columns = [
    "Age",
    "Tenure",
    "MonthlyCharges",
    "TotalCharges"
]

for column in numeric_columns:
    data[column] = pd.to_numeric(
        data[column],
        errors="coerce"
    )

    data[column] = data[column].fillna(
        data[column].median()
    )

data["Gender"] = data["Gender"].map({
    "Female": 0,
    "Male": 1
})

data["ContractType"] = data["ContractType"].map({
    "Month-to-Month": 0,
    "One-Year": 1,
    "Two-Year": 2
})

data["InternetService"] = data["InternetService"].map({
    "None": 0,
    "DSL": 1,
    "Fiber Optic": 2
})

data["TechSupport"] = data["TechSupport"].map({
    "No": 0,
    "Yes": 1
})

data["Churn"] = data["Churn"].map({
    "No": 0,
    "Yes": 1
})

data = data.dropna()

X = data.drop("Churn", axis=1)
y = data["Churn"]

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

print(" CUSTOMER CHURN PREDICTION APP")


age = int(input("\nEnter customer age: "))

print("\nSelect Gender:")
print("1. Male")
print("2. Female")

gender_choice = int(input("Enter choice: "))

if gender_choice == 1:
    gender = 1
elif gender_choice == 2:
    gender = 0
else:
    print("Invalid choice.")
    exit()

tenure = int(
    input("\nEnter customer tenure in months: ")
)

monthly_charges = float(
    input("Enter monthly charges: ")
)

print("\nSelect Contract Type:")
print("1. Month-to-Month")
print("2. One-Year")
print("3. Two-Year")

contract_choice = int(input("Enter choice: "))

if contract_choice == 1:
    contract = 0
elif contract_choice == 2:
    contract = 1
elif contract_choice == 3:
    contract = 2
else:
    print("Invalid choice.")
    exit()

print("\nSelect Internet Service:")
print("1. None")
print("2. DSL")
print("3. Fiber Optic")

internet_choice = int(input("Enter choice: "))

if internet_choice == 1:
    internet = 0
elif internet_choice == 2:
    internet = 1
elif internet_choice == 3:
    internet = 2
else:
    print("Invalid choice.")
    exit()

total_charges = tenure * monthly_charges

print("\nSelect Tech Support:")
print("1. Yes")
print("2. No")

support_choice = int(input("Enter choice: "))

if support_choice == 1:
    tech_support = 1
elif support_choice == 2:
    tech_support = 0
else:
    print("Invalid choice.")
    exit()

customer = pd.DataFrame([{
    "Age": age,
    "Gender": gender,
    "Tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "ContractType": contract,
    "InternetService": internet,
    "TotalCharges": total_charges,
    "TechSupport": tech_support
}])

prediction = model.predict(customer)[0]

print(" RESULT")

print(
    "Calculated Total Charges:",
    round(total_charges, 2)
)

if prediction == 1:
    print("Prediction: Customer is likely to CHURN.")
else:
    print("Prediction: Customer is likely to NOT CHURN.")

