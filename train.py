import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

data = pd.read_csv("data/customer_churn_data.csv")

print("Dataset loaded successfully.")
print("Dataset shape:", data.shape)

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

for column in numeric_columns:
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

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}

results = []

for name, model in models.items():

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        prediction
    )

    precision = precision_score(
        y_test,
        prediction,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        prediction,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        prediction,
        zero_division=0
    )

    results.append([
        name,
        accuracy,
        precision,
        recall,
        f1
    ])

    print("\n" + name)
    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))

results = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]
)

print("\nModel Comparison:")
print(results)

best_model_name = results.loc[
    results["F1 Score"].idxmax(),
    "Model"
]

best_model = models[best_model_name]

print("\nBest Model:", best_model_name)

best_prediction = best_model.predict(X_test)

cm = confusion_matrix(
    y_test,
    best_prediction
)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Churn", "Churn"],
    yticklabels=["No Churn", "Churn"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Customer Churn Confusion Matrix")

plt.tight_layout()

plt.savefig(
    "outputs/confusion_matrix.png"
)

plt.close()

results.set_index("Model")[
    ["Accuracy", "Precision", "Recall", "F1 Score"]
].plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Customer Churn Model Comparison")
plt.ylabel("Score")
plt.ylim(0, 1)

plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig(
    "outputs/model_comparison.png"
)

plt.close()

print("Pictures saved in the outputs")