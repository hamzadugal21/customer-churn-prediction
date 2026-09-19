# Customer Churn Prediction

A simple machine learning project that predicts whether a customer is likely to churn.

I used customer information such as age, gender, tenure, monthly charges, contract type, internet service, total charges, and tech support.

## Models Used

* Logistic Regression
* Decision Tree
* Random Forest

The models are compared using:

* Accuracy
* Precision
* Recall
* F1 Score

## Project Structure

```text
customer-churn-prediction/
│
├── data/
│   └── customer_churn.csv
│
├── outputs/
│   ├── confusion_matrix.png
│   └── model_comparison.png
│
├── train.py
├── predict.py
├── app.py
├── requirements.txt
└── README.md
```

## Technologies

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn

## Dataset

The dataset is stored in:

```text
data/customer_churn.csv
```

It contains 1,000 customer records and 10 columns.

The target column is:

```text
Churn
```

It contains:

```text
Yes
No
```

The dataset includes:

* CustomerID
* Age
* Gender
* Tenure
* MonthlyCharges
* ContractType
* InternetService
* TotalCharges
* TechSupport
* Churn

## How to Run

First, install the required libraries:

```bash
pip install -r requirements.txt
```

### 1. Train the models

```bash
python train.py
```

This trains the models and compares their performance.

It also creates:

```text
outputs/confusion_matrix.png
outputs/model_comparison.png
```

### 2. Make a prediction

```bash
python predict.py
```

You can enter customer information and get a churn prediction.

### 3. Run the application

```bash
python app.py
```

This provides a simple way to enter customer information and get a prediction.

## GitHub

To upload the project to GitHub:

```bash
git init
git add .
git commit -m "Add customer churn prediction project"
git branch -M main
git remote add origin https://github.com/hamzadugal21/customer-churn-prediction.git
git push -u origin main
```

Replace `hamzadugal21` with your GitHub username.

## Author

