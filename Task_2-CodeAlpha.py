import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# Load Dataset
data = pd.read_csv("C:/Users/HP/OneDrive/Desktop/Python/Task2_CodeAlpha/car data.csv")
print("First 5 rows:\n", data.head())

# Clean column names: lowercase + strip spaces
data.columns = data.columns.str.strip().str.lower()
print("\nActual column names:\n", data.columns.tolist())

# Data Preprocessing
print("\nChecking for nulls:\n", data.isnull().sum())
data.dropna(inplace=True)

# Convert year to car age
if 'year' in data.columns:
    data['year'] = 2025 - data['year']
else:
    print("Column 'year' not found!")

# Identify categorical columns and encode them
categorical_cols = ['fuel', 'seller_type', 'transmission', 'owner']
available_categoricals = [col for col in categorical_cols if col in data.columns]
data = pd.get_dummies(data, columns=available_categoricals, drop_first=True)

# Identify & drop all non-numeric columns (like 'name')
non_numeric_cols = data.select_dtypes(include='object').columns.tolist()
print("\nDropping non-numeric columns (if any):", non_numeric_cols)
X = data.drop(columns=non_numeric_cols + ['selling_price']) if 'selling_price' in data.columns else data
y = data['selling_price'] if 'selling_price' in data.columns else None

# Check if target exists
if y is None:
    raise ValueError("'selling_price' column not found. Cannot train the model.")

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)

print("\nR2 Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))

# Visualization
plt.figure(figsize=(8,5))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Prices")
plt.grid(True)
plt.show()
