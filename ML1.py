import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer  # Import SimpleImputer

# Load the train and test data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Define the selected features for training
selected_features = ['GrLivArea', 'TotalBsmtSF', 'OverallQual', 'YearBuilt']

# Separate the features (X) and target (y) for training data
X = train_data[selected_features]
y = train_data['SalePrice']

# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred = model.predict(X_val)

# Calculate Mean Squared Error and R-squared
mse = mean_squared_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

# Print the performance metrics
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R2): {r2}")

# Handle missing values in the test data using SimpleImputer
imputer = SimpleImputer(strategy="median")
X_test = test_data[selected_features]  # Extract features from the test data
X_test_imputed = imputer.fit_transform(X_test)  # Impute missing values

# Convert the imputed numpy array back to a DataFrame with feature names
X_test_imputed_df = pd.DataFrame(X_test_imputed, columns=selected_features)

# Predict on the test data
test_predictions = model.predict(X_test_imputed_df)

# Print sample test predictions
print("Sample Test Predictions:")
print(test_predictions[:10])
