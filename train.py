from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
import joblib
# Load the iris dataset
iris_df = load_iris()
X = iris_df.data
y = iris_df.target

# Perform train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# Create a directory for data if it doesn't exist
# data_dir = "./data"
# model_dir = './model'
# os.mkdirs(data_dir, exist_ok = True)
# os.mkdirs(model_dir, exist_ok = True)

# Create and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, "logistic_model.joblib")
print("Training complete")