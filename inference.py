# Necessary Imports
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("./logistic_model.joblib")

# Get user inputs
sepalLength = float(input("Enter sepal Length: "))
sepalWidth = float(input("Enter sepal Width: "))
petalLength = float(input("Enter petal Length: "))
petalWidth = float(input("Enter petal width: "))

# Convert to data point (np.ndarray)
user_input = [[sepalLength, sepalWidth, petalLength, petalWidth]]

# Make predictions on class
predictions = model.predict(user_input)
classes = ["Iris-Setosa", "Iris-Versicolor", "Iris-Virginica"]
print("Predicted class is,", classes[predictions[0]])
print("Inference complete")