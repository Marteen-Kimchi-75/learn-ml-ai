# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("agg")

# Importing the dataset
iris_dataset = pd.read_csv("LogisiticRegression\iris.csv")
print("Sample of the dataset:")
print(iris_dataset.head())

# Mapping species to numerical values for classification
iris_dataset['species'] = iris_dataset['species'].astype('category').cat.codes
print("Species mapped to numerical labels:")
print(iris_dataset.head())

# Splitting data into features (X) and target (y)
X = iris_dataset.iloc[:, :-1].values
y = iris_dataset.iloc[:, -1].values
num_classes = len(np.unique(y))
num_features = X.shape[1]

# One-hot encoding the target labels
y_one_hot = np.eye(num_classes)[y]

# Adding bias term to features
X_bias = np.c_[np.ones(X.shape[0]), X]

# Parameters initialization
theta = np.zeros((num_features + 1, num_classes))

# Hyperparameters
learning_rate = 0.1
num_iterations = 5000

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Numerical stability
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def compute_cost(X, y_true, theta):
    m = X.shape[0]
    logits = np.dot(X, theta)
    probabilities = softmax(logits)
    cost = -np.mean(np.sum(y_true * np.log(probabilities + 1e-9), axis=1))
    return cost

def gradient_descent(X, y_true, theta, learning_rate, num_iterations):
    m = X.shape[0]
    cost_history = []

    for i in range(num_iterations):
        logits = np.dot(X, theta)
        probabilities = softmax(logits)
        error = probabilities - y_true
        gradients = np.dot(X.T, error) / m
        theta -= learning_rate * gradients

        # Compute and store cost for visualization
        cost = compute_cost(X, y_true, theta)
        cost_history.append(cost)

        if i % 500 == 0:
            print(f"Iteration {i}: Cost = {cost}")

    return theta, cost_history

# Training the model
print("\nStarting Gradient Descent...")
theta_final, cost_history = gradient_descent(X_bias, y_one_hot, theta, learning_rate, num_iterations)
print("Training complete.")
print(f"Final cost: {cost_history[-1]}")
print(f"Learned parameters (theta): \n{theta_final}")

# Visualizing cost over iterations
plt.figure(figsize=(8, 6))
plt.plot(range(num_iterations), cost_history, color='purple')
plt.title("Cost Over Iterations")
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.savefig("LoR_Cost-over-iterations.png")

# Model evaluation on new custom input
def predict(X, theta):
    logits = np.dot(X, theta)
    probabilities = softmax(logits)
    return np.argmax(probabilities, axis=1)

# Utility to test custom data points
def test_custom_input():
    print("\nTest the Model with Custom Inputs:")
    try:
        sepal_length = float(input("Enter sepal length (cm): "))
        sepal_width = float(input("Enter sepal width (cm): "))
        petal_length = float(input("Enter petal length (cm): "))
        petal_width = float(input("Enter petal width (cm): "))
        
        custom_data = np.array([1, sepal_length, sepal_width, petal_length, petal_width])  # Bias term included
        prediction = predict(custom_data.reshape(1, -1), theta_final)
        species_map = {0: "setosa", 1: "versicolor", 2: "virginica"}
        
        print(f"Prediction: {species_map[prediction[0]]}")
    except ValueError:
        print("Invalid input! Please enter numeric values.")

# Call the test function
test_custom_input()