import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Generate dummy data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)  # y = 4 + 3x + noise

# Plotting initial data points
plt.figure(figsize=(10, 5))
plt.scatter(X, y, color="blue", label="Data points")
plt.title("Initial Dataset Visualization")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.savefig("LR_initial-data-points.png")

# Initial regression line (before training)
X_line = np.array([[0], [2]])  # Two points for the line
X_line_b = np.c_[np.ones((2, 1)), X_line]  # Add bias term (X_0 = 1)
initial_theta = np.random.randn(2, 1)  # Random initialization of theta
y_line = X_line_b @ initial_theta  # Initial predictions

plt.title("Initial Regression Line")
plt.plot(X_line, y_line, color="orange", label="Initial regression line")
plt.legend()
plt.savefig("LR_Initial-Regression-Line.png")

# Add bias term (X_0 = 1)
X_b = np.c_[np.ones((100, 1)), X]

# Hyperparameters
learning_rate = 0.1
n_iterations = 1000
m = len(X_b)

# Initialize parameters (theta)
theta = np.random.randn(2, 1)  # Random initialization

# Gradient Descent
cost_history = []

def compute_cost(X, y, theta):
    """Compute the Mean Squared Error (MSE)."""
    predictions = X @ theta
    errors = predictions - y
    return (1 / (2 * m)) * np.sum(errors ** 2)

# Training loop
for iteration in range(n_iterations):
    gradients = (1 / m) * X_b.T @ (X_b @ theta - y)
    theta -= learning_rate * gradients

    cost = compute_cost(X_b, y, theta)
    cost_history.append(cost)

    # Verbose updates
    if iteration % 100 == 0 or iteration == n_iterations - 1:
        print(
            f"Iteration {iteration}: "
            f"Cost = {cost:.4f}, "
            f"Theta0 = {theta[0][0]:.4f}, Theta1 = {theta[1][0]:.4f}"
        )

# Visualizing cost reduction over time
plt.figure(figsize=(10, 5))
plt.plot(range(n_iterations), cost_history, color="purple")
plt.title("Cost Reduction Over Iterations")
plt.xlabel("Iteration")
plt.ylabel("Cost (MSE)")
plt.savefig("LR_Cost-Reduction.png")

# Visualizing the final regression line
plt.figure(figsize=(10, 5))
plt.scatter(X, y, label="Data points")
plt.plot(X, X_b @ theta, color="red", label="Final regression line")
plt.title("Final Linear Regression Fit")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.savefig("LR_Final-Regression-Line.png")

print(f"Final Parameters: Intercept = {theta[0][0]:.4f}, Slope = {theta[1][0]:.4f}")