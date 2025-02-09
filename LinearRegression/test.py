import numpy as np

np.random.seed(0)
X_line = np.array([[0], [2]])  # Two points for the line
X_line_b = np.c_[np.ones((2, 1)), X_line]  # Add bias term (X_0 = 1)
initial_theta = np.array([[2, 1], [3, 5]])  # Random initialization of theta
y_line = X_line_b @ initial_theta


print("X line")
print(X_line)
print(X_line.shape)
print("X line bias")
print(X_line_b)
print("initial theta")
print(initial_theta)
print("y line")
print(y_line)