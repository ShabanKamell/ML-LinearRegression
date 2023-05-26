import numpy as np


# Define the cost function
def f(X, t, weights):
    # Compute the predictions for the given weights and input X
    y_pred = np.dot(X, weights)
    # Compute the squared error between the predictions and the target values
    error = y_pred - t
    cost = np.sum(error ** 2) / (2 * len(t))  # Mean squared error
    return cost


# Define the derivative of the cost function with respect to the weights
def f_derivative(X, t, weights):
    error = np.dot(X, weights) - t
    derivative = np.dot(X.T, error) / len(t)  # Derivative of MSE w.r.t. weights
    return derivative


# Generate some data and test the functions
if __name__ == '__main__':
    # Input features (e.g. price)
    X = np.array([0, 0.2, 0.4, 0.8, 1.0])
    # Target values (e.g. quantity sold)
    t = 5 + X  # Output linear, no noise
    X = X.reshape((-1, 1))  # Reshape X to be a column vector
    X = np.hstack([np.ones((X.shape[0], 1)), X])  # Add a column of ones for the bias term
    print(X.shape)  # Expect output: (5, 2) for a linear model y = mx + c

    # Starting parameters
    weights = np.array([1.0, 1.0])

    # Compute the cost with the starting parameters
    print(f(X, t, weights))  # Expect output: 8.0

    # Compute the derivative of the cost with respect to the weights
    print(f_derivative(X, t, weights))  # Expect output: [-4.  -1.92]

