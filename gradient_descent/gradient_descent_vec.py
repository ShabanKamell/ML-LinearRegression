import numpy as np

"""
We implemented the gradient descent for 2 variables (x, y)
● Do simple changes to let the code process the entire vector of variables,
rather than just two variables
○ Test it using the two functions provided
○ Compare the results with those of the old program
● Create file: gradient_descent_vec.py
● def gradient_descent(fderiv, inital_start, step_size = 0.001, precision = 0.00001, max_iter = 1000):
○ inital_start: numpy array of D values
○ fderiv: function can take a vector of D values and returns D gradients
"""
def gradient_descent(
        fderiv,
        initial_start,
        step_size=0.001,
        precision=0.00003,
        max_iter=10000):
    current_pos = initial_start
    for i in range(max_iter):
        prev_pos = current_pos
        gradient = fderiv(current_pos)
        current_pos = current_pos - step_size * gradient
        diff = np.sum(abs(current_pos - prev_pos))
        if diff < precision:
            print("Converged in", i, "iterations")
            return current_pos
    print("Did not converge in", max_iter, "iterations")
    return current_pos


def f1_deriv(x):
    return np.array([2 * x[0], 8 * x[1]])


def f2_deriv(x):
    return np.array([4 * x[0] ** 3, 10 * x[1]])


if __name__ == '__main__':
    initial_start = np.array([1.0, 1.0])

    print("Testing function f1:")
    print(gradient_descent(f1_deriv, initial_start))

    print("Testing function f2:")
    print(gradient_descent(f2_deriv, initial_start))
