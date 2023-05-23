import numpy as np
from gradient_descent_vec import gradient_descent

"""
Consider the function: f(x, y, z) = sin(x) + cos(y) + sin(z)
● Explore several initial starting points, and report a list of minimum values for
this function using gradient descent
● Import; don’t copy!
○ from gradient_descent_vec import gradient_descent
● As an example, one of the outputs should:
Initially start at [1. 2. 3.5]
ends at point: [0.22457445 2.67782963 4.21311734]
with a minimum value of -1.5496155799445872
● Recall derivative of sin(x) = cos(x), and derivative of cos(x) = -sin(x)
○ Or use an online calculator
"""

def f_deriv(x):
    return np.array([
        np.cos(x[0]),
        -np.sin(x[1]),
        np.cos(x[2])]
    )


def f(x):
    return np.sin(x[0]) + np.cos(x[1]) + np.sin(x[2])


initial_starts = [[1.0, 2.0, 3.5],
                  [0.0, 0.0, 0.0],
                  [2.0, -2.0, 2.0],
                  [-1.0, 3.0, -2.5]]

if __name__ == '__main__':
    for start in initial_starts:
        print("Initial start:", start)
        end = gradient_descent(f_deriv, np.array(start))
        print("Ends at point:", end)
        print("Minimum value:", f(end))
