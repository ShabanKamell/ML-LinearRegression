import numpy as np
from gradient_descent_vec import gradient_descent


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
