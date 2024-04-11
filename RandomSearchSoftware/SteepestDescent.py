import numpy as np
def gradient_descent(f, grad_f, x0, learning_rate=0.01, tolerance=1e-6, max_iterations=100):
    """
    Performs the gradient descent optimization algorithm.

    Parameters:
    - f: The objective function to minimize.
    - grad_f: The gradient of the objective function.
    - x0: Initial guess for the parameters.
    - learning_rate: Step size for each iteration.
    - tolerance: Convergence criteria if the change in the objective function value is less than this.
    - max_iterations: Maximum number of iterations to perform.

    Returns:
    - x: The optimized parameters.
    - f(x): The minimum value of the objective function.
    - iterations: The number of iterations performed.
    """
    x = x0
    for i in range(max_iterations):
        grad = grad_f(x)
        x_new = x - learning_rate * grad
        if np.linalg.norm(f(x_new) - f(x)) < tolerance:
            break
        x = x_new
    return x, f(x), i + 1
# Example usage
def f(x):
    return x[0] ** 4 + 2 * x[1] ** 2  # Objective function
def grad_f(x):
    return np.array([4 * x[0] ** 3, 4 * x[1]])  # Gradient of the objective function


x0 = np.array([5, 5])  # Initial guess for three variables
optimized_parameters, minimum_value, iterations = gradient_descent(f, grad_f, x0)

print("Objective function: x[0]**4 + 2*x[1]**2")
print("Iteration numbers: ", iterations)
print("Initial Guess: ", x0)
print("Optimized parameters:", optimized_parameters)
print("Minimum value:", minimum_value)
