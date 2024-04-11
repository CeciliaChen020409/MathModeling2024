import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy as sci
from scipy.optimize import minimize

def objective_function(x):
    return x[0]**6 + x[1]**4 + 5*x[2]**2  # Example function

def gradient(x):
    return np.array([6*x[0]**5, 4*x[1]**3, 10*x[2]])  # Correct gradient

def hessian(x):
    return np.array([[30*x[0]**4, 0,0], [0, 12*x[1]**2, 0],[0, 0, 10]])  # Correct Hessian


def newtons_method(func, grad, hess, initial_guess, max_iter=10):
    x = initial_guess
    points = [x]
    for _ in range(max_iter):
        H_inv = np.linalg.inv(hess(x))
        grad_val = grad(x)
        x = x - np.dot(H_inv, grad_val)
        points.append(x)
    return x, points

# Initial guess and optimization
initial_guess = np.array([1.0, 1.0, 1.0])
solution, points = newtons_method(objective_function, gradient, hessian, initial_guess)
points = np.array(points)
print("Objective function: x[0]**6 + x[1]**4 + 5*x[2]**2")
print("Iteration numbers: ", 10)
print("Initial Guess: ", initial_guess)
print("best solution is : ", solution)
print("best value is: ", objective_function(solution))

