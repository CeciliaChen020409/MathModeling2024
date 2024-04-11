import numpy as np

def get_function_from_user():
    func_str = input("Enter the objective function (use 'x' as variable, e.g., 'x[0]**2 + x[1]**2'): ")

    def func(x):
        return eval(func_str)

    return func

def random_search(func, X1, num_variables, iterations, min_step, stepLength):
    best_solution = X1
    best_value = func(X1)
    i = 0
    while stepLength > min_step:
        while i < iterations:
            rVec = np.random.uniform(-1, 1, size=num_variables)
            rNorm = np.linalg.norm(rVec)
            if rNorm > 1:
                while rNorm > 1:
                    rVec = np.random.uniform(-1, 1, size=num_variables)
                    rNorm = np.linalg.norm(rVec)
            u = 1 / rNorm * rVec
            X = best_solution + stepLength * u
            newVal = func(X)
            if newVal < best_value:
                best_solution = X
                best_value = newVal
                i = 0
            else:
                i += 1
        stepLength /= 2
    return best_solution, best_value


# The interactive inputs remain the same.
# Interactive inputs
num_variables = int(input("Enter the number of variables: "))
iterations = int(input("Enter the number of iterations: "))
min_step = float(input("Enter the minimum step length: "))
stepLength = float(input("Enter the initial step length: "))
X1 = [float(x) for x in input("Enter the initial point, separated by space: ").split()]

# Get the objective function from the user
objective_function = get_function_from_user()

# Perform the random search
solution, value = random_search(objective_function, X1, num_variables, iterations, min_step, stepLength)

# Print the results
print(f"Best solution found: {solution}")
print(f"With objective function value: {value}")
