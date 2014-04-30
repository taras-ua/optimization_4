import numpy as np
import math


# Target matrix
def matrix_a():
    return np.matrix("2.0 2.0; 2.0 2000.0")


# Target vector
def vector_b():
    return np.array([2, 5996])


# Vector scalar
def scalar(x, y):
    return x[0] * y[0] + x[1] * y[1]


# Matrix dot vector
def dot(m, v):
    return np.array([m[0, 0] * v[0] + m[0, 1] * v[1], m[1, 0] * v[0] + m[1, 1] * v[1]])


# General square function
def sq_func(matrix, vector, x):
    return scalar(dot(matrix, x), x) / 2 + scalar(vector, x)       # 0.5 <Ax,x> + bx


# Target function
def f(x):
    return sq_func(matrix_a(), vector_b(), x)


# Derivative from target function by parameter number "param" (0 for x, 1 for y)
def derivative(param, x, eps=0.00001):
    eps_vector = np.array([0.0, 0.0])
    eps_vector[param] = eps
    return (f(x + eps_vector) - f(x - eps_vector)) / (2*eps)


# Gradient from target function
def grad(x):
    return np.array([derivative(0, x), derivative(1, x)])


# Step golden grind
def golden(fun, t, p, eps=0.00001):
    right = 2
    left = -2
    golden_section = (1 + 5 ** 0.5) / 2
    while abs(right - left) > eps:
        x1, x2 = right - (right - left) / golden_section, left + (right - left) / golden_section
        arg1 = t + x1 * p
        arg2 = t + x2 * p
        y1, y2 = fun(arg1), fun(arg2)
        if y1 >= y2:
            left = x1
        else:
            right = x2
    return (x1 + x2) / 2


# Beta parameter from method
def beta(matrix, x, h_prev):                                                                 # <f'(xk), A*h_prev> /
    return scalar(grad(x), dot(matrix, h_prev)) / scalar(h_prev, dot(matrix, h_prev))        # <h_prev, A*h_prev>


# Direction from method
def direction(matrix, x, h_prev):
    return -1 * grad(x) + beta(matrix, x, h_prev) * h_prev                    # h = -f'(xk) + beta_prev * h_prev


# Dual direction method
def dual_direction(x0, eps=0.00001):
    h = -1 * grad(x0)   # h0 = -f'(x0)
    x = x0
    iteration = 0
    while True:
        print ""
        print "*** ITERATION #", iteration, " ***"
        print "x = ", x
        print "f(x) = ", f(x)
        print "h = ", h
        print "||grad|| = ", np.linalg.norm(grad(x))
        iteration += 1
        x += golden(f, x, h) * h
        h = direction(matrix_a(), x, h)
        if np.linalg.norm(grad(x)) < eps:
            break
    print ""
    print "*** R E S U L T ***"
    print "x = ", x
    print "f(x) = ", f(x)
    print "||grad|| = ", np.linalg.norm(grad(x))


# M A I N
def main():
    x0 = np.array([1.0, 1.0])
    dual_direction(x0)


main()