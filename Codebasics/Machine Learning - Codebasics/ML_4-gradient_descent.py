import math

import numpy as np

# y = mx + b
# we have to find here the closest valu of m, b to  the actual value of m, b
# so that we can draw the near to perfect line

def gradient_descent(x, y):
    m_curr = b_curr = 0
    iterations = 10000
    n = len(x)
    learning_rate = 0.08 #start at 0.001
    prev_cost = 1e9

    for i in range(iterations):
        y_predicted = m_curr*x + b_curr
        cost = (1/n) * sum((y - y_predicted)**2)

        md = -(2/n) * sum(x*(y-y_predicted))
        bd = -(2/n) * sum(y-y_predicted)

        m_curr -= learning_rate*md 
        b_curr -= learning_rate*bd 

        print(f'm: {m_curr}, b: {b_curr}, cost: {cost}, iteration: {i+1}')

        is_matched = math.isclose(cost, prev_cost, rel_tol = 1e-9, abs_tol = 0.0)
        if(is_matched):
            break

        prev_cost = cost


x = np.array([1, 2, 3, 4, 5])    
y = np.array([5, 7, 9, 11, 13])    

gradient_descent(x, y)