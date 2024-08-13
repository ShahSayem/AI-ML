import numpy as np 

def gradient_descent(x, y):
    m_curr = b_curr = 0
    iterations = 1000
    n = len(x)

    for i in range(iterations):
        y_predicted = m_curr*x + b_curr
        md = -(2/n) * sum(x*(y-y_predicted))
        bd = -(2/n) * sum(y-y_predicted)

x = np.array[1, 2, 3, 4, 5]    
y = np.array[5, 7, 9, 11, 13]    

gradient_descent(x, y)