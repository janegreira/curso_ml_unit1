import numpy as np


def ex1():
    x1 = np.array([-1,-1], dtype=float)
    x2 = np.array([1,0], dtype=float)
    x3 = np.array([-1,1.5], dtype=float)
    x = np.array([x1, x2, x3])
    y = np.array([1, -1, 1])

def perceptron(x, y, T):
    theta_0 = np.zeros(x.shape[1])
    theta = np.zeros(x.shape[1])

    for t in range(T):
        for i in range(x.shape[0]):
            if y[i]*(theta@x[i] + theta_0) <= 0:
                theta = theta + y[i]*x[i]
                theta_0 = theta_0 + y[i]

    return theta, theta_0

def perceptron_origin(x, y, T):

    theta = np.zeros(x.shape[1])
    print(theta)
    for t in range(T):
        for i in range(x.shape[0]):
            if y[i]*(theta@x[i]) <= 0:
                theta = theta + y[i]*x[i]
                print(theta)

    return theta


if __name__ == "__main__":
    x1 = np.array([-1, -1], dtype=float)
    x2 = np.array([1, 0], dtype=float)
    x3 = np.array([-1, 10], dtype=float)

    y1 = 1
    y2 = -1
    y3 = 1

    x = np.array([x1, x2, x3])
    y = np.array([y1, y2, y3])

    perceptron_origin(x, y, 10)