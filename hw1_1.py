import numpy as np


def ex1():
    x1 = np.array([-1,-1], dtype=float)
    x2 = np.array([1,0], dtype=float)
    x3 = np.array([-1,1.5], dtype=float)
    x = np.array([x1, x2, x3])
    y = np.array([1, -1, 1])

def perceptron(x, y, T):
    theta_0 = 0
    theta = np.zeros(x.shape[1])

#    theta = np.array([-4,2.5])
#    theta_0 = -2

    print("T = {}".format(T))
    print("theta = {}; theta0 = {}".format(theta, theta_0))
#    print("x.shape[0]={}".format(x.shape[0]))
    for t in range(T):
        for i in range(x.shape[0]):
            if y[i]*(theta@x[i] + theta_0) <= 0:
                theta = theta + y[i]*x[i]
                theta_0 = theta_0 + y[i]
                print("theta = {}; theta0 = {}; iteracion: i={}, t={}".format(theta, theta_0, i,t))
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


def clasificar(theta, theta0, x):
    return np.sign(theta@x+theta0)

if __name__ == "__main__":
    """"" x1 = np.array([-1, -1], dtype=float)
    x2 = np.array([1, 0], dtype=float)
    x3 = np.array([-1, 10], dtype=float)

    y1 = 1
    y2 = -1
    y3 = 1

    x = np.array([x1, x2, x3])
    y = np.array([y1, y2, y3])
    perceptron_origin(x, y, 10)
    """
    x1 = np.array([-4,2], dtype=float)
    x2 = np.array([-2,1], dtype=float)
    x3 = np.array([-1,-1], dtype=float)
    x4 = np.array([2,2], dtype=float)
    x5 = np.array([1,-2], dtype=float)

    x = np.array([x1, x2, x3, x4, x5])

    y1 = 1
    y2 = 1
    y3 = -1
    y4 = -1
    y5 = -1

    y = np.array([y1, y2, y3, y4, y5])

    for t in range(3):
        theta, theta0 = perceptron(x, y, t)
#        for item in x:
#            print(clasificar(theta, theta0, item))

