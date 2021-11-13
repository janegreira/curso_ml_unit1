import matplotlib.pyplot as plt
import numpy
import numpy as np
from sklearn.linear_model import LinearRegression, SGDClassifier


def ex1():
    x1 = np.array([-1,-1], dtype=float)
    x2 = np.array([1,0], dtype=float)
    x3 = np.array([-1,1.5], dtype=float)
    x = np.array([x1, x2, x3])
    y = np.array([1, -1, 1])

def perceptron(x, y, T=50):
    theta_0 = 0
    theta = np.zeros(x.shape[1])
    conteo = np.zeros(x.shape[0])
    for t in range(T):
        for i in range(x.shape[0]):
            if y[i]*(theta@x[i] + theta_0) <= 0:
                theta = theta + y[i]*x[i]
                theta_0 = theta_0 + y[i]
                conteo[i] = conteo[i] + 1
#    print(conteo)
    return theta, theta_0, conteo

def perceptron_origin(x, y, T):
    theta = np.zeros(x.shape[1])
    print(theta)
    for t in range(T):
        for i in range(x.shape[0]):
            if y[i]*(theta@x[i]) <= 0:
                theta = theta + y[i]*x[i]
                print(theta)

    return theta


def kernel_function(a, b):
    return a@b.T + np.square(a@b.T)

def suma_kernels(i, x, y, alfa):
    resultado = 0
    for j in range(x.shape[0]):
        # if numpy.not_equal(j,i):
        # resultado = resultado + alfa[j]*y[j]*kernel_function(x[j], x[i])
        k = x[i,0]*x[j,0]+x[i,1]*x[j,1]+2*x[i,0]*x[i,1]*x[j,0]*x[j,1]+x[i,0]^2*x[j,0]^2+x[i,1]^2*x[j,1]^2
        resultado = resultado + alfa[j]*y[j]*k
    return resultado

def closed_form(x,y):
    n = x.shape[0]
    b = np.zeros(x.shape[1] + 1)
    A = np.zeros([x.shape[1] + 1, x.shape[1] + 1])
    for t in range(n):
        xt = np.append(x[t], [1])
        b = b + y[t]*xt
        A = A + xt.reshape([3,1]) @ xt.reshape([3,1]).T
    b = (1/n) * b
    A = (1/n) * A

    theta = b @ numpy.linalg.inv(A)

    return theta



def kernel_perceptron(x, y, T=350):
    conteo = np.zeros(x.shape[0])
    alfa = np.zeros(x.shape[0])
    for t in range(T):
        for i in range(x.shape[0]):
            if y[i] * suma_kernels(i, x, y, alfa) <= 0:
                alfa[i] = alfa[i] + 1
                conteo[i] = conteo[i] + 1
    #    print(conteo)
    theta = 0
    for i in range(x.shape[0]):
        fi = np.array([x[0], x[1], x[0]^2, np.sqrt(2)*x[0]*x[1], x[1]^2])
        theta = theta + alfa[i]*y[i]*fi

    return alfa, conteo, theta


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
    x1 = np.array([0, 0, -1, 1], dtype=float)
    x2 = np.array([2, 0, -1, 9], dtype=float)
    x3 = np.array([3, 0, -1, 10], dtype=float)
    x4 = np.array([0, 2, -1, 5], dtype=float)
    x5 = np.array([2, 2, -1, 9], dtype=float)
    x6 = np.array([5, 1, 1, 11], dtype=float)
    x7 = np.array([5, 2, 1, 0], dtype=float)
    x8 = np.array([2, 4, 1, 3], dtype=float)
    x9 = np.array([4, 4, 1, 1], dtype=float)
    x10 = np.array([5, 5, 1, 1], dtype=float)

    matrix = np.array([x1,x2,x3,x4,x5,x6,x7,x8,x9,x10])

    x = matrix[:,0:2]
    y = matrix[:,2]

    theta = closed_form(x,y)
    print("Closed Form: {}".format(theta))


    errores = matrix[:,3]
    conteo = np.zeros(x.shape[0])
    '''
    while not numpy.array_equal(conteo, errores):
        matrix = numpy.random.permutation(matrix)
        x = matrix[:, 0:2]
        y = matrix[:, 2]
        errores = matrix[:, 3]
        theta, theta0, conteo = perceptron(x, y)


    print("theta = {}; theta0 = {}".format(theta, theta0))
    print(matrix)
    
    reg = LinearRegression().fit(matrix[:, 0:2], matrix[:, 2])
    print(reg.coef_)
    print(reg.intercept_)
'''

    mat2 = np.array([
        [0,0,-1,1],
        [2,0,-1,65],
        [1,1,-1,11],
        [0,2,-1,31],
        [3,3,-1,72],
        [4,1,1,30],
        [5,2,1,0],
        [1,4,1,21],
        [4,4,1,4],
        [5,5,1,15]
    ])
    alfa, conteo, theta = kernel_perceptron(mat2[:,0:2], mat2[:,2])
    print("Salida kernel perceptron")

    # for veces in range(20):
    while not numpy.array_equal(conteo, errores):
        mat2 = numpy.random.permutation(mat2)
        x = mat2[:, 0:2]
        y = mat2[:, 2]
        errores = mat2[:, 3]
        alfa, conteo, theta = kernel_perceptron(x, y)

    print(theta)

