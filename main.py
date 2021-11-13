import numpy as np
from sklearn.cluster import KMeans

def clasificar(x, c0, c1):
    for element in x:
        dist0 = np.linalg.norm(element[0:2]-c0, ord=2)
        dist1 = np.linalg.norm(element[0:2]-c1, ord=2)
        if dist0 < dist1:
            element[2] = 0
        else:
            element[2] = 1


def mover(x):
    c0 = np.array([0,0], dtype=float)
    c1 = np.array([0,0], dtype=float)
    sum0 = 0
    sum1 = 1
    for element in x:
        if element[2] == 0:
            c0 += element[0:2]
            sum0 += 1
        else:
            c1 += element[0:2]
            sum1 += 1

    return c0/sum0, c1/sum1

if __name__ == '__main__':
    x = np.array([[0,-6,0], [4,4, 0], [0,0,0], [-5,2,0]], dtype=float)
    k = 2
    c0, c1 = np.array([-5, 2]), np.array([0, -6])
    for i in range(100):
        clasificar(x, c0, c1)
        c0, c1 = mover(x)

    print(c0)
    print(c1)
    print(x)

    y = np.array([[0,-6], [4,4], [0,0], [-5,2]], dtype=float)
    kmedoids = KMeans(n_clusters=2).fit(y)
    print(kmedoids.cluster_centers_)
    print(kmedoids.labels_)