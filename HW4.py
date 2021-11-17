import numpy as np

def normal(x, mu, sigma2):
    d = 1
    numerador = np.exp (-1*np.square(np.absolute(x-mu))/(2*sigma2))
    denominador = 1/ np.sqrt(2*np.pi*sigma2)
    return numerador * denominador

# theta = np.array([.5,.5,6,7,1,4])
pi1 = .5
pi2 = .5
mu1 = 6
mu2 = 7
sigma1 = 1
sigma2 = 4
muestras = np.array([-1,0,4,5,6])
l = 0

for x in muestras:
    prob1 = normal(x, mu1, sigma1)
    prob2 = normal(x, mu2, sigma2)
    if prob1 > prob2:
        print("Para la muestra {} elijo 1".format(x))
    else:
        print("Para la muestra {} elijo 2".format(x))
    aporte = np.log(pi1*prob1 + pi2*prob2)
    l = l + aporte

print(l)

'''
#### CLUSTER 1 #####
prob_x = theta[0]
mu = theta[2]
sigma2 = theta[4]

for i in muestras:
    pdf = normal(i, mu, sigma2)
    prob_y = prob_x* pdf/ (prob_x * pdf + (1-prob_x) * normal(i, theta[3], theta[5]))
    l = l + prob_y * np.log(prob_x * normal(i, mu, sigma2))

print(l)

'''