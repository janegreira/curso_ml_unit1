import numpy as np

def normal(x, mu, sigma2):
    d = 1
    return np.exp((-(x-mu)^2)/(2*sigma2)) / (2*np.pi*sigma2)^(d/2)

theta = np.array([.5,.5,6,7,1,4])
muestras = np.array([-1,0,4,5,6])
l = 0


#### CLUSTER 1 #####
prob_x = theta[0]
mu = theta[2]
sigma2 = theta[4]

for i in muestras:
    pdf = normal(i, mu, sigma2)
    prob_y = prob_x* pdf/ (prob_x * pdf + (1-prob_x) * normal(i, theta[3], theta[5]))
    l = l + prob_y * np.log(prob_x * normal(i, mu, sigma2))

print(l)

