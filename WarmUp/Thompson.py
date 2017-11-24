import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

N = 1000
data = np.empty(N)

for i in range(N):
    data[i] = np.round(np.random.random())

a = 1
b = 1
for i in range(N):
    a = a + data[i]
    b = b + (1 - data[i])
    if i % 200 == 0:
        # distr = np.power(np.arange(0,1,0.01), a-1) * np.power(1 - np.arange(0,1,0.01), b-1)
        distr = beta.pdf(np.arange(0,1,0.01), a-1, b-1)
        plt.plot(distr)
        plt.show()