import numpy as np
import matplotlib.pyplot as plt

class Bandit:
  def __init__(self, m, mean = 0):
    self.m = m
    self.mean = mean
    self.N = 0
    self.samples = []

  def pull(self):
    x = np.random.randn() + self.m
    self.samples.append(x)
    return x


  def update(self, x):
    self.N += 1
    self.mean = (1 - 1.0/self.N)*self.mean + 1.0/self.N*x


def run_experiments(m1, m2, m3, eps, N):
    bandits = [Bandit(m1), Bandit(m2), Bandit(m3)]

    data = np.empty(N)

    for i in xrange(N):
        g = np.random.random()

        if g < eps:
            j = np.random.choice(3)
        else:
            j = np.argmax([b.mean for b in bandits])

        x = bandits[j].pull()
        bandits[j].update(x)
        data[i] = x

    average = np.cumsum(data)/(np.arange(1, N+1))
    return average


def run_experiments_opt(m1, m2, m3, eps, N):
    bandits = [Bandit(m1, 10), Bandit(m2,10), Bandit(m3,10)]

    data = np.empty(N)

    for i in xrange(N):

        j = np.argmax([b.mean for b in bandits])

        x = bandits[j].pull()
        bandits[j].update(x)
        data[i] = x

    average = np.cumsum(data)/(np.arange(1, N+1))
    return average

def run_experiments_ucb(m1, m2, m3, eps, N):
    bandits = [Bandit(m1), Bandit(m2), Bandit(m3)]

    data = np.empty(N)

    for i in xrange(N):
        j = np.argmax([b.mean + np.sqrt(2*np.log(i)/(b.N+10e-3)) for b in bandits])

        x = bandits[j].pull()
        bandits[j].update(x)
        data[i] = x

    average = np.cumsum(data) / (np.arange(1, N + 1))
    return average

def run_experiments_thomp(m1, m2, m3, eps, N):
    bandits = [Bandit(m1), Bandit(m2), Bandit(m3)]

    data = np.empty(N)

    for i in xrange(N):
        j = np.argmax([np.random.normal(b.mean, 1/(b.N+10e-3), 1) for b in bandits])

        b = bandits[j]
        x = b.pull()
        b.N += 1
        b.mean = (np.sum(b.samples))/(b.N+10e-3)
        data[i] = b.samples[-1]

    average = np.cumsum(data) / (np.arange(1, N + 1))
    return average

if __name__ == '__main__':
    N = 10000
    av1 = run_experiments(1, 2, 3, 0.05, N)
    av2 = run_experiments_opt(1, 2, 3, 0.05, N)
    av3 = run_experiments_ucb(1, 2, 3, 0.05, N)
    av4 = run_experiments_thomp(1, 2, 3, 0.05, N)

    plt.plot(av1, label='normal')
    plt.plot(av2, label='optimistic')
    plt.plot(av3, label='ucb')
    plt.plot(av4, label='Thompson')
    plt.legend()
    plt.plot(np.ones(N))
    plt.plot(2 * np.ones(N))
    plt.plot(3 * np.ones(N))
    plt.xscale('log')
    plt.show()