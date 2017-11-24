import gym
import numpy as np
from matplotlib import pyplot as plt
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import SGDRegressor
from gym.wrappers.time_limit import TimeLimit
from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import StandardScaler

class mySGDRegressor:
    def __init__(self, dimensions):
        self.lr = 0.1
        self.weights = np.random.rand(dimensions)/np.sqrt(dimensions)

    def predict(self,X):
        return X.dot(self.weights)

    def partial_fit(self,X,Y):
        self.weights += self.lr*(Y - X.dot(self.weights)).dot(X)

class FeatureTransformer:
    def __init__(self, env:TimeLimit):
        observation_examples = np.hstack((np.random.random((20000,1)) * 4.8 - 2.4,
                                          np.random.random((20000,1)) * 4.0 - 2.0,
                                          np.random.random((20000,1)) * 0.8 - 0.4,
                                          np.random.random((20000,1)) * 8.0 - 4.0))
        scaler = StandardScaler()
        scaler.fit(observation_examples)


        featurizer = FeatureUnion([
            ('rbf1', RBFSampler(gamma=0.05, n_components=1000)),
            ('rbf2', RBFSampler(gamma=1.0, n_components=1000)),
            ('rbf3', RBFSampler(gamma=0.5, n_components=1000)),
            ('rbf4', RBFSampler(gamma=0.1, n_components=1000)),
        ])

        examples = featurizer.fit_transform(scaler.transform(observation_examples))
        self.scaler = scaler
        self.featurizer = featurizer
        self.dim = examples.shape[1]

    def transform(self, obs):
        scaled = self.scaler.transform(obs)
        return self.featurizer.transform(scaled)

class Model:
    def __init__(self, env, feature_transformer:FeatureTransformer, learning_rate, start_over=False):
        self.env = env
        if start_over:
            self.models = [] # type: list[SGDRegressor]
            for i in range(env.action_space.n):
                model = mySGDRegressor(feature_transformer.dim)
                self.models.append(model)
                self.K = 0
        else:
            self.models = np.load('models.npy')[0]
            print(self.models[0].weights)
            self.K = np.load('models.npy')[1]
        self.ft = feature_transformer


    def predict(self, s):
        x = self.ft.transform(np.atleast_2d(s))
        assert (len(x.shape)==2)
        return np.array([m.predict(x)[0] for m in self.models])
        # return list(map(lambda model, x: model.predict(x)[0], self.models, [x]*3))

    def update(self, s, a, G):
        x = self.ft.transform(np.atleast_2d(s))
        self.models[a].partial_fit(x, [G])

    def sample_action(self, s, eps):
        if np.random.random() < eps:
            return self.env.action_space.sample()
        else:
            pred = self.predict(s)
            return np.argmax(pred)


def play_one(model, eps=0, gamma=0.99):
    obs = model.env.reset()
    done = False
    iters = 0
    total_rew = 0
    while not done:
        model.env.render()
        a = model.sample_action(obs, eps)
        # a = 0
        prev_obs = obs
        obs, rew, done, info = model.env.step(a)

        total_rew += rew

        if done and model.env._elapsed_steps < 199:
            rew = -200

        G = rew + gamma * np.max(model.predict(obs))
        model.update(prev_obs, a, G)

        iters += 1
        # time.sleep(0.1)
    return total_rew


def plot_running_avg(totalrewards:np.ndarray):
    N = len(totalrewards)
    running_avg = np.empty(N)
    for i in range(N):
        running_avg[i] = totalrewards[max(0,i-100):(i+1)].mean()
    plt.plot(running_avg)
    plt.show()



def main():
    env = gym.make('CartPole-v0')
    ft = FeatureTransformer(env)
    model = Model(env, ft, 'constant', start_over=True)

    N = 500
    k = model.K
    totalrewards = np.empty(N)
    gamma = 0.99

    for i in range(k,N+k):
        eps = 0.5 / np.sqrt(i + 1)
        # eps = 0
        totalreward = play_one(model, eps=eps, gamma=gamma)
        totalrewards[i-k] = totalreward
        if i% 10 ==0:
            print("episode:", i, "total reward:", totalreward, "eps:", eps)
            np.save('models', [np.array(model.models),i])
            print(model.models[0].weights)
    print("avg reward for last 100 episodes:", totalrewards[-100:].mean())
    print("total steps:", totalrewards.sum())

    # plt.plot(totalrewards)
    # plt.title('Rewards')
    # plt.show()

    plot_running_avg(totalrewards)


