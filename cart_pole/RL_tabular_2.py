import numpy as np
import gym
from gym.wrappers.time_limit import TimeLimit
from matplotlib import pyplot as plt

N_S = 10
ALPHA = 1e-2

def built_state(vec:list) -> int:
    return int("".join(map(lambda x : str(int(x)), vec)))

def to_bin(val, bins):
    return np.digitize([val],bins)[0]

class FeatureTransformer:
    def __init__(self):
        self.cart_position_bins = np.linspace(-2.4,2.4,N_S-1)
        self.cart_velocity_bins = np.linspace(-2,2,N_S-1)
        self.pole_angle_bins = np.linspace(-0.4,0.4,N_S-1)
        self.pole_velocity_bins = np.linspace(-3.5,3.5,N_S-1)

    def transform(self, observation:list):
        c_pose, c_vel, p_ang, p_vel = observation
        return built_state([
            to_bin(c_pose, self.cart_position_bins),
            to_bin(c_vel, self.cart_velocity_bins),
            to_bin(p_ang, self.pole_angle_bins),
            to_bin(p_vel, self.pole_velocity_bins)
        ])


class Model:
    def __init__(self, env:TimeLimit, feature_transformer: FeatureTransformer):
        self.env = env
        self.ft = feature_transformer
        num_states = N_S**env.observation_space.shape[0]
        num_actions = env.action_space.n
        self.Q = np.random.uniform(-1, 1, (num_states, num_actions))

    def predict(self, s):
        x = self.ft.transform(s)
        return self.Q[x]

    def update(self, s, a, G):
        x = self.ft.transform(s)
        self.Q[x,a] += ALPHA*(G - self.Q[x,a])

    def sample_action(self, s, eps):
        if np.random.random() < eps:
            return self.env.action_space.sample()
        else:
            pred = self.predict(s)
            return np.argmax(pred)

def play_one(model:Model, eps, gamma):
    obs = model.env.reset()
    done = False
    iters = 0
    total_rew = 0
    while not done:
        a = model.sample_action(obs,eps)
        prev_obs = obs
        obs, rew, done, info = model.env.step(a)

        total_rew += rew

        if done and iters < 199:
            rew = -300

        G = rew + gamma*np.max(model.predict(obs))
        model.update(prev_obs,a, G)

        iters += 1

    return total_rew


def plot_running_avg(totalrewards:np.ndarray):
    N = len(totalrewards)
    running_avg = np.empty(N)
    for i in range(N):
        running_avg[i] = totalrewards[max(0,i-100):(i+1)].mean()
    plt.plot(running_avg)
    plt.show()

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    ft = FeatureTransformer()
    model = Model(env, ft)
    gamma = 0.9

    N = 10000
    totalrewards = np.empty(N)

    for i in range(N):
        eps = 1.0/np.sqrt(i+1)
        totalreward = play_one(model, eps, gamma)
        totalrewards[i] = totalreward
        if i % 100 == 0:
            print("episode:", i, "total reward:", totalreward, "eps:", eps)
    print("avg reward for last 100 episodes:", totalrewards[-100:].mean())
    print("total steps:", totalrewards.sum())

    plt.plot(totalrewards)
    plt.title('Rewards')
    plt.show()

    plot_running_avg(totalrewards)