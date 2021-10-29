import numpy as np
import gym

import blackjack

class QLearning:
  def __init__(self):
    self.max_pl_hand = 32 # max sum of player hand
    self.max_dl_hand = 11 # max value of visiable card of dealer
    self.a_num = 2 # number of actions
    self.s_num = self.max_pl_hand * self.max_dl_hand * 2 # number of states
    # self.game = blackjack.BlackjackEnv(natural=True)
    self.game = gym.make('Blackjack-v0', natural=True)
  
  def init_q(self):
    self.Q = np.zeros((self.s_num, self.a_num))
      
  def get_policy(self):
    return self.Q.argmax(1)
  
  def run_episode(self, al, e, g):
    obs = self.game.reset()
    s = self._obs_to_idx(obs)
    pi = self.get_policy()
    if np.random.rand() > e:
        a = pi[s]
    else:
        a = np.random.randint(self.a_num)

    stop = False
    while not stop:
        pi = self.get_policy()
        obs, reward, stop, _ = self.game.step(a)
        s_new = self._obs_to_idx(obs)
        
        if np.random.rand() > e:
            a_new = pi[s_new]
        else:
            a_new = np.random.randint(self.a_num)
        
        self.Q[s, a] = self.Q[s, a] + al * (reward + g * np.max(self.Q[s_new]) - self.Q[s, a])
        s = s_new
        a = a_new
              
  def train(self, al, e, g, n):
    self.init_q()
    pi = self.get_policy()

    for _ in range(n):
        self.run_episode(al, e, g)
        pi = self.get_policy()

    self.pi = pi

  def train_with_means(self, al, e, g, n, k, d):
    self.init_q()
    pi = self.get_policy()
    mean_rewards = []

    j = 0
    for i in range(n):
        self.run_episode(al, e, g)
        pi = self.get_policy()
        if i - j + 1 == k:
          self.pi = pi
          rewards = self.run_experiment(d)
          mean_rewards.append(np.round(np.mean(rewards), 3))
          j += k

    self.pi = pi

    return mean_rewards
      
  def run_experiment(self, n):
    pi = self.pi
    rewards = []

    for _ in range(n):
        reward = 0
        stop = False
        obs = self.game.reset()
        s = self._obs_to_idx(obs)
        
        while not stop:
            a = pi[s]
            obs, reward, stop, _ = self.game.step(a)
            s = self._obs_to_idx(obs)

        rewards.append(reward)

    return rewards

  def _obs_to_idx(self, obs):
    return (obs[0] - 1) * self.max_dl_hand * 2 + (obs[1] - 1) * 2 + obs[2]


class QLearningForDouble(QLearning):
  def __init__(self):
    self.max_pl_hand = 32
    self.max_dl_hand = 11
    self.a_num = 3
    self.s_num = self.max_pl_hand * self.max_dl_hand * 2
    self.game = blackjack.BlackjackDoubleEnv(natural=True)


class QLearningCount(QLearning):
  def __init__(self):
    self.max_pl_hand = 32
    self.max_dl_hand = 11
    self.deck_st = 21
    self.a_num = 3
    self.s_num = self.max_pl_hand * self.max_dl_hand * 2 * self.deck_st
    self.game = blackjack.BlackjackCountEnv(natural=True)
    self.game.start_game()

  def run_experiment(self, n):
    pi = self.pi
    rewards = []

    for _ in range(n):
        reward = 0
        stop = False
        obs = self.game.start_game()
        s = self._obs_to_idx(obs)
        
        while not stop:
            a = pi[s]
            obs, reward, stop, _ = self.game.step(a)
            s = self._obs_to_idx(obs)

        rewards.append(reward)

    return rewards

  def run_episode(self, al, e, g):
    obs = self.game.start_game()
    s = self._obs_to_idx(obs)
    pi = self.get_policy()
    if np.random.rand() > e:
        a = pi[s]
    else:
        a = np.random.randint(self.a_num)

    stop = False
    while not stop:
        pi = self.get_policy()
        obs, reward, stop, _ = self.game.step(a)
        s_new = self._obs_to_idx(obs)
        
        if np.random.rand() > e:
            a_new = pi[s_new]
        else:
            a_new = np.random.randint(self.a_num)
        
        self.Q[s, a] = self.Q[s, a] + al * (reward + g * np.max(self.Q[s_new]) - self.Q[s, a])
        s = s_new
        a = a_new

  def _obs_to_idx(self, obs):
    return (obs[0] - 1) * self.max_dl_hand * 2  * self.deck_st + (obs[1] - 1) * 2 * self.deck_st + obs[2] * self.deck_st + obs[3]