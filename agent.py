import numpy as np
from env import Environment
from consts import *
#%%
#data = np.random.random()
#CLASSES = 2
##FEATURE_DIM = 50
#STATE_DIM = FEATURE_DIM * 2
#ACTION_DIM = FEATURE_DIM + CLASSES
#AGENTS = 1000
#EPSILON_START   = 0.5
#MAX_MASK_CONST = 1.e6
#%%
class Agent():
    def __init__(self, env, pool, brain):
        self.env  = env
        self.pool = pool
        self.brain = brain
        self.epsilon = EPSILON_START
        self.s = self.env.reset()
        self.r = list()
    def store(self, x):
        self.pool.put(x)
    def act(self, s):
        m = np.zeros((AGENTS, ACTION_DIM))    # create max_mask
        m[:, CLASSES:] = s[:, FEATURE_DIM:]
        if self.epsilon < 1.0:
            p = self.brain.predict_np(s) - MAX_MASK_CONST * m     # select an action not considering those already performed
            a = np.argmax(p, axis=1)
        else:
            a = np.zeros(AGENTS, dtype=np.int32)
        # override with random action
        rand_agents = np.where( np.random.rand(AGENTS) < self.epsilon )[0]
        rand_number = np.random.rand(len(rand_agents))

        for i in range(len(rand_agents)):
            agent = rand_agents[i]

            possible_actions = np.where( m[agent] == 0. )[0]     # select a random action, don't repeat an action
            w = int(rand_number[i] * len(possible_actions))
            a[agent] = possible_actions[w]

        return a

    def step(self):
        a = self.act(self.s)
        s_, r = self.env.step(a)
        temp = r
        self.r.append(temp)

        self.store( (self.s, a, r, s_) )

        self.s = s_

    def update_epsilon(self, epoch):
        if epoch >= EPSILON_EPOCHS:
            self.epsilon = EPSILON_END
        else:
            self.epsilon = EPSILON_START + epoch * (EPSILON_END - EPSILON_START) / EPSILON_EPOCHS
            
#%%
#agent = Agent(env,pool,brain)            
