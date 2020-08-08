import tensorflow.compat.v2 as tf
from myutils import *
import pandas as pd
import gym
import numpy as np
import matplotlib.pyplot as plt

class PG(RLAgent):
    def __init__(self,env,gamma=0.9,lr=0.1,eps=0.01):

class TD(RLAgent):
    def __init__(self,env,gamma=0.9,lr=0.1,eps=0.01):
        super().__init__(env=env)
        self.gamma=gamma
        self.lr=lr
        self.eps=eps
        self.q=None

    def decide(self,state,train=False):
        #train using epsilon-greedy
        if train and np.random.uniform()<self.eps:
            return np.random.choice(self.env.nA)
        else:
            return np.argmax(self.q[state])

    def learn(self,s,a,r,s1,a1,done):
        pass

class SARSA(TD):
    def __init__(self,env,gamma=0.9,lr=0.2,eps=0.01):
        super().__init__(env,gamma=gamma,lr=lr,eps=eps)

    def learn(self,s,a,r,s1,a1,done):
        #on policy
        q1=r+self.gamma*(1-done)*self.q[s1,a1]
        err=q1-self.q[s][a]
        self.q[s][a]+=self.lr*err

class ExpectedSARSA(TD):
    def __init__(self,env,gamma=0.9,lr=0.1,eps=0.01):
        super().__init__(env,gamma=gamma,lr=lr,eps=eps)

    def learn(self,s,a,r,s1,a1,done):
        #off policy
        q1=r+self.gamma*(1-done)*(self.eps * np.mean(self.q[s1]) + (1.-self.eps) * self.q[s1].max())
        err=q1-self.q[s][a]
        self.q[s][a]+=self.lr*err

class QLearning(TD):
    #sometimes you'd have some function discretizing q-table
    def __init__(self,env,gamma=0.9,lr=0.1,eps=0.01):
        super().__init__(env,gamma=gamma,lr=lr,eps=eps)

    def learn(self,s,a,r,s1,a1,done):
        #off policy
        q1=r+self.gamma*(1-done)*self.q[s1].max()
        err=q1-self.q[s][a]
        self.q[s][a]+=self.lr*err

class DoubleQLearning(TD):
    #sometimes you'd have some function discretizing q-table
    def __init__(self,env,gamma=0.9,lr=0.1,eps=0.01):
        super().__init__(env,gamma=gamma,lr=lr,eps=eps)
        self.q1=np.zeros((self.env.nS,self.env.nA))

    def learn(self,s,a,r,s1,a1,done):
        #flip a coin
        if np.random.choice(2):
            self.q,self.q1=self.q1,self.q
        #off policy
        q1=r+self.gamma*(1-done)*self.q1[s1,np.argmax(self.q[s1])]
        err=q1-self.q[s][a]
        self.q[s][a]+=self.lr*err

class SARSALambda(TD):
    def __init__(self,env,gamma=0.9,lr=0.2,eps=0.01,beta=1.,lambd=0.6):
        super().__init__(env,gamma=gamma,lr=lr,eps=eps)
        self.lambd=lambd
        self.beta=beta
        #eligibility trace
        self.e=np.zeros_like(self.q)

    def learn(self,s,a,r,s1,a1,done):
        #decay & update eligibility trace
        self.e*=self.gamma*self.lambd
        self.e[s,a]=self.beta*self.e[s,a]+1
        #update q-table weighted by the trace
        q1=r+self.gamma*(1-done)*self.q[s1,a1]
        err=q1-self.q[s][a]
        self.q+=self.lr*self.e*err

class SARSATileEncoding(TD):
    def __init__(self,env,gamma=1.,lr=0.1,eps=1e-3,layers=8,features=1893):
        '''
        this solves MountainCar-v0 using tile encoding. there are 8 layers, the first layer has 64 tiles. the other seven each have 81 tiles. each possible tile layout has 3 states and together you have 1893 features.
        '''
        self.env=env
        self.env.nA=env.action_space.n
        self.gamma=gamma
        self.lr=lr
        self.eps=eps
        self.encoder=TileCoder(layers,features)
        #smaller q-table
        self.q=np.zeros(features)
        self.range=(self.env.high-self.env.low)

    def encode(self,s,a):
        normalized_s=tuple((s-self.env.low)/self.range)
        a1=(a,)
        return self.encoder(normalized_s,a1)

    def get_q(self,s,a):
        features=self.encode(s,a)
        return self.q[features].sum()

    def set_q(self,s,a,err):
        features=self.encode(s,a)
        self.q[features]+=self.lr*err

    def decide(self,s,train=False):
        if train and np.random.rand()<self.eps:
            return np.random.randint(self.env.nA)
        else:
            qs=[self.get_q(s,a) for a in range(self.env.nA)]
            return np.argmax(qs)

    def learn(self,s,a,r,s1,a1,done):
        q1=r+self.gamma*(1-done)*self.get_q(s1,a1)
        err=q1-self.get_q(s,a)
        self.set_q(s,a,err)

class SARSALambdaTileEncoding(SARSATileEncoding):
    def __init__(self,env,layers=8,features=1893,
                 gamma=1.,lr=.03,eps=1e-3,lambd=.9):
        super().__init__(env=env,layers=layers,
                        features=features,gamma=gamma,
                        lr=lr,eps=eps)
        self.lambd=lambd
        #eligibility trace
        self.e=np.zeros(features)

    def set_q(self,s,a,err):
        #unlike vanilla SARSA, q-table updates are weighted by trace
        features=self.encode(s,a)
        self.q+=self.lr*self.e*err

    def learn(self,s,a,r,s1,a1,done):
        q1=0 #scope issue for err's assignment
        if not done:
            #decay & update trace (add beta?)
            self.e*=self.gamma*self.lambd
            features=self.encode(s,a)
            self.e[features]=1.
        #update q-table weighted by trace
            q1=r+self.gamma*self.get_q(s1,a1)
        err=q1-self.get_q(s,a)
        self.set_q(s,a,err)
        if done:
            #reset trace for next episode
            self.e=np.zeros_like(self.e)

class DQNReplayer:
    def __init__(self, capacity):
        self.memory = pd.DataFrame(index=range(capacity),
                columns=['observation', 'action', 'reward',
                'next_observation', 'done'])
        self.i = 0
        self.count = 0
        self.capacity = capacity

    def store(self, *args):
        self.memory.loc[self.i] = args
        self.i = (self.i + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)

    def sample(self, size):
        indices = np.random.choice(self.count, size=size)
        return (np.stack(self.memory.loc[indices, field]) for field in
                self.memory.columns)

class DQNAgent:
    def __init__(self, env, net_kwargs={}, gamma=0.99, epsilon=0.001,
             replayer_capacity=10000, batch_size=64):
        observation_dim = env.observation_space.shape[0]
        self.action_n = env.action_space.n
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.replayer = DQNReplayer(replayer_capacity)
        self.evaluate_net = self.build_network(input_size=observation_dim,
                output_size=self.action_n, **net_kwargs)
        self.target_net = self.build_network(input_size=observation_dim,
                output_size=self.action_n, **net_kwargs)
        self.target_net.set_weights(self.evaluate_net.get_weights())

    def build_network(self, input_size, hidden_sizes, output_size,
                activation=tf.nn.relu, output_activation=None,
                learning_rate=0.01):
        model = keras.Sequential()
        for layer, hidden_size in enumerate(hidden_sizes):
            kwargs = dict(input_shape=(input_size,)) if not layer else {}
            model.add(keras.layers.Dense(units=hidden_size,
                    activation=activation, **kwargs))
        model.add(keras.layers.Dense(units=output_size,
                activation=output_activation))
        optimizer = tf.optimizers.Adam(lr=learning_rate)
        model.compile(loss='mse', optimizer=optimizer)
        return model

    def learn(self, observation, action, reward, next_observation, next_action, done):
        self.replayer.store(observation, action, reward, next_observation, done)
        observations, actions, rewards, next_observations, dones = \
            self.replayer.sample(self.batch_size)
        next_qs = self.target_net.predict(next_observations)
        next_max_qs = next_qs.max(axis=-1)
        us = rewards + self.gamma * (1. - dones) * next_max_qs
        targets = self.evaluate_net.predict(observations)
        targets[np.arange(us.shape[0]), actions] = us
        self.evaluate_net.fit(observations, targets, verbose=0)
        if done:
            self.target_net.set_weights(self.evaluate_net.get_weights())

    def decide(self, observation,train=False):
        if train and np.random.rand() < self.epsilon:
            return np.random.randint(self.action_n)
        qs = self.evaluate_net.predict(observation[np.newaxis])
        return np.argmax(qs)

class DoubleDQNAgent(DQNAgent):
    def learn(self, observation, action, reward, next_observation, next_action, done):
        self.replayer.store(observation, action, reward, next_observation, done)
        observations, actions, rewards, next_observations, dones = \
            self.replayer.sample(self.batch_size)
        next_eval_qs = self.evaluate_net.predict(next_observations)
        next_actions = next_eval_qs.argmax(axis=-1)
        next_qs = self.target_net.predict(next_observations)
        next_max_qs = next_qs[np.arange(next_qs.shape[0]), next_actions]
        us = rewards + self.gamma * next_max_qs * (1. - dones)
        targets = self.evaluate_net.predict(observations)
        targets[np.arange(us.shape[0]), actions] = us
        self.evaluate_net.fit(observations, targets, verbose=0)
        if done:
            self.target_net.set_weights(self.evaluate_net.get_weights())

if __name__=='__main__':
    name='CartPole-v0'

    env=bootstrap(name)

    agent=PG(env)
    pg_agent=solve(agent,env,1000,100)

    pd.DataFrame(pg_agent.q)
    policy=np.argmax(sarsa_agent.q,axis=1)
    pd.DataFrame(policy)

    agent=ExpectedSARSA(env)
    expected_sarsa_agent=solve(agent,env,5000,300,100)

    agent=QLearning(env)
    q_learning_agent=solve(agent,env,4000,300,100)

    agent=DoubleQLearning(env)
    double_q_learning_agent=solve(agent,env,9000,300,100)

    agent=SARSALambda(env)
    sarsa_lambda_agent=solve(agent,env,5000,300,100)

    name='MountainCar-v0'

    env=bootstrap(name)

    agent=SARSATileEncoding(env)
    sarsa_tile_encoding_agent=solve(agent,env,400,100)

    agent=SARSALambdaTileEncoding(env)
    sarsa_lambda_tile_encoding_agent=solve(agent,env,140,100)

    net_kwargs = {'hidden_sizes' : [64, 64], 'learning_rate' : 0.001}
    agent = DQNAgent(env, net_kwargs=net_kwargs)
    dqn_agent=solve(agent,env,500,100)

    net_kwargs = {'hidden_sizes' : [64, 64], 'learning_rate' : 0.001}
    agent = DoubleDQNAgent(env, net_kwargs=net_kwargs)
    double_dqn_agent=solve(agent,env,500,100)
