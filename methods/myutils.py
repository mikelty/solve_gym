from tqdm import tqdm
from time import perf_counter
import pandas as pd
import gym
import numpy as np
import matplotlib.pyplot as plt

sep='-'*30

class RLAgent:
    def __init__(self,env):
        self.env=env

    def decide(self,state,train=False):
        pass

    def learn(self,state,action,reward,new_state,new_action,done):
        pass

    def evaluate(self,policy):
        print("evaluation not implemented yet")

def bootstrap(name,render=False,print_more_info=None):
    print(sep)
    print(name)
    env=gym.make(name)
    try:
        env.nS=env.observation_space.n
    except:
        pass
    try:
        env.nA=env.action_space.n
    except:
        pass
    print(f"observation space : {env.observation_space}")
    print(f"action space : {env.action_space}")
    if render:
        print(f"visualization : {env.render()}")
    print(f"start: {env.reset()}")
    if print_more_info is not None:
        print("environment specific information:")
        print_more_info(env)
    return env

def play(agent,env,train=False,render=False):
    state, done, total_reward=env.reset(), False, 0
    while not done:
        action=agent.decide(state,train=train)
        new_state,reward,done,info=env.step(action)
        new_action=agent.decide(new_state,train=train)
        if train:
            agent.learn(state,action,reward,new_state,new_action,done)
        state=new_state
        total_reward+=reward
        if render:
            env.render()
    return total_reward

def solve(agent,env,rep,test_rep):
    '''
    train and test the RL agent.
    rep: how many rounds to train.
        if rep is 0, simply call the learn function (e.g. bellman, monte carlo)
    test_rep: how many rounds to test (take the average)
    '''
    #generate & print title
    print(sep)
    name='Unknown Gym Enviornment'
    try:
        name=type(vars(env)['env']).__name__
    except:
        pass
    try:
        name=vars(env)['spec'].id
    except:
        pass
    print(f"solving {name} using model {type(agent).__name__}")
    #train
    print("training")
    if rep==0:
        agent.learn()
    else:
        rewards=[]
        for i in tqdm(range(rep)):
            rewards.append(play(agent,env,train=True))
        plt.plot(rewards)
    #test
    print(f"testing agent {test_rep} times...")
    start=perf_counter()
    avg_reward=sum(play(agent,env) for _ in range(test_rep))/test_rep
    print(f"avg test reward: {avg_reward}")
    print(f"testing time: {perf_counter() - start}")
    return agent

class TileCoder:
    #tile encoding to shrink q table size
    def __init__(self,layers,features):
        self.layers=layers
        self.features=features
        self.codebook={}

    def get_feature(self,codeword):
        if codeword in self.codebook:
            return self.codebook[codeword]
        count=len(self.codebook)
        if count>=self.features:#don't understand this
            return hash(codeword) % self.features
        self.codebook[codeword]=count
        return count

    def __call__(self,floats={},ints=()):
        dim=len(floats)
        scaled_floats=tuple(f*self.layers**2 for f in floats)
        features=[]
        for layer in range(self.layers):
            codeword = (layer,) + tuple(int((f + (1 + dim * i) * layer) /
                    self.layers) for i, f in enumerate(scaled_floats)) + ints
            feature = self.get_feature(codeword)
            features.append(feature)
        return features
