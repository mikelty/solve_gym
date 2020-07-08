#TODO: add rounds parameter to policy and value iteration
from utils import *
import gym
import numpy as np

class DP(RLAgent):
    def __init__(self,env,gamma=1.,tolerance=1e-6):
        #note tolerance can be better divided into two tolerances
        super().__init__(env=env)
        self.gamma=gamma
        self.tolerance=tolerance
        self.policy=None

    def decide(self,state,train=False):
        return np.argmax(self.policy[state])

    def v2q(self,v,s):
        q=np.zeros(self.env.nA)
        for a in range(self.env.nA):
            for p,n,r,d in self.env.P[s][a]:
                q[a]+=p*(r+(1.-d)*v[n]*self.gamma)

    def evaluate(self,policy):
        v=np.zeros(self.env.nS)
        while True:
            delta=0
            for s in range(self.env.nS):
                new_vs=sum(policy[s]*self.v2q(v,s))
                delta=max(delta,abs(new_vs-v[s]))
                v[s]=new_vs
            if delta<self.tolerance:
                break
        return v

    def improve_policy(self,v,policy):
        optimal=True
        for s in range(self.env.observation_space.n):
            q=self.v2q(v,s)
            a=np.argmax(q)
            if policy[s][a]!=1.:
                optimal=False
                policy[s]=0.
                policy[s][a]=1.
        return optimal

class PolicyIteration(DP):
    def __init__(self,env,gamma=1.,tolerance=1e-6):
        super().__init__(env=env,gamma=gamma,tolerance=tolerance)

    def learn(self):
        policy=np.ones((self.env.nS,self.env.nA))/self.env.nA
        while True:
            values=self.evaluate(policy)
            if self.improve_policy(values,policy):
                break
        self.policy=policy

class ValueIteration(DP):
    def __init__(self,env,gamma=1.,tolerance=1e-6):
        super().__init__(env=env,gamma=gamma,tolerance=tolerance)

    def learn(self):
        v=np.zeros(self.env.nS)
        while True:
            delta=0
            for s in range(self.env.nS):
                new_vs=max(self.v2q(v,s))
                delta=max(delta,abs(new_vs-v[s]))
                v[s]=new_vs
            if delta<self.tolerance:
                break
        p=np.zeros((self.env.nS,self.env.nA))
        for s in range(self.env.nS):
            a=np.argmax(self.v2q(v,s))
            p[s][a]=1
        self.policy=p

if __name__=='__main__':

    name="FrozenLake-v0"
    env=bootstrap(name,render=True)

    agent=PolicyIteration(env)
    policy_iteration_agent=solve(agent,env,0,0,100)

    agent=ValueIteration(env)
    value_iteration_agent=solve(agent,env,0,0,100)
