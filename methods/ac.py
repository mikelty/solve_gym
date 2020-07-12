from myutils import *
import gym
import numpy as np
import scipy

class AC(RLAgent):
    def __init__(self,env,gamma=1.):
        super().__init__(env=env)
        self.gamma=gamma
        self.v=None
        self.q=None
        self.env.nS=self.env.observation_space.n
        self.env.nA=self.env.action_space.n

    def decide(self,state,train=False):
        return np.argmax(self.q[state])

    def evaluate(self,policy):
        # identity transition minus heuristics transition
        A=np.eye(self.env.nS)
        # rewards
        B=np.zeros((self.env.nS))
        # for each transition, update a and b accordingly
        for s in range(self.env.nS-1):
            for a in range(env.nA):
                pa=policy[s][a]
                for ps1, s1, reward, done in env.P[s][a]:
                    A[s,s1]-=self.gamma*pa*ps1
                    B[s]+=reward*pa*ps1
        # state values
        v=scipy.linalg.solve(A,B)
        # action values
        q=np.zeros((self.env.nS,self.env.nA))
        # for each transition, update q accordingly
        for s in range(self.env.nS-1):
            for a in range(env.nA):
                for ps1, s1, reward, done in env.P[s][a]:
                    q[s,a]+=ps1*(reward+self.gamma*v[s1])
        return v,q

    def learn(self,state=None,action=None,reward=None,new_state=None,new_action=None,done=None):
        print(f"agent learning by solving bellman equation of {self.env.nS} variables...")
        # transition prob. matrix
        p=np.zeros((self.env.nS,self.env.nA,self.env.nS))
        # reward matrix
        r=np.zeros((self.env.nS,env.nA))
        # populate p and r
        for s1 in range(self.env.nS-1):
            for a in range(env.nA):
                for ps2, s2, reward, done in env.P[s1][a]:
                    p[s1][a][s2]=ps2
                    r[s1][a]+=reward*ps2
        c=np.ones(self.env.nS)
        # negative of A in evaluate expanded along actions
        a_ub=self.gamma*p.reshape(-1,self.env.nS) \
             - np.repeat(np.eye(self.env.nS),env.nA,axis=0)
        # negative of B in evaluate expanded along actions
        b_ub=-r.reshape(-1)
        a_eq=np.zeros((0,self.env.nS))
        b_eq=np.zeros(0)
        bounds=[(None, None)]*self.env.nS
        sln=scipy.optimize.linprog(c,a_ub,b_ub,a_eq,b_eq,bounds=bounds,method='interior-point')
        v=sln.x
        q=r+self.gamma*np.dot(p,v)
        self.v,self.q=v,q

if __name__=='__main__':

    np.set_printoptions(precision=3)
    name='CliffWalking-v0'
    env=bootstrap(name,render=True)

    agent=Bellman(env)
    bellman_agent=solve(agent,env,0,0,1)
    bellman_policy=np.copy(bellman_agent.q)
    print(sep)
    bellman_actions=np.argmax(bellman_policy,axis=-1).reshape(env.shape)
    print(f"bellman actions: \n{bellman_actions}")

    print(sep)
    random_policy=np.ones((env.nS,env.nA))/env.nA
    state_values, action_values=agent.evaluate(random_policy)
    state_values=state_values.reshape(env.shape)
    action_values=np.argmax(action_values,axis=1).reshape(env.shape)
    print(f"random policy state values:\n{state_values}")
    print(f"random policy action values:\n{action_values}")

    print(sep)
    perfect_moves=np.ones(env.observation_space.n,dtype=int).reshape(env.shape)
    perfect_moves[-1,:]=0 #at bottom row, move up
    perfect_moves[:,-1]=2 #at rightmost column, move down
    print(f"perfect moves:\n{perfect_moves}")
    perfect_policy=np.eye(env.action_space.n)[perfect_moves.reshape(-1)]
    state_values, action_values=agent.evaluate(perfect_policy)
    state_values=state_values.reshape(env.shape)
    action_values=np.argmax(action_values,axis=-1).reshape(env.shape)
    print(f"perfect policy state values:\n{state_values}")
    print(f"perfect policy action values:\n{action_values}")
