import numpy as np
np.set_printoptions(precision=2)
from tqdm import tqdm
import gym
from time import perf_counter

sep='-'*80

env_name='FrozenLake8x8-v0'
#env_name='FrozenLake-v0'

print(sep)
print(f'{env_name} info:')

env=gym.make(env_name)
env.nS=env.observation_space.n
env.nA=env.action_space.n
print(env.__dict__)
print(env.reset())
print(env.render())
print(env.step(0))
print(env.P[0])

print(sep)
print(f'play once & test:')

def play(p,render=False):
    done,ret,s=False,0,env.reset()
    while not done:
        a=np.random.choice(env.nA,p=p[s])
        if render:
            env.render()
            print(f'state: {s}, action: {a}')
        s,r,done,info=env.step(a)
        ret+=r
    return ret

def test(p,reps=100):
    total=0
    for i in tqdm(range(reps),desc='testing',unit='eps'):
        total+=play(p)
    return total / reps

uniform_p=np.ones((env.nS,env.nA))/env.nA
print(test(uniform_p))

def value_iter(env,reps=100,gamma=0.9):
    v=np.zeros(env.nS)
    for i in tqdm(range(reps),desc='value iteration',unit='eps'):
        for s in range(env.nS):
            v_best=-float('inf')
            for a in range(env.nA):
                v_cur=0
                for p,s_,r,done in env.P[s][a]:
                    v_cur+=p*(r+(1-done)*gamma*v[s_])
                v_best=max(v_best,v_cur)
            v[s]=v_best
    q=np.zeros((env.nS,env.nA))
    for s in range(env.nS):
        for a in range(env.nA):
            for p,s_,r,done in env.P[s][a]:
                q[s,a]+=p*(r+(1-done)*gamma*v[s_])
    return np.eye(env.nA)[np.argmax(q,axis=-1)]

p=value_iter(env)
print(test(p))

def policy_iter(env,eval_reps=1000,reps=10,gamma=1.):
    policy=np.ones((env.nS,env.nA))
    for i in tqdm(range(reps),desc=f'policy iteration (gamma: {gamma})',unit='eps'):
        #eval
        v=np.zeros(env.nS)
        for j in range(eval_reps):
            delta=0
            for s in range(env.nS):
                qs=np.zeros(env.nA)
                for a in range(env.nA):
                    for p,s_,r,d in env.P[s][a]:
                        qs[a]+=p*(r+gamma*(1-d)*v[s_])
                v[s]=sum(policy[s]*qs)
        #improve
        for s in range(env.nS):
            qs=np.zeros(env.nA)
            for a in range(env.nA):
                for p,s_,r,d in env.P[s][a]:
                    qs[a]+=p*(r+gamma*(1-d)*v[s_])
            policy[s]=0.
            a=np.argmax(qs)
            policy[s,a]=1.
    return policy

p=policy_iter(env)
print(test(p))
