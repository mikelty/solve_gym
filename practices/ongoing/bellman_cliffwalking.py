import scipy
from tqdm import tqdm
import numpy as np
import gym

env_name='CliffWalking-v0'
env=gym.make(env_name)
print(env.__dict__)

def test(env,pi,reps=10):
    total_R=0
    for _ in tqdm(range(reps),desc='test',unit='eps'):
        d,s=False,env.reset()
        while not d:
            a=np.random.choice(env.nA,p=pi[s])
            s,r,d,i=env.step(a)
            total_R+=r
    print(f'score: {total_R/reps}')
    return total_R/reps

random_pi=np.ones((env.nS,env.nA))/env.nA
test(env,random_pi)
perfect_pi=np.ones(env.nS).reshape(env.shape) #always move right, except:
perfect_pi[-1,0]=0 #move up in lower left corner
perfect_pi[:,-1]=2 #move down on the rightmost column.
print(f'perfect policy: \n{perfect_pi}')
perfect_pi=np.eye(env.nA)[perfect_pi.reshape(-1).astype(np.int)]
test(env,perfect_pi)

def eval_bellman(env,pi,gamma=1.):
    left,right=np.eye(env.nS),np.zeros((env.nS))
    for s in range(env.nS-1):
        for a in range(env.nA):
            psa=pi[s][a]
            for p,s_,r,d in env.P[s][a]:
                left[s][s_]-=gamma*psa*p
                right[s]+=psa*p*r
    v=scipy.linalg.solve(left,right)
    q=np.zeros((env.nS,env.nA))
    for s in range(env.nS-1):
        for a in range(env.nA):
            for p,s_,r,d in env.P[s][a]:
                q[s][a]+=p*(r+gamma*v[s_])
    return v,q

v,q=eval_bellman(env,random_pi)
print(f'random policy: \n v: \n {v} \n q: \n {q}')
v,q=eval_bellman(env,perfect_pi)
print(f'perfect policy: \n v: \n {v} \n q: \n {q}')

def optimize_bellman(env,gamma=1.):
    '''
    x is v - nS
    A_ub is negated v-gamma*transition_probability*v_ - (nS*nA) by nS
    b_ub is negated r*transition_probability - ns*nA
    all other parameters are essentially irrelevant
    '''
    trans_prob=np.zeros((env.nS,env.nA,env.nS))
    rewards=np.zeros((env.nS,env.nA))
    for s in range(env.nS-1):
        for a in range(env.nA):
            for p,s_,r,d in env.P[s][a]:
                trans_prob[s][a][s_]+=p
                rewards[s][a]+=r*p
    c=np.ones(env.nS)
    bounds=[(None,None),]*env.nS
    sln=scipy.optimize.linprog(c,
                               gamma*trans_prob.reshape(-1,env.nS)-np.repeat(np.eye(env.nS),env.nA,axis=0),
                               -rewards.reshape(-1),
                               bounds=bounds,
                               method='interior-point')
    v=sln.x
    q=rewards+gamma*np.dot(trans_prob,v)
    return v,q

v,q=optimize_bellman(env)
pi=np.eye(env.nA)[q.argmax(axis=1)]
print(f'bellman policy: \n {pi.argmax(axis=1).reshape(env.shape)}')
print(f'bellman policy: \n {pi.argmax(axis=1)}')
test(env,pi)
