import scipy
from tqdm import tqdm
import numpy as np
import gym

env_name='CliffWalking-v0'
env=gym.make(env_name)
print(env.__dict__)

def test(env,pi,reps=1):
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
perfect_pi=np.ones(env.nS).reshape(env.shape)
perfect_pi[-1,0]=0 #up in lower-left corner
perfect_pi[:,-1]=2 #down at rightmost column
print(perfect_pi)
perfect_pi=np.eye(env.nA)[perfect_pi.reshape(-1).astype(np.int)]
test(env,perfect_pi)

def eval_bellman(env,pi,gamma=1.):
    lhs,rhs=np.eye(env.nS),np.zeros((env.nS))
    for s in range(env.nS-1):
        for a in range(env.nA):
            for p,s_,r,d in env.P[s][a]:
                lhs[s][s_]-=gamma*pi[s][a]*p
                rhs[s]+=pi[s][a]*p*r
    v=np.linalg.solve(lhs,rhs)
    q=np.zeros((env.nS,env.nA))
    for s in range(env.nS-1):
        for a in range(env.nA):
            for p,s_,r,d in env.P[s][a]:
                q[s][a]+=p*(r+gamma*v[s_])
    return v,q

v,q=eval_bellman(env,random_pi)
print(f'random policy eval: \n v: \n {v} \n q: \n {q}')
v,q=eval_bellman(env,perfect_pi)
print(f'perfect policy eval: \n v: \n {v} \n q: \n {q}')

def optimize_bellman(env,gamma=1.):
    '''
    for stochastic environments like frozenlake, use gamma less than 1 e.g. 0.99
    '''
    tp=np.zeros((env.nS,env.nA,env.nS)) #transition probability
    rd=np.zeros((env.nS,env.nA)) #rewards
    for s in range(env.nS-1):
        for a in range(env.nA):
            for p,s_,r,d in env.P[s][a]:
                tp[s][a][s_]+=p
                rd[s][a]+=r*p
    c=np.ones(env.nS)
    A_ub=gamma*tp.reshape(-1,env.nS)-np.repeat(np.eye(env.nS),env.nA,axis=0)
    b_ub=-rd.reshape(-1)
    bounds=[(None,None),]*env.nS
    sln=scipy.optimize.linprog(c,A_ub,b_ub,bounds=bounds,method='interior-point')
    v=sln.x
    q=rd+gamma*np.dot(tp,v)
    pi=np.eye(env.nA)[q.argmax(axis=1)]
    return v,q,pi

v,q,bellman_pi=optimize_bellman(env)
print(f'bellman policy eval: \n v: \n {v} \n q: \n {q}')
test(env,bellman_pi)
