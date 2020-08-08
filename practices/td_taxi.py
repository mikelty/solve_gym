'''
parameter tweaking from frozenlake
'''
import numpy as np
np.set_printoptions(precision=2)
from tqdm import tqdm
import gym

sep='-'*80

env_name='Taxi-v3'

print(sep)
print(f'{env_name} info:')

env=gym.make(env_name)
env.nS=env.observation_space.n
env.nA=env.action_space.n
print(env.desc)
print(env.__dict__)
print(env.reset())
print(env.step(0))

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

def test(p,num_iter=1_000):
    total_return=0
    for i in tqdm(range(num_iter),desc='test',unit='eps'):
        total_return+=play(p)
    return total_return/num_iter

uniform_p=np.ones((env.nS,env.nA))/env.nA
print(test(uniform_p))

def td0_eval(p,num_iter=3_000,alpha=0.1,gamma=0.9):
    v=np.random.uniform(size=env.nS)
    v[-1]=0 #last state is terminal state
    for i in tqdm(range(num_iter),desc=f'td0_eval (alpha: {alpha}, gamma: {gamma})',unit='eps'):
        done,s=False,env.reset()
        while not done:
            a=np.random.choice(env.nA,p=p[s])
            s_,r,done,info=env.step(a)
            v[s]=(1-alpha)*v[s]+alpha*(r+(1-done)*gamma*v[s_])
        s=s_
    return v
'''
p=np.ones((env.nS,env.nA))/env.nA
v=td0_eval(p)
print(v)
'''
def SARSA(num_iter=3_000,alpha=0.1,gamma=0.9,eps=0.01,expected=True,lam=0.6):
    '''
    SARSA algo. for solving frozen lake.
    should work for taxi as well.
    expected SARSA is better than vanilla SARSA. of course, an eligibility trace would be even better (SARSA-lambda).
    if lam is not None, SARSA(lambda) is used (i.e. eligibility trace)
    TODO: dutch trace
    '''
    q=np.random.uniform(size=(env.nS,env.nA))
    q[-1]=np.zeros(env.nA) #terminal states of q-table is 0
    for i in tqdm(range(num_iter),desc=f'SARSA (alpha: {alpha}, gamma: {gamma}, epsilon: {eps}, expected: {expected}, lambda: {lam})',unit='eps'):
        done,s=False,env.reset()
        if lam is not None:
            z=np.zeros_like(q) #trace
        while not done:
            #eps-greedy choice of action
            if np.random.uniform()<eps:
                a=np.random.randint(env.nA)
            else:
                a=np.argmax(q[s])
            s_,r,done,info=env.step(a)
            #calculate the q-value for the next (state, action) pair
            if expected:
                q_=np.mean(q[s_])*eps+(1-eps)*q[s_,np.argmax(q[s_])]
            else:
                #eps-greedy again for next action
                if np.random.uniform()<eps:
                    a_=np.random.randint(env.nA)
                else:
                    a_=np.argmax(q[s_])
                q_=q[s_,a_]
            if lam is not None:
                z=gamma*lam*z #decay trace
                z[s,a]=1 #replace trace
                q_real=r+(1-done)*q_*gamma #unroll one step
                q+=alpha*z*(q_real-q[s,a]) #update weighted by trace
            else:
                q[s,a]=(1-alpha)*q[s,a]+alpha*(r+(1-done)*q_*gamma)
            s=s_
    #greedy policy
    return np.eye(env.nA)[np.argmax(q,axis=-1)],q

p,q=SARSA(alpha=0.2,expected=False,lam=None)
#print(f'p:\n{p}')
#print(f'q:\n{q}')
print(test(p))

p,q=SARSA(alpha=0.2,lam=None)
#print(f'p:\n{p}')
#print(f'q:\n{q}')
print(test(p))

p,q=SARSA(expected=False)
#print(f'p:\n{p}')
#print(f'q:\n{q}')
print(test(p))

p,q=SARSA()
#print(f'p:\n{p}')
#print(f'q:\n{q}')
print(test(p))

def q_learning(num_iter=30_000,alpha=0.1,gamma=0.9,eps=0.01,double_q=True):
    '''
    with this set of constants, double q-learning produces good result at around 100,000 episodes
    '''
    #initialize q-table(s)
    q=np.random.uniform(size=(env.nS,env.nA))
    q[-1]=np.zeros(env.nA)
    if double_q:
        q1=np.random.uniform(size=(env.nS,env.nA))
        q1[-1]=np.zeros(env.nA)
    for i in tqdm(range(num_iter),desc=f'q-learning (alpha: {alpha}, gamma: {gamma}, epsilon: {eps}, double_q: {double_q})',unit='eps'):
        s,done=env.reset(),False
        while not done:
            #eps-soft
            if np.random.uniform()>eps:
                if double_q:
                    a=np.argmax(q[s]+q1[s])
                else:
                    a=np.argmax(q[s])
            else:
                a=np.random.randint(env.nA)
            s_,r,done,info=env.step(a)
            if double_q:
                if np.random.randint(2):
                    q,q1=q1,q
                #this avoids maximization bias by decoupling sampling with retrieving the actual q-value
                q[s,a]=(1-alpha)*q[s,a]+alpha*(r+(1-done)*gamma*q1[s_,np.argmax(q[s_])])
            else:
                q[s,a]=(1-alpha)*q[s,a]+alpha*(r+(1-done)*gamma*max(q[s_]))
            s=s_
    #greedy policy
    if double_q:
        p=np.eye(env.nA)[np.argmax(q+q1,axis=-1)]
    else:
        p=np.eye(env.nA)[np.argmax(q,axis=-1)]
    return p,q

p,q=q_learning(double_q=False)
#print(f'p:\n{p}')
#print(f'q:\n{q}')
print(test(p))

p,q=q_learning()
#print(f'p:\n{p}')
#print(f'q:\n{q}')
print(test(p))
