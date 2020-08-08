import numpy as np
from tqdm import tqdm
import gym

env_name='Blackjack-v0'
env=gym.make(env_name)

print(env.__dict__)
env.nA=env.action_space.n

bespoke=np.zeros((32,11,2,2))
bespoke[20:,:,:,0]=1 #stick when point>=20
bespoke[:20,:,:,1]=1 #hit when point<20

def o2s(o):
    return (o[0],o[1],int(o[2]))

def play(env,pi):
    '''
    pi for policy
    '''
    d,o=False,env.reset()
    while not d:
        s=o2s(o)
        a=np.random.choice(env.nA,p=pi[s])
        o,r,d,i=env.step(a)
    return r

def test(env,pi,reps=1_000):
    R=0.
    for _ in tqdm(range(reps),desc='test',unit='eps'):
        R+=play(env,pi)
    return R/reps

print(test(env,bespoke))

def mc_soft(env,reps=500_000,eps=0.1):
    pi=np.ones((22,11,2,2))*.5
    q=np.zeros_like(pi)
    c=np.zeros_like(pi)
    for _ in tqdm(range(reps),desc=f'mc_soft (eps {eps})',unit='eps'):
        d,o,hist=False,env.reset(),[]
        while not d:
            s=o2s(o)
            #eps-greedy
            if np.random.uniform()<eps:
                a=np.random.randint(env.nA)
            else:
                a=np.random.choice(env.nA,p=pi[s])
            hist.append((s,a))
            o,r,d,i=env.step(a)
        for s,a in hist:
            c[s][a]+=1.
            q[s][a]+=(r-q[s][a])/c[s][a]
            pi[s]=0.
            pi[s][np.argmax(q[s])]=1.
    return pi

pi=mc_soft(env)
print(test(env,pi))

def mc_importance_resampling(env,reps=500_000):
    pi=np.ones((22,11,2,2))*.5 #behavioral policy is soft
    P=np.zeros_like(pi)
    P[:,:,:,0]=1 #target policy init to always sticks
    q=np.zeros_like(pi)
    c=np.zeros_like(pi)
    for _ in tqdm(range(reps),desc=f'mc_importance_resampling',unit='eps'):
        d,o,hist=False,env.reset(),[]
        while not d:
            s=o2s(o)
            a=np.random.choice(env.nA,p=pi[s])
            hist.append((s,a))
            o,r,d,i=env.step(a)
        rho=1.
        for s,a in reversed(hist):
            c[s][a]+=rho
            q[s][a]+=(rho/c[s][a])*(r-q[s][a])
            P[s]=0.
            P[s][np.argmax(q[s])]=1.
            if a!=np.argmax(q[s]):
                #all upstream values does not depend on this wrong branch, stop updating
                break
            rho/=pi[s][a]
    return P

#pi=mc_importance_resampling(env)
#print(test(env,pi))

def mc_exploring_start(env,reps=500_000):
    pi=np.ones((22,11,2,2))*.5
    q=np.zeros_like(pi)
    c=np.zeros_like(pi)
    for _ in tqdm(range(reps),desc=f'monte carlo exploring start',unit='eps'):
        env.player=[np.random.randint(1,11),np.random.randint(1,11)]
        env.dealer[0]=np.random.randint(1,11)
        d,o,hist=False,env._get_obs(),[]
        while not d:
            s=o2s(o)
            a=np.random.choice(env.nA,p=pi[s])
            hist.append((s,a))
            o,r,d,i=env.step(a)
        for s,a in hist:
            c[s][a]+=1.
            q[s][a]+=(r-q[s][a])/c[s][a]
            pi[s]=0.
            pi[s][np.argmax(q[s])]=1.
    return pi

pi=mc_exploring_start(env)
print(test(env,pi))
