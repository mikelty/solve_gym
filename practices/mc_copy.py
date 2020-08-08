import numpy as np
from tqdm import tqdm
import gym

env_name='Blackjack-v0'
env=gym.make(env_name)
print(env.__dict__)

env.nA=env.action_space.n

bespoke=np.zeros((22,11,2,2))
bespoke[:20,:,:,0]=1
bespoke[20:,:,:,1]=1

def o2s(o):
    return (o[0],o[1],int(o[2]))

def test(env,pi,pi_name,reps=1000):
    total_R=0
    for _ in tqdm(range(reps),desc='test',unit='eps'):
        d,o=False,env.reset()
        while not d:
            s=o2s(o)
            a=np.random.choice(env.nA,p=pi[s])
            o,r,d,i=env.step(a)
            total_R+=r
    print(f'{pi_name} policy score: {total_R/reps}')
    return total_R/reps

test(env,bespoke,'bespoke')

def mc_soft(env,reps=500_000,eps=0.1):
    pi=np.ones((22,11,2,2))*0.5
    q=np.zeros_like(pi)
    c=np.zeros_like(pi)
    for _ in tqdm(range(reps),desc=f'mc soft (eps: {eps})',unit='eps'):
        hist,d,o=[],False,env.reset()
        while not d:
            s=o2s(o)
            if np.random.uniform()<eps:
                a=np.random.randint(env.nA)
            else:
                a=np.random.choice(env.nA,p=pi[s])
            o,r,d,i=env.step(a)
            hist.append((s,a))
        for s,a in hist:
            c[s][a]+=1.
            q[s][a]+=(r-q[s][a])/c[s][a]
            pi[s]=0.
            pi[s][q[s].argmax()]=1.
    return pi

#pi=mc_soft(env)
#test(env,pi,'soft')

def mc_exploring_start(env,reps=500_000):
    pi=np.ones((22,11,2,2))*0.5
    q=np.zeros_like(pi)
    c=np.zeros_like(pi)
    for _ in tqdm(range(reps),desc=f'mc soft',unit='eps'):
        env.player=[np.random.randint(1,11),np.random.randint(1,11)]
        env.dealer=[np.random.randint(1,11)]
        hist,d,o=[],False,env._get_obs()
        while not d:
            s=o2s(o)
            a=np.random.choice(env.nA,p=pi[s])
            o,r,d,i=env.step(a)
            hist.append((s,a))
        for s,a in hist:
            c[s][a]+=1.
            q[s][a]+=(r-q[s][a])/c[s][a]
            pi[s]=0.
            pi[s][q[s].argmax()]=1.
    return pi

#pi=mc_exploring_start(env)
#test(env,pi,'exploring start')

def mc_importance_sampling(env,reps=500_000):
    pi=np.ones((22,11,2,2))*0.5
    P=np.zeros_like(pi)
    q=np.zeros_like(pi)
    c=np.zeros_like(pi)
    for _ in tqdm(range(reps),desc=f'mc importance sampling',unit='eps'):
        hist,d,o=[],False,env.reset()
        while not d:
            s=o2s(o)
            a=np.random.choice(env.nA,p=pi[s])
            o,r,d,i=env.step(a)
            hist.append((s,a))
        rho=1.
        for s,a in hist:
            c[s][a]+=rho
            q[s][a]+=rho*(r-q[s][a])/c[s][a]
            rho/=pi[s][a]
            pi[s]=0.
            pi[s][q[s].argmax()]=1.
            if a!=q[s].argmax():
                break #so that anything upstream doesn't get updated according to another branch
    return P

pi=mc_exploring_start(env)
test(env,pi,'importance sampling')
