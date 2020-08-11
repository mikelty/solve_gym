import tensorflow as tf
import gym
import pandas as pd
import numpy as np
from tqdm import tqdm

env_name='MountainCar-v0'
env=gym.make(env_name)
print(env.__dict__)

env.nS=env.observation_space.shape[0]
env.nA=env.action_space.n

eps=0.001
gamma=0.99
lr=0.01
batch_size=64
replayer_capacity=10_000
double_q=True #used in learn
train_reps=500
test_reps=100

class Replayer:
    def __init__(self,cap):#capacity
        self.cap=cap
        self.i=0
        self.count=0
        self.mem=pd.DataFrame(columns=('s','a','r','s_','d'))

    def store(self,*record):
        self.mem.loc[self.i]=record
        self.i=(self.i+1)%self.cap
        self.count=min(self.count+1,self.cap)

    def batch(self,size):
        indices=np.random.choice(self.count,size=size)
        return (np.stack(self.mem.loc[indices,col]) for col in self.mem.columns)

replayer=Replayer(replayer_capacity)

def build_net(env):
    net=tf.keras.Sequential()
    net.add(tf.keras.layers.Dense(64,activation='relu',input_shape=(env.nS,)))
    net.add(tf.keras.layers.Dense(64,activation='relu'))
    net.add(tf.keras.layers.Dense(env.nA,activation='relu'))
    opt=tf.optimizers.Adam(learning_rate=lr)
    net.compile(optimizer=opt,loss='mse')
    return net

behavior=build_net(env)
target=build_net(env)
target.set_weights(behavior.get_weights())

def learn(s,a,r,s_,d):
    replayer.store(s,a,r,s_,d)
    S,A,R,S_,D=replayer.batch(batch_size)
    if double_q:
        indices_Q_=behavior.predict(S_).argmax(axis=-1)
        Q_=target.predict(S_)
        stable_Q_=Q_[np.arange(indices_Q_.shape[0]),indices_Q_]
    else:
        stable_Q_=target.predict(S_).max(axis=-1)
    rollout=R+(1.-D)*gamma*stable_Q_
    Q=behavior.predict(S)
    Q[np.arange(rollout.shape[0]),A]=rollout
    behavior.fit(S,Q,verbose=0)
    if d:
        target.set_weights(behavior.get_weights())

def play_dqn(env,train=False):
    d,s,R=False,env.reset(),0
    while not d:
        if np.random.uniform()<eps:
            a=np.random.randint(env.nA)
        else:
            qs=behavior.predict(s[np.newaxis])
            a=np.argmax(qs)
        s_,r,d,i=env.step(a)
        if train:
            learn(s,a,r,s_,d)
        s,R=s_,R+r
    return R

#train
for _ in tqdm(range(train_reps),desc=f'train DQN (double_q: {double_q})',unit='eps'):
    play_dqn(env,train=True)
#test
test_R=0
for _ in tqdm(range(test_reps),desc=f'train DQN (double_q: {double_q})',unit='eps'):
    test_R+=play_dqn(env)
print('score: {test_R/test_reps}')
