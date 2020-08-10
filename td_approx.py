'''
too much to test here...
'''
import numpy as np
np.set_printoptions(precision=2)
import tensorflow as tf
import pandas as pd
from tqdm import tqdm
import gym

sep='-'*80

env_name='MountainCar-v0'
#env_name='FrozenLake-v0'

print(sep)
print(f'{env_name} info:')

env=gym.make(env_name)
env.nA=env.action_space.n
print(env.__dict__)
print(env.reset())
print(env.step(0))

print(sep)
print(f'play once & test:')

def go_right():
    '''
    always go right
    '''
    done,ret,s=False,0,env.reset()
    while not done:
        env.render()
        a=2 #right
        s,r,done,info=env.step(a)
        ret+=r
    env.close()
    return ret

#print(go_right()) #for demo only

class Replayer:
    def __init__(self,cap):#capacity
        self.cap=cap
        self.i=0#index of new entry
        self.count=0#size
        self.storage=pd.DataFrame(columns=['s','a','r','s_','done'])

    def add(self,*args):
        self.storage.loc[self.i]=args
        self.i=(self.i+1)%self.cap
        self.count=min(self.cap,self.count+1)

    def sample(self,batch_size):
        indices=np.random.choice(self.count,size=batch_size)
        return (np.stack(self.storage.loc[indices,field])\
                for field in self.storage.columns)

class DQN:
    def __init__(self,env,other_args,gamma=0.99,eps=1e-3,batch_size=64,replayer_cap=10_000,double_q=False):
        self.env=env
        num_in=env.observation_space.shape[0]
        num_out=env.action_space.n
        self.behavior=self._build_net(num_in,num_out,**other_args)
        self.target=self._build_net(num_in,num_out,**other_args)
        self.target.set_weights(self.behavior.get_weights())
        self.gamma=gamma
        self.eps=eps
        self.batch_size=batch_size
        self.replayer=Replayer(replayer_cap)
        self.double_q=double_q

    def _build_net(self,num_in,num_out,num_hiddens,\
                   activation=tf.nn.relu,out_activation=None,
                   lr=0.01):
        net=tf.keras.Sequential()
        for i,h in enumerate(num_hiddens):
            kw={} if i else {'input_shape':(num_in,)}
            net.add(tf.keras.layers.Dense(h,activation=activation,**kw))
        net.add(tf.keras.layers.Dense(num_out,activation=out_activation))
        net.compile(loss='mse',
                    optimizer=tf.optimizers.Adam(learning_rate=lr))
        return net

    def learn(self,s,a,r,s_,done):
        '''
        this is going to be similar to q-learning.
        a dqn uses two networks to imitate one q-table, which can be viewed as a function that has a state as an input and a bunch of actions values as outputs.
        although the index arithematic and python shorthand can be confusing for neural nets solutions in RL, it's really nothing different from the TD algorithms. for this one, it's really not that different from the q-learning algorithm in eseence.
        for the double-q case, the behavior net is used to index which q values from the target net to replace the old prediction
        '''
        #housekeeping
        self.replayer.add(s,a,r,s_,done)
        S,A,R,S_,DONE=self.replayer.sample(self.batch_size)
        if self.double_q:
            q1_=self.target.predict(S_)
            max_q1_idx=q1_.argmax(axis=-1)
            q_=self.target.predict(S_)
            max_q=q_[np.arange(self.batch_size),max_q1_idx]
        else:
            q_=self.target.predict(S_)
            max_q=q_.max(axis=-1)
        unrolled=R+(1.-DONE)+self.gamma*max_q
        AV=self.behavior.predict(S)
        AV[np.arange(self.batch_size),A]=unrolled
        self.behavior.fit(S,AV,verbose=0)

    def decide(self,s):
        if np.random.uniform()<self.eps:
            return np.random.randint(self.env.nA)
        else:
            return np.argmax(self.behavior.predict(s[np.newaxis]))

def dqn_wrapper(agent,train=True):
    done,s,R=False,env.reset(),0
    while not done:
        a=agent.decide(s)
        s_,r,done,info=env.step(a)
        if train:
            agent.learn(s,a,r,s_,done)
        s,R=s_,R+r
    return R

def test_dqn(double_q=False):
    other_args={'lr':1e-3,'num_hiddens':[64,64]}
    agent,num_iter=DQN(env,other_args),500
    for i in tqdm(range(num_iter),desc='training',unit='eps'):
        dqn_wrapper(agent)
    num_iter,total_test_reward=100,0.
    for i in tqdm(range(num_iter),desc='testing',unit='eps'):
        total_test_reward+=dqn_wrapper(agent,train=False)
    print(f'avg. test reward: {total_test_reward/num_iter}')

test_dqn()
test_dqn(double=True)
