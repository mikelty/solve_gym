'''
parameter tweaking from frozenlake
this doesn't work. a lengthy expected-SARSA-lambda doesn't work better than a very simple bespoke policy.
monte carlo search is needed for this.
'''
import gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from functools import reduce

env=gym.make('Blackjack-v0')
print(env.__dict__)
obs_shape=[env.observation_space[i].n for i in range(len(env.observation_space))]
env.nA=env.action_space.n
env.nS=obs_shape[0]*obs_shape[1]*obs_shape[2]

policy_shape=(32,11,2,2)
#this policy is from sutton's book
bespoke_policy=np.zeros(policy_shape)
bespoke_policy[:,:,:,1]=1 #always twist
#except for at and over 20
bespoke_policy[20:,:,:,0]=1
bespoke_policy[20:,:,:,1]=0

def obs2state(obs):
    '''
    obs[2] is True if the agent has a usable Ace (or something like that), this function converts obs[2] to an int such that True=1, False=0 and returns the numerically encoded state
    '''
    return (obs[0],obs[1],1 if obs[2] else 0)

def play_once(policy):
    '''
    given a policy, plays the game once
    '''
    done, obs, reward = False, env.reset(), 0
    while not done:
        state=obs2state(obs)
        action=np.random.choice(env.action_space.n,p=policy[state])
        new_obs, reward, done, info=env.step(action)
        obs=new_obs
    return reward

play_once(bespoke_policy)

def test_policy(policy,reps=1000):
    '''
    repetitive tests on policy are needed for better accuracy of environments such as blackjack
    '''
    return sum(play_once(policy) for _ in range(reps)) / reps

score=test_policy(bespoke_policy)
print(f'test score of bespoke policy: {score}')

def plot_predictions(value,reps):
    '''
    plots two charts, one with ace and one without for the agent's value given a value function
    '''
    value_no_ace, value_ace=np.squeeze(value[:,:,0]), np.squeeze(value[:,:,1])
    fig=plt.figure(figsize=(20,10))
    fig.suptitle(f'monte carlo eval after {reps} episodes')
    plt.subplot(1,2,1)
    plt.imshow(value_no_ace)
    plt.title('without ace')
    plt.xlabel('dealer\'s card')
    plt.ylabel('my total')
    plt.subplot(1,2,2)
    plt.imshow(value_ace)
    plt.title('with ace')
    plt.xlabel('dealer\'s card')
    plt.ylabel('my total')
    plt.show()


def SARSA(num_iter=300_000,alpha=0.01,gamma=0.99,eps=0.1,expected=True,lam=0.6):
    '''
    SARSA algo. for solving frozen lake.
    should work for taxi as well.
    expected SARSA is better than vanilla SARSA. of course, an eligibility trace would be even better (SARSA-lambda).
    if lam is not None, SARSA(lambda) is used (i.e. eligibility trace)
    TODO: dutch trace
    '''
    q=np.random.uniform(size=(*obs_shape,env.nA))
    q[-1]=np.zeros(env.nA) #terminal states of q-table is 0
    for i in tqdm(range(num_iter),desc=f'SARSA (alpha: {alpha}, gamma: {gamma}, epsilon: {eps}, expected: {expected}, lambda: {lam})',unit='eps'):
        done,s=False,obs2state(env.reset())
        if lam is not None:
            z=np.zeros_like(q) #trace
        while not done:
            #eps-greedy choice of action
            if np.random.uniform()<eps:
                a=np.random.randint(env.nA)
            else:
                a=np.argmax(q[s])
            s_,r,done,info=env.step(a)
            s_=obs2state(s_)
            #calculate the q-value for the next (state, action) pair
            if expected:
                q_=np.mean(q[s_])*eps+(1-eps)*q[s_][np.argmax(q[s_])]
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
                q+=alpha*z*(q_real-q[s][a]) #update weighted by trace
            else:
                q[s,a]=(1-alpha)*q[s,a]+alpha*(r+(1-done)*q_*gamma)
            s=s_
    #greedy policy
    return np.eye(env.nA)[np.argmax(q,axis=-1)],q

p,q=SARSA(alpha=0.2)
#print(f'p:\n{p}')
#print(f'q:\n{q}')
print(test_policy(p))
