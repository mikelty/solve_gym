from tqdm import tqdm
from time import perf_counter
from myutils import *
import gym
import numpy as np

class MonteCarlo(RLAgent):
    def __init__(self,env,rep=500_000,exploratory=True,eps=0.):
        #note tolerance can be better divided into two tolerances
        super().__init__(env=env)
        self.rep=rep
        self.exploratory=exploratory
        self.eps=eps
        self.policy=None

    def ob2state(self,observation):
        return min(21,observation[0]), observation[1], int(observation[2])

    def decide(self,state,train=False):
        #there are some soft policies so when training we want this
        if train:
            return np.random.choice(self.env.nA,p=self.policy[state])
        else:
            return np.argmax(self.policy[state])

    def learn(self):
        '''
        monte carlo agent
        self play for <episode_num> times
        can use exploratory play
        can also use <eps>ilon soft policy, otherwise set <eps>ilon to 0.
        todo: add automated tests
        todo: push this to a git repo
        '''
        start=perf_counter()
        #compress states
        if self.eps>0:
            self.policy = np.ones((22, 11, 2, 2))*0.5 #soft policy
        else:
            self.policy = np.zeros((22, 11, 2, 2))
            self.policy[:, :, :, 1] = 1.
        q = np.zeros_like(self.policy)
        c = np.zeros_like(self.policy)
        for i in tqdm(range(self.rep)):
            if self.exploratory:
                state = (np.random.randint(12, 22),
                         np.random.randint(1, 11),
                         np.random.randint(2))
                action = np.random.randint(2)
                env.reset()
                if state[2]: #ace treated as 11
                    env.player = [1, state[0] - 11]
                else: #ace not treated as 11
                    if state[0] == 21:
                        env.player = [10, 9, 2]
                    else:
                        env.player = [10, state[0] - 10]
                env.dealer[0] = state[1]
            else:
                state=self.ob2state(env.reset())
                action=np.random.choice(self.env.nA,p=self.policy[state])
            state_actions, total_reward=[],0
            while True:
                state_actions.append((state,action))
                next_observation,reward,done,info=env.step(action)
                if done:
                    break
                state=self.ob2state(next_observation)
                action=np.random.choice(self.env.nA,p=self.policy[state])
                total_reward+=reward
            for state,action in state_actions:
                c[state][action] += 1.
                #the more we explore a branch the more certain we are about it so the less we update
                q[state][action] += (total_reward - q[state][action]) / c[state][action]
                a = np.argmax(q[state])
                self.policy[state] = self.eps / len(self.policy[state])
                self.policy[state][a] += (1. - self.eps)

    def evaluate(self, sample_policy, rep=500_000):
        '''
        Calculate the action value of a policy using monte carlo methods by playing it out many times.
        '''
        print(f"evaluating")
        q=np.zeros_like(sample_policy)
        c=np.zeros_like(sample_policy)
        for i in tqdm(range(rep)):
            state_actions, observation, done, total_reward = [], env.reset(), False, 0
            while not done:
                state=self.ob2state(observation)
                action=np.random.choice(self.env.nA,p=sample_policy[state])
                state_actions.append((state,action))
                observation,reward,done,info=env.step(action)
                total_reward+=reward
            for state,action in state_actions:
                c[state][action] += 1.
                q[state][action] += (total_reward - q[state][action]) / c[state][action]
        return q

    def decide(self,observation,train=False):
        state=self.ob2state(observation)
        return np.random.choice(self.env.nA,p=self.policy[state])

class MonteCarloImportanceResampling(MonteCarlo):
    def __init__(self,env,rep=500_000):
        super().__init__(env=env,rep=rep)

    def evaluate(self, policy, behavior_policy,rep):
        print(f"evaluating")
        q = np.zeros_like(policy)
        c = np.zeros_like(policy)
        for i in tqdm(range(rep)):
            # 用行为策略玩一回合
            state_actions = []
            observation = self.env.reset()
            while True:
                state = self.ob2state(observation)
                action = np.random.choice(self.env.action_space.n,
                        p=behavior_policy[state])
                state_actions.append((state, action))
                observation, reward, done, _ = self.env.step(action)
                if done:
                    break # 玩好了
            g = reward # 回报
            rho = 1. # 重要性采样比率
            for state, action in reversed(state_actions):
                c[state][action] += rho
                q[state][action] += (rho / c[state][action] * (g - q[state][action]))
                rho *= (policy[state][action] / behavior_policy[state][action])
                if rho == 0:
                    break # 提前终止
        return q

    def learn(self):
        policy = np.zeros((22, 11, 2, 2))
        policy[:, :, :, 0] = 1.
        behavior_policy = np.ones_like(policy) * 0.5 # 柔性策略
        q = np.zeros_like(policy)
        c = np.zeros_like(policy)
        for i in tqdm(range(self.rep)):
            # 用行为策略玩一回合
            state_actions = []
            observation = self.env.reset()
            while True:
                state = self.ob2state(observation)
                action = np.random.choice(self.env.action_space.n,
                        p=behavior_policy[state])
                state_actions.append((state, action))
                observation, reward, done, _ = self.env.step(action)
                if done:
                    break # 玩好了
            g = reward # 回报
            rho = 1. # 重要性采样比率
            for state, action in reversed(state_actions):
                c[state][action] += rho
                q[state][action] += (rho / c[state][action] * (g - q[state][action]))
                # 策略改进
                a = q[state].argmax()
                policy[state] = 0.
                policy[state][a] = 1.
                if a != action: # 提前终止
                    break
                rho /= behavior_policy[state][action]
        self.policy=policy

if __name__=='__main__':

    np.set_printoptions(precision=3)

    name="Blackjack-v0"
    env=bootstrap(name)

    #exploratory monte carlo
    agent=MonteCarlo(env)
    monte_carlo_agent=solve(agent,env,0,0,10_000)

    #not exploratory monte carlo with soft policy (epsilon=0.1)
    agent=MonteCarlo(env,exploratory=False,eps=0.1)
    monte_carlo_agent=solve(agent,env,0,0,10_000)

    bespoke_policy=np.zeros((22,11,2,2))
    bespoke_policy[20:,:,:,0]=1#stop when my points are at or above 20
    bespoke_policy[:20,:,:,1]=1#hit when my points are below 20
    evaluation=agent.eval_policy(bespoke_policy)
    print(f"evaluate bespoke policy by monte carlo:\n{evaluation}")

    agent=MonteCarloImportanceResampling(env)
    monte_carlo_importance_resampling_agent=solve(agent,env,0,0,10_000)

    behavior_policy = np.ones_like(bespoke_policy) * 0.5
    evaluation=agent.evaluate(bespoke_policy, behavior_policy,500_000)
    print(f"evaluate bespoke policy by monte carlo with importance resampling:\n{evaluation}")
