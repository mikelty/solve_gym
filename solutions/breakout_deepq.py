import sys
import os
s=os.path.join(os.getcwd(),os.pardir)
sys.path.append(os.path.abspath(s)+'/methods')
from td import DQNAgent
from myutils import *

name='Breakout-v0'

env=bootstrap(name)

net_kwargs = {'hidden_sizes' : [64, 64], 'learning_rate' : 0.001}
agent = DQNAgent(env, net_kwargs=net_kwargs)
dqn_agent=solve(agent,env,500,100)
