import sys
import os
s=os.path.join(os.getcwd(),os.pardir)
sys.path.append(os.path.abspath(s)+'/methods')
from td import DQNAgent
