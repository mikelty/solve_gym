# solve_gym
* various methods and reinforcement learning libraries (stable baseline, my own) to solve various openAI gym environments
* note that stable baseline uses a different virtualenvironment. i used [autoswitch-venv](https://github.com/MichaelAquilina/zsh-autoswitch-virtualenv) for this.
* currently i implemented monte carlo (some borrowed [here](https://github.com/ZhiqingXiao/rl-book)), various TD (SARSA family, Q-learning family), and bellman equations (linear programming, policy & value iterations). each implementation comes with a test example.
* TODO
  * add rounds parameter for policy & value iteration
  * add prioritize replay to DQN
  * add automated tests
  * add more complex algorithms like AC family, PG family, etc. along with example solutions.
  * perhaps change the API a bit so that it looks more like stable baseline's.
  * there's a good NEAT library in Python and perhaps i'd reproduce it for a lot of the examples. (parameter tuning can be quite time consuming though)
  * another general purpose algorithm like NEAT is MuZero, which is very effective at atari games and board games. so perhaps implement that too. (need a beefier machine)
