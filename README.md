# solve_gym
* various methods and reinforcement learning libraries (stable baseline, my own) to solve various openAI gym environments
* note that stable baseline uses a different virtualenvironment. i used [autoswitch-venv](https://github.com/MichaelAquilina/zsh-autoswitch-virtualenv) for this.
* currently i implemented monte carlo (some borrowed [here](https://github.com/ZhiqingXiao/rl-book)), various TD (SARSA family, Q-learning family), and bellman equations (linear programming, policy & value iterations). each implementation comes with a test example.
* TODO
  * policy iteration can be shortened.
  * add various plotters, but might not be necessary because stable baseline can take care of that.
  * add more complex algorithms like AC family, PG family, etc. along with example solutions.
  * perhaps change the API a bit so that it looks more like stable baseline's.
