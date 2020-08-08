'''
policy & value iterations in a small grid world
based on sutton's RL book
'''


import numpy as np
np.set_printoptions(precision=2)


SEP='-'*80

GAMMA=1 # no decay

NUM_ITER=10

v=np.zeros((4,4))

valid=lambda r,c:0<=r<4 and 0<=c<4

get_nbr=lambda r,c:list((x,y) for x,y in ((r,c-1),(r-1,c),(r+1,c),(r,c+1)))

ACTIONS=['<','^','v','>']

def reward():
    return -1

def is_terminal(r,c):
    return True if r==c and (r == 0 or r == 3) else False

def render(i,v):
    print(f'iteration #{i}')
    for r in range(4):
        print(str(v[r]))
    print('\n\n')

print(SEP)
print('Policy Evaluation:')
print('\n\n')

for i in range(NUM_ITER):
    render(i,v) #visualize value function at the start of each iteration
    v1=np.zeros((4,4))
    for r in range(4):
        for c in range(4):
            if not is_terminal(r,c):
                nbrs=get_nbr(r,c)
                for x,y in nbrs:
                    if valid(x,y):
                        v1[r][c]+=0.25*(reward()+GAMMA*v[x][y]) #uniform policy
                    else: #moving out of grid gives thes same state
                        v1[r][c]+=0.25*(reward()+GAMMA*v[r][c])
    v=v1
render(NUM_ITER,v) #visualize value function at the end

print(SEP)
print('Value Iteration Policy:')

for r in range(4):
    for c in range(4):
        if not is_terminal(r,c):
            nbrs=get_nbr(r,c)
            best_value, best_action=-1e32,-1
            for i,(x,y) in enumerate(nbrs):
                if valid(x,y) and best_value<v[x][y]:
                    best_value, best_action=v[x][y],i
            print(ACTIONS[best_action],end=',')
        else:
            print('o',end=',')
    print()

print(SEP)
print('Policy Iteration:')

def policy_evaluation(policy,render_scene=False,gamma=0.99,theta=7e-1):
    '''
    given a policy, evaluate the value of each state
    value for a state is the one step rollout.
    :param gamma the decay rate
    :param theta tolerance
    '''
    value=np.zeros((4,4))
    num_iter=0
    while True:
        delta=0
        new_value=np.zeros((4,4))
        for r in range(4):
            for c in range(4):
                if not is_terminal(r,c):
                    nbrs=get_nbr(r,c)
                    for i,(x,y) in enumerate(nbrs):
                        if valid(x,y):
                            new_value[r][c]+=policy[r][c][i]*(reward()+gamma*value[x][y])
                        else:
                            new_value[r][c]+=policy[r][c][i]*(reward()+gamma*value[r][c])
                    delta=max(delta,abs(value[r][c]-new_value[r][c]))
        value=new_value
        if render_scene:
            render(num_iter,value)
        if delta<theta:
            break
        num_iter+=1
    return value

#for testing
#this is slightly off from sutton's book, don't know why.
#uniform_policy=np.ones((4,4,4))/4
#policy_evaluation(uniform_policy,render_scene=True,gamma=1.0,theta=7e-1)

def policy_improvement(v_star):
    '''
    returns the best policy under v_star
    '''
    p_star=np.zeros((4,4,4))
    for r in range(4):
        for c in range(4):
            nbrs=get_nbr(r,c)
            nbr_values=[]
            for x,y in nbrs:
                if valid(x,y):
                    nbr_values.append(v_star[x,y])
                else:
                    nbr_values.append(v_star[r,c])
            nbr_values=np.array(nbr_values)
            p_star[r][c]=np.eye(4)[np.argmax(nbr_values)]
    return p_star

def policy_iteration(num_policy_iter=10,render_scene=False,gamma=0.99,theta=7e-1):
    policy_stable=True
    p_star=np.ones((4,4,4))/4
    for i in range(num_policy_iter):
        v_star=policy_evaluation(p_star)
        p_star=policy_improvement(v_star)
    return p_star

p_star=policy_iteration()

for r in range(4):
    for c in range(4):
        if not is_terminal(r,c):
            best_action=np.argmax(p_star[r][c])
            print(ACTIONS[best_action],end=',')
        else:
            print('o',end=',')
    print()
