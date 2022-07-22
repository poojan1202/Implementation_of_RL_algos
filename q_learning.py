from gym_minigrid.wrappers import *
import numpy as np
import random as rd
import matplotlib.pyplot as plt


env = gym.make('MiniGrid-Empty-8x8-v0')

env.reset()

done = False
#pos = env.agent_pos
#dir = env.agent_dir
#act = env.action_space.sample

def state_fun(pos):
    state = 0
    #print('POS : ')
    #print(pos)
    a,b = pos
    for j in range(0,6):
        for i in range(0,6):
            if a == i+1 and b == j+1:
                #print('STATE : ')
                #print(state)
                return state
            else:
                state += 1

#policy pi


def q_max_action(act_vals):

    i=-1
    maximum = act_vals[0]
    idx = 0
    for values in act_vals:
        i += 1
        if values>=maximum:
            maximum = values
            idx = i
    return idx


greedy_action = np.zeros((36,4),dtype=int)

epsilon = 1

gamma = 0.9

ep_list = np.arange(0,500,1)
step_list = np.zeros(500)
action_values = np.zeros((36,4,3))
#states = []
#policy_pi = np.zeros((36,4))
#count = np.zeros((36,4,3),dtype=int)
for ep in range(0,500):
    step = 0
    print('episode : ',end='')
    print(ep+1)
    env.reset()
    #states = []
    done = False
    i = 0
    ep_action_values = np.zeros((36,4,3))

    #initial position
    pos1 = env.agent_pos
    state_no1 = state_fun(pos1)
    dir1 = env.agent_dir


    while not done:
        step+=1
        #env.render()
        if ep==499:
            env.render()


        #print('dir :')
        #print(dir)

        p = rd.random()
        # if state_no==15:
        #  action=7
        if p < epsilon:
            a = rd.randint(0, 2)
            action = a
        else:
            action = greedy_action[state_no1][dir1]

        #tp = (state_no1, dir1, action)
        #states.append(tp)
        #print('action',action)
        obs, rew, done, _ = env.step(action)  # env.step(env.action_space.sample())

        #next state of agent
        pos2 = env.agent_pos
        #print('pos2',pos2)
        state_no2 = state_fun(pos2)
        dir2 = env.agent_dir

        amax = q_max_action(action_values[state_no2][dir2])
        #print('st d',state_no2,dir2)

        action_values[state_no1][dir1][action] += 0.2*(rew + gamma*action_values[state_no2][dir2][amax] - action_values[state_no1][dir1][action])
        #print('stateno1,dir1 = ',end='')
        #print(state_no1,dir1)
        #print('stateno2,dir2 = ',end='')
        #print(state_no2,dir2)
        state_no1 = state_no2
        dir1 = dir2

    step_list[ep] = step

    epsilon /= 1.1

    #print(action_values)
    for i in range(0, 36):
        for j in range(0, 4):
            maxi = action_values[i][j][0]
            g_act = 0
            for k in range(0, 3):
                if action_values[i][j][k] >= maxi:
                    maxi = action_values[i][j][k]
                    g_act = k
            greedy_action[i][j] = g_act

fig,ax = plt.subplots()
ax.plot(ep_list,step_list)
plt.show()