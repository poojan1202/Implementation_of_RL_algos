import gym
from gym_minigrid.wrappers import *
import numpy as np
import random as rd
import matplotlib.pyplot as plt

env = gym.make('MiniGrid-Empty-8x8-v0')

env.reset()

done = False
pos = env.agent_pos
dir = env.agent_dir
act = env.action_space.sample
reward_list = []
#print(pos)
#print(dir)
#print(act)

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


greedy_action = np.zeros((36,4),dtype=int)

epsilon = 1
#states = []
action_values = np.zeros((36,4,3))
count = np.zeros((36,4,3),dtype=int)
for ep in range(0,75):
    print('episode : ',end='')
    print(ep+1)
    env.reset()
    states = []
    done = False
    gamma = 1
    i = 0
    ep_action_values = np.zeros((36,4,3))
    rew_tot=0
    while not done:
        #env.render()
        #if ep==70:
        #    env.render()

        pos = env.agent_pos
        state_no = state_fun(pos)
        dir = env.agent_dir
        #print('dir :')
        #print(dir)

        p = rd.random()
        # if state_no==15:
        #  action=7
        if p < epsilon:
            a = rd.randint(0, 2)
            action = a
        else:
            action = greedy_action[state_no][dir]

        tp = (state_no, dir, action)
        if tp not in states:

            states.append(tp)

        obs, rew, done, _ = env.step(action)  # env.step(env.action_space.sample())
        rew_tot+=rew
        i += 1
        for st in states:
            st_n, d, s_act = st
            #print(st_n,d,s_act)
            ep_action_values[st_n][d][s_act] += gamma * rew
            gamma = gamma / 0.9
        gamma = 0.9 ** i
    reward_list.append(rew_tot)
    epsilon = epsilon/1.1

    for i in range(0,36):
        for j in range(0,4):
            for k in range(0,3):
                if (i,j,k) in states:
                    count[i][j][k] +=1
                    action_values[i][j][k] += (1/count[i][j][k])*(ep_action_values[i][j][k]-action_values[i][j][k])


    for i in range(0, 36):
        for j in range(0, 4):
            maxi = action_values[i][j][0]
            g_act = 0
            for k in range(0, 3):
                if action_values[i][j][k] >= maxi:
                    g_act = k
            greedy_action[i][j] = g_act

        # current_st = states[state_fun(pos)][dir]
        # print(pos)
        # print(dir)
        # print(rew)

    #print(action_values)
    #print(greedy_action)



fig, ax = plt.subplots()
ax.plot(np.arange(len(reward_list)),reward_list)
ax.set_xlabel("Episodes")
ax.set_ylabel("Rewards")
plt.show()
### 50 episodes to converge to Vpi*