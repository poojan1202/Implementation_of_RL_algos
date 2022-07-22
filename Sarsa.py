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
state_fun(pos)




greedy_action = np.zeros((36,4),dtype=int)
final_reward_list = []
epsilon = 1
sar_lambda = 0.9
action_values = np.zeros((36,4,3))
for ep in range(0,50):
    print('ep : ',ep+1)

    #print('episode :', end='')
    #print(ep+1)
    env.reset()
    states = []
    rew_list = []
    st_return = np.zeros((36,4,3))
    lam_return = np.zeros((36,4,3))
    done = False
    i = 0
    ep_action_values = np.zeros((36,4,3))
    tot_rew = 0
    while not done:
        #env.render()
        if ep==49:
            env.render()

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
        states.append(tp)

        obs, rew, done, _ = env.step(action)  # env.step(env.action_space.sample())
        rew_list.append(rew)
        tot_rew+=rew

    #print('rew_list', rew_list)
    epsilon = epsilon/1.1
    final_reward_list.append(tot_rew)

    for i in range(0,len(states)):
        st_n,d, s_act = states[i]
        for j in range(0,len(rew_list)):
            gamma = 1
            for k in range(0, j+1):
                st_return[st_n,d,s_act] = gamma*rew_list[k]
                gamma = gamma*0.9
            lam_return[st_n,d,s_act] += (1-sar_lambda)*(sar_lambda**(j+1)*st_return[st_n,d,s_act])


        action_values[st_n,d,s_act] += 0.1*(lam_return[st_n,d,s_act]-action_values[st_n,d,s_act])
    #print('lam return : ',lam_return)


    for i in range(0, 36):
        for j in range(0, 4):
            maxi = action_values[i][j][0]
            g_act = 0
            for k in range(0, 3):
                if action_values[i][j][k] >= maxi:
                    g_act = k
            greedy_action[i][j] = g_act


fig, ax = plt.subplots()
ax.plot(np.arange(len(final_reward_list)),final_reward_list)
ax.set_xlabel("Episodes")
ax.set_ylabel("Rewards")
plt.show()