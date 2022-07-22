import gym
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
#state_fun(pos)




greedy_action = np.zeros((36,4),dtype=int)

epsilon = 1
sar_lambda = 0.9
gamma=0.9
action_values = np.zeros((36,4,3))
ep_list = np.arange(0,500,1)
step_list = np.zeros(500)
final_rew_list = []
for ep in range(0,25):

    print('episode :', end='')
    print(ep+1)
    env.reset()
    states = []
    uq_state = []
    elig_trace = np.zeros((36,4,3))
    rew_list = []
    #st_return = np.zeros((36,4,3))
    #lam_return = np.zeros((36,4,3))
    done = False
    tot_rew=0
    i = 0


    pos1 = env.agent_pos
    state_no1 = state_fun(pos1)
    dir1 = env.agent_dir
    # print('dir :')
    # print(dir)

    p = rd.random()
    # if state_no==15:
    #  action=7
    if p < epsilon:
        a1 = rd.randint(0, 2)
        action1 = a1
    else:
        action1 = greedy_action[state_no1][dir1]
    step = 0



    while not done:
        step+=1
        #env.render()
        if ep == 24:
            env.render()
        obs, rew, done, _ = env.step(action1)  # env.step(env.action_space.sample())
        rew_list.append(rew)
        tot_rew+=rew

        pos = env.agent_pos
        c_state_no = state_fun(pos)
        c_dir = env.agent_dir
        #print('dir :')
        #print(dir)

        p = rd.random()
        # if state_no==15:
        #  action=7
        if p < epsilon:
            a = rd.randint(0, 2)
            action = a
        else:
            action = greedy_action[c_state_no][c_dir]

        delta = rew + (0.9*action_values[c_state_no][c_dir][action]) - action_values[state_no1][dir1][action1]
       # print('cs,cd,ca',c_state_no,c_dir,action)
       # print('gamma*.. : ',gamma*action_values[c_state_no][c_dir][action])
        #print('action value',action_values[state_no1][dir1][action1])
        #print('delta :',delta)

        tp = (state_no1, dir1, action1)
        states.append(tp)
        if tp not in uq_state:
            uq_state.append(tp)



        for st in uq_state:
            u_st,u_d,u_s_act = st

            if u_st == state_no1 and u_d == dir1 and u_s_act == action1:
                if state_no1 == c_state_no and dir1 == c_dir:
                    elig_trace[u_st][u_d][u_s_act] = elig_trace[u_st][u_d][u_s_act]
                else:
                    elig_trace[u_st][u_d][u_s_act] = elig_trace[u_st][u_d][u_s_act] + 1
                #print('el tr : ',elig_trace[u_st][u_d][u_s_act])
            else:
                elig_trace[u_st][u_d][u_s_act] = 0.9*0.9*elig_trace[u_st][u_d][u_s_act]

        for st in states:
            st_n,d,s_act = st
            action_values[st_n][d][s_act] += 0.2*delta*elig_trace[st_n][d][s_act]
            #if action_values[st_n][d][s_act]>1000:
                #print(st_n,d,s_act,action_values[st_n][d][s_act])

        prev_act = action1
        prev_st = state_no1
        prev_dir = dir1

        action1=action
        state_no1 = c_state_no
        dir1 = c_dir


    #print('rew_list', rew_list)
    epsilon = epsilon/1.2
    final_rew_list.append(tot_rew)




    for i in range(0, 36):
        for j in range(0, 4):
            maxi = action_values[i][j][0]
            g_act = 0
            for k in range(0, 3):
                if action_values[i][j][k] >= maxi:
                    maxi = action_values[i][j][k]
                    g_act = k
            greedy_action[i][j] = g_act

    #print(action_values)
    #print(greedy_action)
    #print('el tr')
    #print(elig_trace)
    step_list[ep] = step

fig,ax = plt.subplots()
#ax.plot(ep_list,step_list)
ax.plot(np.arange(len(final_rew_list)),final_rew_list)
ax.set_xlabel("Episodes")
ax.set_ylabel("Rewards")
plt.show()
plt.show()


