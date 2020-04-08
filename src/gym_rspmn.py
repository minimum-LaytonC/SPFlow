import gym
import numpy as np
from numpy.random import randint
from copy import deepcopy
import argparse
from spn.algorithms.MEU import rmeu, best_next_decision
from spn.algorithms.SPMN import SPMN
from spn.algorithms.MPE import mpe

trials = 10000
steps = 10

slip = True
noisy = False

display = False

env = gym.make("FrozenLake-v0",is_slippery=slip)

file = open('data/frozen_lake/rspmn_407.pkle','rb')
import pickle
rspmn = pickle.load(file)

all_trails_reward = 0
for i in range(trials):
    total_reward = 0
    prev_state   = 0
    _ = env.reset()
    if display:
        print("\n\n\ngame: "+str(i))
        env.render()
        print("\t[t,action,observation,reward]:")
    done = False
    t=0
    while not done:
        if t == 0:
            input_data = np.array([[0]+[np.nan]*4])
        action = best_next_decision(rspmn, input_data, depth=100).reshape(1).astype(int)[0]
        #action = env.action_space.sample()
        new_state, reward, done, info = env.step(action)
        total_reward += reward
        if noisy:
            observation = new_state if randint(1,101) <= 85 else prev_state
            prev_state  = new_state
        else:
            observation = new_state
        input_data[0,1] = action
        input_data[0,2] = observation
        input_data[0,3] = reward
        belief_state = mpe(rspmn.spmn_structure, input_data).astype(int)[0,-1]
        input_data = np.array([[belief_state]+[np.nan]*4])
        if display:
            print("\t"+str([t,action,observation,reward,belief_state]))
            env.render()
        t+=1
    #if display:
    print("\ntotal reward for trial " + str(i+1) + ":\t"+str(total_reward))
    all_trails_reward += total_reward
    if i%100 == 0:
        print("% reached goal:\t" + str(total_reward/i))

average_reward = all_trails_reward / trials
