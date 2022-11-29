import os
from typing import Type
import gym
import policy_gradient
from policy_gradient import Net as pg
import numpy as np
from random import choices, random
import torch
from time import sleep

# Demo Mountain car

def play_demo(env,policy: Type[policy_gradient.Net], NUM_episodes):

    obs = env.reset()         # Reset the env at the start of each episode

    for i in range(NUM_episodes): # Run for a total num of episodes of NUM_episodes

        done = False
        while not done:
            action_prob = policy(obs).detach()
            action = choices([0,1,2], action_prob)[0]  # Choose an action from the state

            obs, _, done, _ = env.step(action)    # Take a step using the action
            env.render()
            sleep(0.1)
        
        obs = env.reset()       # reset the environment at the end of the episode

if __name__ == '__main__':

    env = gym.make('MountainCar-v0')  
    NUM_episodes = 25
    MAX_length = 300   
    env._max_episode_steps = MAX_length
    
    gamma = 1
    
    policy = policy_gradient.Net()  # Initialize the policy
    policy = policy.float()         # Convert parameters to float datatype

    # Load the state dict of the model
    SAVE_PATH = '/mnt/c/Documents and Settings/Amogh/Documents/RBCDSAI/self learning/projects/Mountain Car_Policy Gradient'
    policy.load_state_dict(torch.load(os.path.join(SAVE_PATH,'mountain_car_policy_5.pth')))
    policy.eval()

    # Run demo
    policy = play_demo(env, policy, NUM_episodes)

    env.close()
