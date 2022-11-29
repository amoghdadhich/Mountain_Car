import os
from typing import Type
import gym
import policy_gradient
from policy_gradient import Net as pg
import numpy as np
from random import choices, random
import torch

# Goal: Implement Policy Gradient to learn an optimal policy for mountain car
# Action space: 3 discrete actions
# Observation space: Continuous 2 dimensional space

def train_agent(env,policy: Type[policy_gradient.Net], optimizer: Type[torch.optim.SGD], gamma, NUM_episodes, NUM_batches):

    batch = 0
    obs = env.reset()         # Reset the env at the start of each episode

    while batch < NUM_batches:
        
        batch += 1
        batch_states = []
        batch_rewards = torch.FloatTensor([])
        batch_actions = []

        for i in range(NUM_episodes): # Run for a total batch size = NUM_episodes

            states = []
            rewards = []
            actions = []
            P_At = []
            states.append(obs)        # Initial state
            done = False
            
            while not done:
                action_prob = policy(obs).detach()
                action = choices([0,1,2], action_prob)[0]  # Choose an action from the state

                obs, reward, done, _ = env.step(action)    # Take a step using the action
                actions.append(action)              # A_t
                states.append(obs)                  # S_t+1
                rewards.append(reward)              # R_t+1
                P_At.append(action_prob[action])    # P(At)
            
            # rewards are discounted at each time step  and multiplied by gamma^i

            rewards_mod = pg.calculate_disc_reward(rewards, gamma)  
            batch_rewards = torch.cat((batch_rewards, rewards_mod), 0)
                                                      
            batch_states.extend(states[:-1])  # all states except terminal state are added to the batch
            batch_actions.extend(actions)
    
            print(f'Batch #{batch} Episode #{i} Return: {rewards_mod[0]}')  

            obs = env.reset()       # reset the environment at the end of the episode

    # Implement policy gradient for every batch of __ episodes   
    # Find the total discounted reward for the episode

        policy.update(optimizer= optimizer, states= batch_states, actions= batch_actions, rewards= batch_rewards, gamma= gamma)

    return policy 

if __name__ == '__main__':

    env = gym.make('MountainCar-v0')  
    NUM_batches = 1000 
    NUM_episodes = 10
    MAX_length = 9000   
    env._max_episode_steps = MAX_length
    
    gamma = 1
    
    policy = policy_gradient.Net()  # Initialize the policy
    policy = policy.float()         # Convert parameters to float datatype

    # Load the state dict of the model
    SAVE_PATH = '/mnt/c/Documents and Settings/Amogh/Documents/RBCDSAI/self learning/projects/Mountain Car_Policy Gradient'
    policy.load_state_dict(torch.load(os.path.join(SAVE_PATH,'mountain_car_policy_4.pth')))
    policy.train()

    # Initialize the optimizer
    learning_rate = 0.001
    optimizer = torch.optim.Adam(policy.parameters(), lr= learning_rate)

    # Run training
    policy = train_agent(env, policy, optimizer, gamma, NUM_episodes, NUM_batches)

    torch.save(policy.state_dict(),os.path.join(SAVE_PATH,'mountain_car_policy_5.pth'))
    env.close()
