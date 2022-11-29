from typing import List
from typing import Type
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd
import numpy as np

class Net(nn.Module):

    def __init__(self) -> None:
        super(Net, self).__init__()

        # Connection from input to first hidden layer
        self.fc1 = nn.Linear(2,4)
        # Second hidden layer
        self.fc2 = nn.Linear(4,4)
        # Output layer
        self.fc3 = nn.Linear(4,3)

    def forward(self, x): # x is the input to the network

        x = torch.from_numpy(x)
        x = x.float()
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.softmax(self.fc3(x), dim=-1)  # Note that we use the softmax function to find the probability distribution across the action space

        return x

    def update(policy, optimizer: Type[torch.optim.SGD], states: List, actions: List, rewards: torch.FloatTensor, gamma: float) -> float:
        '''
        Implements REINFORCE algorithm
        Modifies a policy object by implementing gradient ascent 
        Returns the discounted return for the whole episode
        '''
                             
        # Get an output probabilities over all actions for all states
        states = np.array(states, dtype=float)
        probs = policy(states)

        # Compute the loss function
        actions = torch.LongTensor(actions)         # Actions will be the indices to pick the probabilities of the chosen action
        P_At = torch.gather(probs, 1, actions.unsqueeze(1)).squeeze()
        loss = torch.log(P_At) * rewards            # The rewards vector already contains the gamma^i terms

        # take NEGATIVE OF THE MEAN since we need to perform gradient ascent
        loss = -torch.mean(loss)

        # Compute gradient
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

    @staticmethod
    def calculate_disc_reward(rewards: List, gamma: float)->torch.FloatTensor:

        '''Input: list of rewards from an episode
        Returns a torch tensor of discounted rewards multiplied with gamma^t where t is the timestep from 
        which the return is calculated'''

        T = len(rewards)    # episode length
        disc_returns_ = []  # stores the discounted return from each time step
        gamma_vec = torch.FloatTensor([gamma**i for i in range(T)])

        episode_return = torch.sum(gamma_vec*torch.FloatTensor(rewards)) # return for the whole episode

        disc_returns_.append(episode_return)

        for i in range(1, T): # Loop through each step of the episode
                              
            # Calculate the discounted return from t = 1 incrementally 

            disc_return = disc_returns_[i-1]/gamma - rewards[i-1]
            disc_returns_.append(disc_return)  
        
        rewards_mod = torch.FloatTensor(disc_returns_) * gamma_vec

        return rewards_mod