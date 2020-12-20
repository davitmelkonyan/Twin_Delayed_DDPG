import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import pybullet_envs
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import wrappers
from torch.autograd import Variable
from collections import deque

#STEP 1
class ReplayBuffer(object): #doesn't inherit from any clsss
  def __init__(self, max_size = 1e6):
    self.storage = [] #list of all transitions
    self.max_size = max_size
    self.ptr = 0 #start from first cell of memory

  def add(self, transition):
    if len(self.storage) == self.max_size:
      self.storage[int(self.ptr)] = transition
      self.ptr = (self.ptr + 1) % self.max_size
    else:
      self.storage.append(transition)

  def sample(self, batch_size):
    ind = np.random.randint(0, len(self.storage),size = batch_size)
    batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = [],[],[],[],[]
    for i in ind:
      state, next_state, action, reward, done = self.storage[i]
      batch_states.append(np.array(state,copy=False))
      batch_next_states.append(np.array(next_state,copy=False))
      batch_actions.append(np.array(action,copy=False))  
      batch_rewards.append(np.array(reward,copy=False))
      batch_dones.append(np.array(done, copy=False))
    return np.array(batch_states), np.array(batch_next_states), np.array(batch_actions), np.array(batch_rewards).reshape(-1,1), np.array(batch_dones).reshape(-1,1) #reshape convertes the array into horizontal 1D array

    
#STEP2
#Build NN for actor model and one for actor target
class Actor(nn.Module):#nn is from pytorch
  def __init__(self, state_dim, action_dim, max_action): #action_dim = # of actions at the same time
    super(Actor, self).__init__() #activating the inheritance
    self.layer_1 = nn.Linear(state_dim, 400)
    self.layer_2 = nn.Linear(400, 300)
    self.layer_3 = nn.Linear(300, action_dim)
    self.max_action = max_action

  def forward(self, x): #x is input state
    x = F.relu(self.layer_1(x)) #ReLu - Rectifier
    x = F.relu(self.layer_2(x))
    x = self.max_action * torch.tanh(self.layer_3(x))
    return x

class Critic(nn.Module):#nn is from pytorch
  def __init__(self, state_dim, action_dim): #action_dim = # of actions at the same time ,,,critics return q value so dont need max_action
    super(Critic, self).__init__() #activating the inheritance
    self.layer_1 = nn.Linear(state_dim + action_dim, 400)
    self.layer_2 = nn.Linear(400, 300)
    self.layer_3 = nn.Linear(300, 1) #1 for one q value
    #need two critics so below is second critic nns
    self.layer_4 = nn.Linear(state_dim + action_dim, 400)
    self.layer_5 = nn.Linear(400, 300)
    self.layer_6 = nn.Linear(300, 1) #1 for one q value

  def forward(self, x, u): #x is input state
    xu = torch.cat([x,u],1) #1 for vertical concatentation, for horizontal=>0
    #first critic forw. propag
    x1 = F.relu(self.layer_1(xu)) #ReLu - Rectifier
    x1 = F.relu(self.layer_2(x1))
    x1 = self.layer_3(x1)
    #second critic
    x2 = F.relu(self.layer_4(xu)) #ReLu - Rectifier
    x2 = F.relu(self.layer_5(x2))
    x2 = self.layer_6(x2)
    return x1, x2

  def Q1(self, x, u): #x is input state
    xu = torch.cat([x,u],1) #1 for vertical concatentation, for horizontal=>0
    x1 = F.relu(self.layer_1(xu)) #ReLu - Rectifier
    x1 = F.relu(self.layer_2(x1))
    x1 = self.layer_3(x1)
    return x1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #selecting cpu vs gpu

#class for training process
class TD3(object):
  def __init__(self, state_dim, action_dim, max_action):
    self.actor = Actor(state_dim, action_dim, max_action).to(device) #through gradient descent 
    self.actorTarget = Actor(state_dim, action_dim, max_action).to(device) #through polyak averaging
    self.actorTarget.load_state_dict(self.actor.state_dict())
    self.actorOptimizer = torch.optim.Adam(self.actor.parameters()) #stochastic gradient descent

    self.critic = Critic(state_dim, action_dim).to(device) #through gradient descent 
    self.actorTarget = Critic(state_dim, action_dim).to(device) #through polyak averaging
    self.criticTarget.load_state_dict(self.critic.state_dict())
    self.criticOptimizer = torch.optim.Adam(self.critic.parameters()) #stochastic gradient descent
    self.max_action = max_action

  def select_action(self,state):
    state = torch.Tensor(state.reshape(1,-1)).to(device)
    return self.actor(state).cpu().data.numpy().flatten()

  def train(self, replay_buffer, iterations, batch_size = 100, discount = 0.99, tau = 0.005, policy_noise = 0.2, noise_clip = 0.5, policy_freq = 2):
    for it in range(iterations):
      #Step 4 Sample batch of transitions (s,s', a, r) from the memory
      batch_states, batch_next_states, batch_actions, batch_rewards,batch_dones = replay_buffer.sample(batch_size)
      state = torch.Tensor(batch_states).to(device)
      next_state = torch.Tensor(batch_next_states).to(device)
      action = torch.Tensor(batch_actions).to(device)
      reward = torch.Tensor(batch_rewards).to(device)
      done = torch.Tensor(batch_dones).to(device)
      #step5: from next state s', the actor target plays the next action a'
      next_action = self.actorTarget(next_state)



