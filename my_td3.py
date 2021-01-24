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
      #Sample batch of transitions (s,s', a, r) from the memory
      batch_states, batch_next_states, batch_actions, batch_rewards,batch_dones = replay_buffer.sample(batch_size)
      state = torch.Tensor(batch_states).to(device)
      next_state = torch.Tensor(batch_next_states).to(device)
      action = torch.Tensor(batch_actions).to(device)
      reward = torch.Tensor(batch_rewards).to(device)
      done = torch.Tensor(batch_dones).to(device)
      #from next state s', the actor target plays the next action a'
      next_action = self.actorTarget(next_state)
      noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise).to(device)
      noise = noise.clamp(-noise_clip, noise_clip)
      next_action = (next_action + noise).clamp(-self.max_action, self.max_action)
      # two Critic targets take each the couple (s’, a’) as input and return two Q-values Qt1(s’,a’) and Qt2(s’,a’) as outputs
      target_Q1, target_Q2 = self.critic_target(next_state, next_action)
      
      #keep the minimum of these two Q-values: min(Qt1, Qt2)
      target_Q = torch.min(target_Q1, target_Q2)
      
      #get the final target of the two Critic models, which is: Qt = r + γ * min(Qt1, Qt2), where γ is the discount factor
      target_Q = reward + ((1 - done) * discount * target_Q).detach()
      
      #two Critic models take each the couple (s, a) as input and return two Q-values Q1(s,a) and Q2(s,a) as outputs
      current_Q1, current_Q2 = self.critic(state, action)
      
      # compute the loss coming from the two Critic models: Critic Loss = MSE_Loss(Q1(s,a), Qt) + MSE_Loss(Q2(s,a), Qt)
      critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
      
      #backpropagate this Critic loss and update the parameters of the two Critic models with a SGD optimizer
      self.critic_optimizer.zero_grad()
      critic_loss.backward()
      self.critic_optimizer.step()
      
      # Once every two iterations, we update our Actor model by performing gradient ascent on the output of the first Critic model
      if it % policy_freq == 0:
        actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
          target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
          target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

  def save(self, filename, directory):
    torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
    torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))
  
  # Making a load method to load a pre-trained model
  def load(self, filename, directory):
    self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
    self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))


def evaluate_policy(policy, eval_episodes=10):
  avg_reward = 0.
  for _ in range(eval_episodes):
    obs = env.reset()
    done = False
    while not done:
      action = policy.select_action(np.array(obs))
      obs, reward, done, _ = env.step(action)
      avg_reward += reward
  avg_reward /= eval_episodes
  print ("---------------------------------------")
  print ("Average Reward over the Evaluation Step: %f" % (avg_reward))
  print ("---------------------------------------")
  return avg_reward




#training
env_name = "AntBulletEnv-v0" # Name of a environment (set it to any Continous environment you want)
seed = 0 # Random seed number
start_timesteps = 1e4 # Number of iterations/timesteps before which the model randomly chooses an action, and after which it starts to use the policy network
eval_freq = 5e3 # How often the evaluation step is performed (after how many timesteps)
max_timesteps = 5e5 # Total number of iterations/timesteps
save_models = True # Boolean checker whether or not to save the pre-trained model
expl_noise = 0.1 # Exploration noise - STD value of exploration Gaussian noise
batch_size = 100 # Size of the batch
discount = 0.99 # Discount factor gamma, used in the calculation of the total discounted reward
tau = 0.005 # Target network update rate
policy_noise = 0.2 # STD of Gaussian noise added to the actions for the exploration purposes
noise_clip = 0.5 # Maximum value of the Gaussian noise added to the actions (policy)
policy_freq = 2 # Number of iterations to wait before the policy network (Actor model) is updated
file_name = "%s_%s_%s" % ("TD3", env_name, str(seed))
print ("---------------------------------------")
print ("Settings: %s" % (file_name))
print ("---------------------------------------")

if not os.path.exists("./results"):
  os.makedirs("./results")
if save_models and not os.path.exists("./pytorch_models"):
  os.makedirs("./pytorch_models")

env = gym.make(env_name)
env.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
policy = TD3(state_dim, action_dim, max_action)
replay_buffer = ReplayBuffer()
evaluations = [evaluate_policy(policy)]

def mkdir(base, name):
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path
work_dir = mkdir('exp', 'brs')
monitor_dir = mkdir(work_dir, 'monitor')
max_episode_steps = env._max_episode_steps
save_env_vid = False
if save_env_vid:
  env = wrappers.Monitor(env, monitor_dir, force = True)
  env.reset()

total_timesteps = 0
timesteps_since_eval = 0
episode_num = 0
done = True
t0 = time.time()

while total_timesteps < max_timesteps:
  
  if done:
    if total_timesteps != 0:
      print("Total Timesteps: {} Episode Num: {} Reward: {}".format(total_timesteps, episode_num, episode_reward))
      policy.train(replay_buffer, episode_timesteps, batch_size, discount, tau, policy_noise, noise_clip, policy_freq)

    if timesteps_since_eval >= eval_freq:
      timesteps_since_eval %= eval_freq
      evaluations.append(evaluate_policy(policy))
      policy.save(file_name, directory="./pytorch_models")
      np.save("./results/%s" % (file_name), evaluations)
    
    obs = env.reset()
    
    done = False
    
    episode_reward = 0
    episode_timesteps = 0
    episode_num += 1
  
  # Before 10000 timesteps, we play random actions
  if total_timesteps < start_timesteps:
    action = env.action_space.sample()
  else: # After 10000 timesteps, we switch to the model
    action = policy.select_action(np.array(obs))
    if expl_noise != 0:
      action = (action + np.random.normal(0, expl_noise, size=env.action_space.shape[0])).clip(env.action_space.low, env.action_space.high)
  
  new_obs, reward, done, _ = env.step(action)
  
  done_bool = 0 if episode_timesteps + 1 == env._max_episode_steps else float(done)
  
  episode_reward += reward
  
  replay_buffer.add((obs, new_obs, action, reward, done_bool))

  obs = new_obs
  episode_timesteps += 1
  total_timesteps += 1
  timesteps_since_eval += 1

evaluations.append(evaluate_policy(policy))
if save_models: policy.save("%s" % (file_name), directory="./pytorch_models")
np.save("./results/%s" % (file_name), evaluations)
  



