import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import statistics
from itertools import count
import numpy as np
from game import SnakeGameAI
from helper import plot
from torch.distributions import Categorical


class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.policy_head = nn.Linear(hidden_size, output_size) #Logits for actions

    def forward(self, x):
        x = F.relu(self.linear1(x))
        logits = self.policy_head(x)
        return logits

    def save(self, file_name='Actormodel.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.value_head = nn.Linear(hidden_size, output_size) #Q(s,a)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        value = self.value_head(x)
        return value

    def save(self, file_name='Criticmodel.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

#Reimplement for actor critic
class ActorCritic:
    def __init__(self, mdp, actor, critic):
        self.mdp = mdp #Agent class
        self.actor = actor #Policy based actor
        self.critic = critic #Q value based critic
        self.actor_optimizer = optim.Adam(actor.parameters(), lr=0.001)
        self.critic_optimizer = optim.Adam(critic.parameters(), lr=0.001)

    def execute(self, episodes=10000, max_episode_length=float("inf"), game=None):
        if(game is None):
            game = SnakeGameAI()
        episode_rewards = []
        #Plot
        plot_reward = []
        plot_mean_reward = []
        total_reward = 0
        for episode in range(episodes):
            actions = []
            states = []
            rewards = []
            deltas = []
            log_probs = []
            dones = []

            #Reset the game every episode
            game.reset()
            state = self.mdp.get_state(game) #Does it need to get initial state or just the current state??
            action_idx, final_move, log_prob = self.mdp.get_action_idx(state)
            episode_reward = 0.0
            for step in count():
                reward, done, score = game.play_step(final_move) #Executes next step to get new state and action
                next_state = self.mdp.get_state(game)
                next_action_idx, next_final_move, next_log_prob = self.mdp.get_action_idx(next_state)
                #delta = self.get_TD(reward, state, action_idx, next_state, next_action_idx, self.critic, done)

                #Store information
                states.append(state)
                actions.append(action_idx)
                rewards.append(reward)
                dones.append(done)
                #deltas.append(delta)
                log_probs.append(log_prob)

                state = next_state
                final_move = next_final_move
                action_idx = next_action_idx
                log_prob = next_log_prob

                #Logging discounted return from beginning of episode
                episode_reward += reward * (0.95 ** step)


                if done or step >= max_episode_length:
                    break
            
            self.update_critic(states, actions, rewards, dones)
            self.update_actor(states, actions, rewards, dones, log_probs)

            episode_rewards.append(episode_reward)

            
            plot_reward.append(episode_reward)
            total_reward += episode_reward
            mean_reward = total_reward / (episode + 1)
            plot_mean_reward.append(mean_reward)

            plot(plot_reward, plot_mean_reward, save_every=100, episode_idx=episode)
        return episode_rewards
    
    #Calculate the tempral difference
    def get_TD(self, reward, state, action, next_state, next_action, critic, done):
        gamma = 0.95
        td = reward + gamma * self.mdp.get_value(next_state, next_action) * (1 - done) - self.mdp.get_value(state, action)
        return td

    #Updates the critic based off episode list of states, actions, and deltas
    def update_critic(self, states, actions, rewards, dones):
        # Convert to tensors
        states_t = torch.tensor(np.array(states), dtype=torch.float)
        actions_t = torch.tensor(actions, dtype=torch.long)
        rewards_t = torch.tensor(rewards, dtype=torch.float)
        dones_t = torch.tensor(dones, dtype=torch.float)
        
        # ONE forward pass for all states with current network
        values = self.critic(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        # Shape: [T] - values for taken actions at each timestep
        
        # ONE forward pass for next states (no grad needed)
        with torch.no_grad():
            next_states_t = torch.cat([states_t[1:], states_t[-1:]], dim=0)
            next_actions_t = torch.cat([actions_t[1:], actions_t[-1:]], dim=0)
            next_values = self.critic(next_states_t).gather(1, next_actions_t.unsqueeze(1)).squeeze(1)
        
        # Compute TD targets using current network's perspective
        targets = rewards_t + 0.95 * next_values * (1 - dones_t)
        
        # Loss is difference between current value estimate and target
        critic_loss = F.mse_loss(values, targets)
        
        # Clean gradient flow: values → critic_loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    
    #Updates the actor based off episode list of states actions and deltas
    def update_actor(self, states, actions, rewards, dones, log_probs):
        # Recompute advantages (TD errors)
        states_t = torch.tensor(np.array(states), dtype=torch.float)
        actions_t = torch.tensor(actions, dtype=torch.long)
        rewards_t = torch.tensor(rewards, dtype=torch.float)
        dones_t = torch.tensor(dones, dtype=torch.float)
    
        # Compute advantages with critic (no grad)
        with torch.no_grad():
            values = self.critic(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
            next_states_t = torch.cat([states_t[1:], states_t[-1:]], dim=0)
            next_actions_t = torch.cat([actions_t[1:], actions_t[-1:]], dim=0)
            next_values = self.critic(next_states_t).gather(1, next_actions_t.unsqueeze(1)).squeeze(1)
            advantages = rewards_t + 0.95 * next_values * (1 - dones_t) - values
        
        # Recompute log_probs fresh with current actor
        logits = self.actor(states_t)
        dist = Categorical(logits=logits)
        log_probs_fresh = dist.log_prob(actions_t)
        
        # Now clean gradient flow: actor.parameters() → logits → log_probs_fresh → actor_loss
        actor_loss = -(log_probs_fresh * advantages).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step() 
