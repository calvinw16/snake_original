import torch
import random
import numpy as np
import os
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Actor, Critic,  ActorCritic
from helper import plot
from torch.distributions import Categorical

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.003

class Agent:

    def __init__(self, checkpoint_path=False):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.actor = Actor(11, 256, 3)
        self.critic = Critic(11, 256, 3)
       

        if checkpoint_path:
            if os.path.exists("model/actor.pth"):
                print("LOAD WEIGHTS")
                state_dict = torch.load("model/actor.pth", map_location=torch.device("mps"))
                self.actor.load_state_dict(state_dict)
                if os.path.exists("model/critic.pth"):
                    state_dict_crit = torch.load("model/critic.pth", map_location=torch.device("mps"))
                    self.critic.load_state_dict(state_dict_crit)
            else:
                print("No checkpoint found at 'model/actor.pth', starting fresh.")

    
    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        return np.array(state, dtype=int)

    #Need to change to get action from policy network instead
    def get_action_idx(self, state):
        state_t = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        logits = self.actor(state_t)
        dist = Categorical(logits=logits)

        action = dist.sample() #Returns index of chosen category
        action_idx = action.item()

        final_move = [0,0,0]
        final_move[action_idx] = 1
        log_prob = dist.log_prob(action)

        return action_idx, final_move, log_prob

    #Get value from critic
    #Q(s,a) value function
    def get_value(self, state, action_idx):
        state_t = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        q_values = self.critic(state_t)
        q_sa = q_values[0,action_idx]
        return q_sa
    


    


#Need to change
def train():
    

    agent = Agent(checkpoint_path=True)
    game = SnakeGameAI()
    actor_critic = ActorCritic(agent, agent.actor, agent.critic)

    print(agent.actor is actor_critic.actor)
    print(agent.critic is actor_critic.critic)  

    episode_rewards = actor_critic.execute(episodes = 100000, game=game)

    #Save model weights
    agent.critic.save("critic.pth")
    agent.actor.save("actor.pth")

    

    
    
    


if __name__ == '__main__':
    train()