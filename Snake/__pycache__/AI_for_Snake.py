import tensorflow as tf
import numpy as np
from AI_snake import SnakeGameAI, Direction, Point
import random
import matplotlib.pyplot as plt
from IPython import display 
import sys 
import os

BLOCK_SIZE = 20
load = False 
store = False
plt.ion()  

if len(sys.argv) in [3]:
    if sys.argv[1] == 'load':
        load_path = sys.argv[2]
        load = True
    elif sys.argv[1] == 'store':
        store_path = sys.argv[2]
        store = True
    else:
        load_path = sys.argv[1]
        store_path = sys.argv[2]
        load = True
        store = True

# game loop    
class Agent(): 
    def __init__(self):
        self.replay_buffer = []
        if load:
            self.model = tf.keras.models.load_model(load_path) 
            print('True')
        else:
            self.model = Snake_AI()
        
    def get_state(self, game):
        state = np.zeros(11) 
        # [danger straight, danger right, danger left,
        #  direction left, direction right, direction up, direction down,
        #  food left, food right, food up, food down]
        
        direction = game.direction
        #direction/last action
        if direction == Direction.RIGHT:
            state[3] = 1
        elif direction == Direction.LEFT:
            state[4] == 1
        elif direction == Direction.UP:   
            state[5] == 1 
        elif direction == Direction.DOWN:   
            state[6] == 1 
        
        head = game.head
        food = game.food  
        # place of food
        if head.y < food.y: #down
            state[10]=1
            
        if head.y > food.y: #up
            state[9]=1
        
        if head.x < food.x: #right
            state[8] = 1
        
        if head.x > food.x: #left 
            state[7] = 1
            
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        # danger 
        if direction == Direction.RIGHT:
            if game.collision(point_r):
                state[0] = 1
            if game.collision(point_d):
                state[1] = 1
            if game.collision(point_l):
                state[2] = 1
        elif direction == Direction.LEFT:
            if game.collision(point_l):
                state[0] = 1
            if game.collision(point_u):
                state[1] = 1
            if game.collision(point_d):
                state[2] = 1
        elif direction == Direction.UP:   
            if game.collision(point_u):
                state[0] = 1
            if game.collision(point_r):
                state[1] = 1
            if game.collision(point_l):
                state[2] = 1
        elif direction == Direction.DOWN:   
            if game.collision(point_d):
                state[0] = 1
            if game.collision(point_l):
                state[1] = 1
            if game.collision(point_r):
                state[2] = 1   
        return state.tolist() 
        
    def action(self, old_state, n_games):
        prob = 150 - n_games
        if load == True:
            prob = 0 
        action = [0,0,0] 
        if random.randint(1, 500) < prob:
            action[random.randint(0, 2)] = 1
        else:    
            input = tf.reshape(tf.convert_to_tensor(old_state, dtype=float), (-1, 11))
            action[tf.argmax(self.model(input, training =False), axis =1).numpy()[0]] = 1
        return action   
    
    def train_short(self):
        old_states, actions, rewards, new_states, game_overs = zip(*[self.replay_buffer[-1]])
        Snake_AI_trainer(self.model, old_states, actions, new_states, rewards, game_overs)   
    
    def train_long(self):
        if len(self.replay_buffer) > 10000:
            self.replay_buffer = self.replay_buffer[-10000:]
        
        if len(self.replay_buffer) > 1000:
            data = random.sample(self.replay_buffer, 1000) # list of tuples
        else:
            data = self.replay_buffer
            
        old_states, actions, rewards, new_states, game_overs = zip(*data)
        Snake_AI_trainer(self.model, old_states, actions, new_states, rewards, game_overs)    
    
    def store(self, old_state, action, new_state, reward, game_over):
        self.replay_buffer.append((old_state, action, reward, new_state, game_over))  
   
    def train(self, num_games):    
        scores = []
        mean_scores = []
        high_score = 0
        
        for i in range(num_games):
            game = SnakeGameAI(i)
            while True:
                old_state = self.get_state(game)
                action = self.action(old_state, i)
                game_over, reward, score = game.play_step(action)
                new_state = self.get_state(game)
                # store
                self.store(old_state, action, new_state, reward, game_over) 
                self.train_short()
                
                if game_over == 1:
                    game.reset()
                    scores.append(score)
                    mean_scores.append(np.mean(scores).item())
                    
                    if score > high_score:
                        high_score = score
                        
                    if score >= high_score and score > 50 and store:
                        print('store')
                        self.model.save(store_path)
                       
                    self.train_long()
                     
                    print('game',i, 'score',score)
                    display.clear_output(wait=True)
                    display.display(plt.gcf())
                    plt.gcf()
                    plt.title('training progress')
                    plt.xlabel('games')
                    plt.ylabel('score')
                    plt.plot(scores)
                    plt.plot(mean_scores)
                    plt.show(block=False)
                    plt.pause(.1)
                    break        
    
    
def Snake_AI():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(256,activation='relu', input_shape=(11,)),
        tf.keras.layers.Dense(3, activation='linear')])
    return model

optimizer = tf.keras.optimizers.Adam(learning_rate=0.00005)
loss_fn = tf.keras.losses.MeanSquaredError()
def Snake_AI_trainer(model, old_states, actions, new_states, rewards, game_overs, gamma = 0.96):
    old_states = np.vstack(old_states)
    actions = np.array(actions)
    rewards = np.array(rewards)
    new_states = np.vstack(new_states)
    game_overs = np.array(game_overs)
    
    target = rewards + gamma * np.max(model.predict(new_states), axis=1) * (1 - game_overs)
    
    with tf.GradientTape() as tape:
        predictions = model(old_states, training=True)
        targets = predictions.numpy()
        for idx in range(len(targets)):
            targets[idx][np.argmax(actions[idx]).item()] = target[idx]
        loss = loss_fn(targets, predictions)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

if __name__ == '__main__':

    num_games = 200
    agent = Agent()
    agent.train(num_games)
            