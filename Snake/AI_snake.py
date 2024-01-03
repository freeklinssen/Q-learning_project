import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np


pygame.init()
font = pygame.font.Font('arial.ttf', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

BLOCK_SIZE = 20
SPEED = 60

class SnakeGameAI:
    def __init__(self, number, w=640, h=480):
        self.w = w
        self.h = h 
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake game %d' %number)
        self.clock = pygame.time.Clock()
        self.reset()
        
    
    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head, 
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        
        self.score = 0
        self.food = None
        self.place_food()
        self.frame_iteration = 0 
    
    def place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE)//BLOCK_SIZE)* BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE)//BLOCK_SIZE)* BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self.place_food()
   
   
    def play_step(self, action):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
    
        self.frame_iteration += 1
        self.move(action)
        self.snake.insert(0, self.head)
        
        game_over = 0
        reward = 0
        if self.collision() or self.frame_iteration > 80*len(self.snake):
            reward = -10
            game_over = 1
            return  game_over, reward, self.score        
        
        if self.head == self.food:
            self.score += 1
            self.place_food()
            reward = 25
            print('okay!!')
        else:
            self.snake.pop()     
        
        self.update_ui()
        self.clock.tick(SPEED) #dit hoeft dan niet meer, of heel hoog zetten
        return  game_over, reward, self.score   
    
    def collision(self, pt=None):
        if pt is None:
            pt = self.head
        if pt in self.snake[1:]:
            return True
        if pt.x > self.w-BLOCK_SIZE or pt.x <0 or pt.y > self.h-BLOCK_SIZE or pt.y < 0:
            return True
        return False
        
    def update_ui(self):
        self.display.fill(BLACK)
        
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
            
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()
        
    def move(self, action):
        x= self.head.x
        y= self.head.y
       
        clock = [Direction.RIGHT, Direction.DOWN , Direction.LEFT, Direction.UP]
        idx = clock.index(self.direction)
        
        if np.array_equal(action,[1, 0, 0]): #  streigth
            new_direction = self.direction
        elif np.array_equal(action,[0, 1, 0]): # right
            new_idx = (idx+1)%4
            new_direction = clock[new_idx]
        elif np.array_equal(action,[0, 0, 1]): # left
            new_idx = (idx-1)%4
            new_direction = clock[new_idx]
        
        self.direction = new_direction       
        if self.direction == Direction.RIGHT:
           x += BLOCK_SIZE
        if self.direction == Direction.LEFT:
           x -= BLOCK_SIZE
        if self.direction == Direction.DOWN:
           y += BLOCK_SIZE
        if self.direction == Direction.UP:
           y -= BLOCK_SIZE
            
        self.head = Point(x, y)
         
        
    
        
                    


        
        
        


