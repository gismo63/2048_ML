import tensorflow as tf      # Deep Learning library
import numpy as np           # Handle matrices

import random                # Handling random number generation
import time                  # Handling time calculation
from skimage import transform# Help us to preprocess the frames
import sys

from collections import deque# Ordered collection with ends
import matplotlib.pyplot as plt # Display graphs

import warnings # This ignore all the warning messages that are normally printed during the training because of skiimage
warnings.filterwarnings('ignore') 

from base_game import *

def main():
    mainBoard = generateStartingBoard(BOARDHEIGHT,BOARDWIDTH)
    wasChanged = True
    score = 0
    reward = 0

    total_episodes = 50        # Total episodes
    total_test_episodes = 10     # Total test episodes
    max_steps = 10000                # Max steps per episode
    batch_size = 12 

    learning_rate = 0.3           # Learning rate
    gamma = 0.618                 # Discounting rate

    # Exploration parameters
    epsilon = 1.0                 # Exploration rate
    max_epsilon = 1.0             # Exploration probability at start
    min_epsilon = 0.01            # Minimum exploration probability 
    decay_rate = 0.01             # Exponential decay rate for exploration prob

    ### MEMORY HYPERPARAMETERS
    pretrain_length = batch_size   # Number of experiences stored in the Memory when initialized for the first time
    memory_size = 1000000          # Number of experiences the Memory can keep

    left = [1,0,0,0]
    up = [0,1,0,0]
    right = [0,0,1,0]
    down = [0,0,0,1]
    actionSpace = [left,up,right,down]

    # Instantiate memory
    memory = Memory(max_size = memory_size)


    for i in range(pretrain_length):
        # If it's the first step
        if i == 0:
            # First we need a state
            currentBoard = generateStartingBoard(BOARDHEIGHT,BOARDWIDTH)
        
        # Random action
        action = random.choice(actionSpace)
        
        # Get the rewards
        newBoard, reward, wasChanged = newBoardLogML(currentBoard, action)
        # Add experience to memory
        memory.add((currentBoard, action, reward, newBoard, wasChanged))
        
        # If we're dead
        if wasChanged:
            # First we need a state
            currentBoard = newBoard
            
            
        else:
            # Our state is now the next_state
            currentBoard = generateStartingBoard(BOARDHEIGHT,BOARDWIDTH)
    print("made it")


def DQNN(inputs,actions,targets,learningRate):

    flatten = tf.keras.layers.Flatten(inputs)
            ## --> [1152]
            
    
    fc = tf.keras.layers.dense(inputs = flatten,
                          units = tf.size(flatten)*2,
                          activation = tf.nn.relu,
                               kernel_initializer=tf.contrib.layers.xavier_initializer(),
                        name="fc1")
    
    
    output = tf.keras.layers.dense(inputs = fc, 
                                   kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                  units = 4, 
                                activation=None)

  
    # Q is our predicted Q value.
    Q = tf.reduce_sum(tf.multiply(output, actions), axis=1)
    
    
    # The loss is the difference between our predicted Q_values and the Q_target
    # Sum(Qtarget - Q)^2
    loss = tf.reduce_mean(tf.square(targets - Q))
    
    optimizer = tf.train.RMSPropOptimizer(learningRate).minimize(loss)

    return output,Q,loss,optimizer

class Memory():
    def __init__(self, max_size):
        self.buffer = deque(maxlen = max_size)
    
    def add(self, experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size),
                                size = batch_size,
                                replace = False)
        
        return [self.buffer[i] for i in index]


def generateStartingBoard(height,width):
    board = np.zeros(height,width)
    startingPos = random.sample(range(height*width),2)
    for i in startingPos:
        if random.randint(0,9):
            board[i//height][i%width] = 1
        else:
            board[i//height][i%width] = 2
    return board



def newBoardLogML(board, move):#################################################HERE
    isChanged = False
    newScore = 0
    Left = [1,0,0,0]
    Up = [0,1,0,0]
    Right = [0,0,1,0]
    Down = [0,0,0,1]
    if move == Left:
        for i in range(BOARDHEIGHT):
            edge = 0 #if combined then edge should be moved past
            for j in range(1,BOARDWIDTH):
                if board[i][j] != 0:
                    pos = j
                    while pos>edge:
                        if board[i][pos-1] == 0:
                            pos -= 1
                        elif board[i][pos-1] == board[i][j]:
                            newScore+=2**(board[i][j]+1)
                            board[i][pos-1] += 1
                            edge = pos
                            board[i][j] = 0
                            isChanged = True
                        else:
                            if pos != j:
                                board[i][pos] = board[i][j]
                                board[i][j] = 0
                                isChanged = True
                            edge = pos

                    if board[i][edge] == 0:
                        board[i][edge] = board[i][j]
                        board[i][j] = 0
                        isChanged = True
    if move == Up:
        for j in range(BOARDWIDTH):
            edge = 0 #if combined then edge should be moved past
            for i in range(1,BOARDHEIGHT):
                if board[i][j] != 0:
                    pos = i
                    while pos>edge:
                        if board[pos-1][j] == 0:
                            pos -= 1
                        elif board[pos-1][j] == board[i][j]:
                            newScore+=2**(board[i][j]+1)
                            board[pos-1][j] += 1
                            edge = pos
                            board[i][j] = 0
                            isChanged = True
                        else:
                            if pos != i:
                                board[pos][j] = board[i][j]
                                board[i][j] = 0
                                isChanged = True
                            edge = pos

                    if board[edge][j] == 0:
                        board[edge][j] = board[i][j]
                        board[i][j] = 0
                        isChanged = True
    if move == Right:
        for i in range(BOARDHEIGHT):
            edge = 0 #if combined then edge should be moved past
            for j in range(1,BOARDWIDTH):
                if board[BOARDHEIGHT-i-1][BOARDHEIGHT-j-1] != 0:
                    pos = j
                    while pos>edge:
                        if board[BOARDHEIGHT-i-1][BOARDHEIGHT-pos] == 0:
                            pos -= 1
                        elif board[BOARDHEIGHT-i-1][BOARDHEIGHT-pos] == board[BOARDHEIGHT-i-1][BOARDHEIGHT-j-1]:
                            newScore+=2**(board[BOARDHEIGHT-i-1][BOARDHEIGHT-j-1]+1)
                            board[BOARDHEIGHT-i-1][BOARDHEIGHT-pos] += 1
                            edge = pos
                            board[BOARDHEIGHT-i-1][BOARDHEIGHT-j-1] = 0
                            isChanged = True
                        else:
                            if pos != j:
                                board[BOARDHEIGHT-i-1][BOARDHEIGHT-pos-1] = board[BOARDHEIGHT-i-1][BOARDHEIGHT-j-1]
                                board[BOARDHEIGHT-i-1][BOARDHEIGHT-j-1] = 0
                                isChanged = True
                            edge = pos

                    if board[BOARDHEIGHT-i-1][BOARDHEIGHT-edge-1] == 0:
                        board[BOARDHEIGHT-i-1][BOARDHEIGHT-edge-1] = board[BOARDHEIGHT-i-1][BOARDHEIGHT-j-1]
                        board[BOARDHEIGHT-i-1][BOARDHEIGHT-j-1] = 0
                        isChanged = True
    if move == Down:
        for j in range(BOARDWIDTH):
            edge = 0 #if combined then edge should be moved past
            for i in range(1,BOARDHEIGHT):
                if board[BOARDHEIGHT-i-1][BOARDHEIGHT-j-1] != 0:
                    pos = i
                    while pos>edge:
                        if board[BOARDHEIGHT-pos][BOARDHEIGHT-j-1] == 0:
                            pos -= 1
                        elif board[BOARDHEIGHT-pos][BOARDHEIGHT-j-1] == board[BOARDHEIGHT-i-1][BOARDHEIGHT-j-1]:
                            newScore+=2**(board[BOARDHEIGHT-i-1][BOARDHEIGHT-j-1]+1)
                            board[BOARDHEIGHT-pos][BOARDHEIGHT-j-1] += 1
                            edge = pos
                            board[BOARDHEIGHT-i-1][BOARDHEIGHT-j-1] = 0
                            isChanged = True
                        else:
                            if pos != i:
                                board[BOARDHEIGHT-pos-1][BOARDHEIGHT-j-1] = board[BOARDHEIGHT-i-1][BOARDHEIGHT-j-1]
                                board[BOARDHEIGHT-i-1][BOARDHEIGHT-j-1] = 0
                                isChanged = True
                            edge = pos

                    if board[BOARDHEIGHT-edge-1][BOARDHEIGHT-j-1] == 0:
                        board[BOARDHEIGHT-edge-1][BOARDHEIGHT-j-1] = board[BOARDHEIGHT-i-1][BOARDHEIGHT-j-1]
                        board[BOARDHEIGHT-i-1][BOARDHEIGHT-j-1] = 0
                        isChanged = True
    if isChanged:
        zeroInd = []
        numZeros = 0
        for i in range(BOARDHEIGHT):
            for j in range(BOARDWIDTH):
                if not board[i][j]:
                    zeroInd.append([i,j])
                    numZeros+=1
        zeroLoc = random.randint(0,numZeros-1)
        if random.randint(0,9):
            board[zeroInd[zeroLoc][0]][zeroInd[zeroLoc][1]] = 1
        else:
            board[zeroInd[zeroLoc][0]][zeroInd[zeroLoc][1]] = 2
    return board, isChanged, newScore