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
    wasChanged = True
    score = 0
    reward = 0

    state_dim = BOARDWIDTH*BOARDHEIGHT
    action_dim = 4

    total_episodes = 5000        # Total episodes
    total_test_episodes = 10     # Total test episodes
    max_steps = 10000                # Max steps per episode
    batch_size = 64

    learningRate = 0.0001           # Learning rate
    gamma = 0.75                # Discounting rate

    # Exploration parameters
    epsilon = 1.0                 # Exploration rate
    max_epsilon = 1.0             # Exploration probability at start
    min_epsilon = 0.01            # Minimum exploration probability 
    decay_rate = 0.005             # Exponential decay rate for exploration prob

    ### MEMORY HYPERPARAMETERS
    pretrain_length = batch_size   # Number of experiences stored in the Memory when initialized for the first time
    memory_size = 1000000          # Number of experiences the Memory can keep

    left = [1,0,0,0]
    up = [0,1,0,0]
    right = [0,0,1,0]
    down = [0,0,0,1]
    actionSpace = np.array([left,up,right,down])

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
        newBoard, wasChanged, reward = newBoardLogML(currentBoard, action)
        # Add experience to memory
        memory.add((currentBoard, action, reward, newBoard, wasChanged))
        
        # If we're dead
        if wasChanged:
            # First we need a state
            currentBoard = newBoard
            
            
        else:
            # Our state is now the next_state
            currentBoard = generateStartingBoard(BOARDHEIGHT,BOARDWIDTH)
    writer = tf.summary.create_file_writer("/tensorboard/dqn/1")

    currentBoard = generateStartingBoard(BOARDHEIGHT,BOARDWIDTH)
    # Initialize the decay rate (that will use to reduce epsilon) 
    decay_step = 0

    DQNN = construct_q_network(state_dim, action_dim)

    opt = tf.keras.optimizers.RMSprop(learning_rate=learningRate)

    loss = 0

    for episode in range(total_episodes):
        with tf.GradientTape(persistent=True) as tape:
            # Set step to 0
            step = 0
            
            # Initialize the rewards of the episode
            episode_rewards = []

            while step < max_steps:
                step += 1
                
                # Increase decay_step
                decay_step +=1
                
                # Predict the action to take and take it
                action, explore_probability = predict_action(max_epsilon, min_epsilon, decay_rate, decay_step, currentBoard, actionSpace,DQNN)

                newBoard, wasChanged, reward = newBoardLogML(currentBoard, action)
                
                # Add the reward to total reward
                episode_rewards.append(reward)

                # If the game is finished
                if not wasChanged:
                    # Set step = max_steps to end the episode
                    step = max_steps

                    # Get the total reward of the episode
                    total_reward = np.sum(episode_rewards)

                    print('Episode: {}'.format(episode),
                            'Total reward: {}'.format(total_reward),
                            'Training loss: {:.4f}'.format(loss),
                            'Explore P: {:.4f}'.format(explore_probability))

                    memory.add((currentBoard, action, reward, newBoard, wasChanged))

                else:
                    # Add experience to memory
                    memory.add((currentBoard, action, reward, newBoard, wasChanged))
                    
                    # st+1 is now our current state
                    currentBoard = newBoard


                ### LEARNING PART            
                # Obtain random mini-batch from memory
                batch = memory.sample(batch_size)
                boards_mb = np.array([each[0] for each in batch], ndmin=2)
                actions_mb = np.array([each[1] for each in batch])
                rewards_mb = np.array([each[2] for each in batch]) 
                newBoards_mb = np.array([each[3] for each in batch], ndmin=2)
                changes_mb = np.array([each[4] for each in batch])

                target_Qs_batch = []

                # Get Q values for next_state 
                QsNextState = []
                QsCurrentState = []
                for i in range(0, len(batch)):
                    QsNextState.append(DQNN(tf.constant([newBoards_mb[i].flatten()])))
                    QsCurrentState.append(DQNN(tf.constant([boards_mb[i].flatten()])))
                
                
                # Set Q_target = r if the episode ends at s+1, otherwise set Q_target = r + gamma*maxQ(s', a')
                for i in range(0, len(batch)):
                    terminal = changes_mb[i]

                    # If we are in a terminal state, only equals reward
                    if not terminal:
                        target_Qs_batch.append(rewards_mb[i])
                        
                    else:
                        target = rewards_mb[i] + gamma * np.max(QsNextState[i])
                        target_Qs_batch.append(target)
                        

                targets_mb = np.array([each for each in target_Qs_batch])

                Qs = tf.math.reduce_sum(tf.multiply(QsCurrentState, actions_mb), axis=2)

                loss = tf.math.reduce_mean(tf.square(-targets_mb + Qs))

                grads = tape.gradient(loss, DQNN.trainable_variables)

                opt.apply_gradients(zip(grads, DQNN.trainable_variables))
                """
                # Write TF Summaries
                summary = sess.run(write_op, feed_dict={DQNetwork.inputs_: states_mb,
                                                DQNetwork.target_Q: targets_mb,
                                                DQNetwork.actions_: actions_mb})
                writer.add_summary(summary, episode)
                writer.flush()
                """
            """
            # Save model every 5 episodes
            if episode % 5 == 0:
                save_path = saver.save(sess, "./models/model.ckpt")
                print("Model Saved")
            """


"""
This function will do the part
With ϵ select a random action atat, otherwise select at=argmaxaQ(st,a)
"""
def predict_action(explore_start, explore_stop, decay_rate, decay_step, board, actions,NN):
    ## EPSILON GREEDY STRATEGY
    # Choose action a from state s using epsilon greedy.
    ## First we randomize a number
    exp_exp_tradeoff = np.random.rand()

    # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
    explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
    
    if (explore_probability > exp_exp_tradeoff):
        # Make a random action (exploration)
        action = random.choice(actions)
        
    else:
        # Get action from Q-network (exploitation)
        # Estimate the Qs values state
        Qs = NN(tf.constant([board.flatten()]))
        
        # Take the biggest Q value (= the best action)
        choice = np.argmax(Qs)
        action = actions[int(choice)]
                
    return action, explore_probability

"""
def DQNN(inputs):

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
    return output
"""
"""
DQNN = tf.keras.Sequential([
    tf.keras.layers.dense(units = 32,activation = tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer(),name="fc1"),
    tf.keras.layers.dense(kernel_initializer=tf.contrib.layers.xavier_initializer(),units = 4,activation=None),
])
"""

def construct_q_network(state_dim, action_dim):
    """Construct the q-network with q-values per action as output"""
    inputs = tf.keras.layers.Input(shape=(state_dim,))  # input dimension
    hidden1 = tf.keras.layers.Dense(
        state_dim, activation="relu", kernel_initializer=tf.keras.initializers.glorot_normal()
    )(inputs)
    q_values = tf.keras.layers.Dense(
        action_dim, kernel_initializer=tf.keras.initializers.glorot_normal()
    )(
        hidden1
    )

    return tf.keras.Model(inputs=inputs, outputs=[q_values])

def predQs(output,actions) :
    # Q is our predicted Q value.
    Q = tf.reduce_sum(tf.multiply(output, actions), axis=1)
    return Q
    
def mean_squared_error_loss(q_value: tf.Tensor, reward: tf.Tensor) -> tf.Tensor:
    """Compute mean squared error loss"""
    loss = 0.5 * (q_value - reward) ** 2

    return loss

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
    board = np.zeros((height,width))
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
    Left = np.array([1,0,0,0])
    Up = np.array([0,1,0,0])
    Right = np.array([0,0,1,0])
    Down = np.array([0,0,0,1])
    if (move == Left).all():
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
    if (move == Up).all():
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
    if (move == Right).all():
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
    if (move == Down).all():
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
    else:
        newScore-=10
    return board, isChanged, newScore

if __name__ == '__main__':
    main()